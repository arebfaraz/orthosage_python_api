"""HTTP inference API for AALNet.

Upload an original PA cephalogram; the server applies the same preprocessing
as the training data (Otsu skull ROI crop + letterbox to 800x800), runs the
fold models, averages their landmark coordinates, maps them back to the
original image and computes the asymmetry measurements.

Research use only: not a medical device, not for clinical decisions.

Endpoints:
    GET  /health            loaded folds and device
    GET  /landmarks         the 33 landmark names in model order
    POST /predict           landmarks (original pixels) + asymmetry measurements (JSON)
    POST /predict/overlay   the radiograph with predicted landmarks and midlines (JPEG)
    GET  /docs              interactive Swagger UI

Usage (run from the repository root):
    python -m aalnet.scripts.serve --weights_dir weights/lambda02 --port 8000
"""

from __future__ import annotations  # orthosage: allow 'X | None' hints on Python < 3.10

import argparse
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from ..datasets.landmark_dataset import IMAGENET_MEAN, IMAGENET_STD
from ..models.aalnet import AALNet
from ..preprocess.crop_and_letterbox import letterbox_resize_with_params
from ..preprocess.roi_detect import build_threshold_mask, extract_bbox_from_mask
from ..utils.clinical_metrics import (
    compute_canting,
    compute_cg_ans,
    compute_dental_deviation,
    compute_lowerface_deviation,
    compute_midface_deviation,
    compute_msr,
)
from ..utils.coord_utils import coords_to_original

TARGET_SIZE = 800

# Model output order (README "The 33 landmarks").
LANDMARK_NAMES = [
    'Antegonion right', 'Antegonion left', 'Anterior nasal spine', 'Menton',
    'Right maxillary 6 crown', 'Right maxillary 6 root',
    'Right maxillary 1 crown', 'Right maxillary 1 root',
    'Left maxillary 1 crown', 'Left maxillary 1 root',
    'Left maxillary 6 crown', 'Left maxillary 6 root',
    'Left mandibular 6 crown', 'Left mandibular 6 root',
    'Left mandibular 1 crown', 'Left mandibular 1 root',
    'Right mandibular 1 crown', 'Right mandibular 1 root',
    'Right mandibular 6 crown', 'Right mandibular 6 root',
    'Crista galli',
    'Zygomatico-maxillary suture right', 'Zygomatico-maxillary suture left',
    'Jugal process left', 'Jugal process right',
    'Zygoma right', 'Zygoma left', 'Mastoid right', 'Mastoid left',
    'Latero-orbitale right', 'Latero-orbitale left',
    'Upper dental midline', 'Lower dental midline',
]

# Defaults of roi_detect.py.
ROI_ARGS = SimpleNamespace(
    border_percent=0.04, thresh_delta=0, thresh_scale=2.0 / 3.0, margin_ratio=0.12,
    min_area_ratio=0.003, center_weight=0.35,
    min_bbox_width_ratio=0.18, min_bbox_height_ratio=0.25,
)


# ── Preprocessing (mirrors roi_detect.py + crop_and_letterbox.py) ───

def detect_roi(gray: np.ndarray) -> tuple[int, int, int, int]:
    h, w = gray.shape
    mask, _ = build_threshold_mask(
        gray=gray, thresh_mode='otsu', thresh_value=20, thresh_invert=False,
        border_percent=ROI_ARGS.border_percent, thresh_delta=ROI_ARGS.thresh_delta,
        thresh_scale=ROI_ARGS.thresh_scale, thresh_band=0,
    )
    x1, y1, x2, y2, _ = extract_bbox_from_mask(
        mask=mask, image_shape=(h, w),
        min_area_ratio=ROI_ARGS.min_area_ratio, margin_ratio=ROI_ARGS.margin_ratio,
        border_percent=ROI_ARGS.border_percent, center_weight=ROI_ARGS.center_weight,
        min_bbox_width_ratio=ROI_ARGS.min_bbox_width_ratio,
        min_bbox_height_ratio=ROI_ARGS.min_bbox_height_ratio,
        reject_ruler_like=True,
        reject_top_border_ratio=0.0, reject_bottom_border_ratio=0.02,
        reject_left_border_ratio=0.03, reject_right_border_ratio=0.03,
    )
    return x1, y1, x2, y2


def preprocess(gray: np.ndarray) -> tuple[torch.Tensor, dict]:
    """Original grayscale image -> [1, 3, 800, 800] tensor and inverse-mapping params."""
    h, w = gray.shape
    bx1, by1, bx2, by2 = detect_roi(gray)
    # Same clamping and slicing as crop_and_letterbox.py.
    bx1 = max(0, min(bx1, w - 1))
    by1 = max(0, min(by1, h - 1))
    bx2 = max(bx1 + 1, min(bx2, w))
    by2 = max(by1 + 1, min(by2, h))
    crop = gray[by1:by2, bx1:bx2]

    lb, lb_params = letterbox_resize_with_params(crop, TARGET_SIZE)
    # Training images were stored as JPEG (quality 95); reproduce that round trip.
    lb = cv2.imdecode(cv2.imencode('.jpg', lb, [cv2.IMWRITE_JPEG_QUALITY, 95])[1],
                      cv2.IMREAD_GRAYSCALE)

    img = np.stack([lb, lb, lb], axis=-1).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

    params = {
        'bbox': [bx1, by1, bx2, by2],
        'scale': lb_params['scale'],
        'pad_top': lb_params['pad_top'],
        'pad_left': lb_params['pad_left'],
    }
    return tensor, params


# ── Model ensemble ──────────────────────────────────────────────────

class Ensemble:
    def __init__(self, weights_dir: Path, folds: list[int] | None, device: torch.device):
        self.device = device
        self.models = {}
        fold_dirs = sorted(weights_dir.glob('fold*'))
        if folds is not None:
            fold_dirs = [d for d in fold_dirs if int(d.name[4:]) in folds]
        for d in fold_dirs:
            ckpt = torch.load(d / 'checkpoint_best.pth', map_location='cpu', weights_only=True)
            model = AALNet(num_landmarks=len(LANDMARK_NAMES), pretrained=False)
            model.load_state_dict(ckpt['model'])
            self.models[d.name] = model.to(device).eval()
            print(f"  Loaded {d.name} (epoch {ckpt.get('epoch')}, best val MRE {ckpt.get('best_mre')})")
        if not self.models:
            raise FileNotFoundError(f'No fold*/checkpoint_best.pth under {weights_dir}')
        self.lock = threading.Lock()

    @torch.no_grad()
    def predict(self, tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mean coords [L, 2], per-fold coords [F, L, 2]) in letterbox pixels."""
        x = tensor.to(self.device)
        per_fold = []
        with self.lock:
            for model in self.models.values():
                per_fold.append(model.get_coordinates(model(x))[0].float().cpu().numpy())
                # Full-resolution activations are large; release them between models
                # so the allocator cache does not push small-memory machines into swap.
                if self.device.type == 'mps':
                    torch.mps.empty_cache()
                elif self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        per_fold = np.stack(per_fold)
        return per_fold.mean(axis=0), per_fold


# ── Measurements ────────────────────────────────────────────────────

def measure(coords: np.ndarray, spacing: float | None) -> dict:
    """Asymmetry measurements from predicted landmarks (original pixels).

    Signed distances: positive = patient right, negative = patient left.
    """
    s = spacing if spacing is not None else 1.0
    unit = 'mm' if spacing is not None else 'px'

    msr_pt, msr_dir = compute_msr(coords)
    cg_pt, cg_dir = compute_cg_ans(coords)
    lf = compute_lowerface_deviation(coords, msr_pt, msr_dir)
    mf = compute_midface_deviation(coords, msr_pt, msr_dir)
    dental = compute_dental_deviation(coords, msr_pt, msr_dir)
    cant = compute_canting(coords, msr_pt, msr_dir)

    def tilt_deg(d):
        # Angle of a midline from the image vertical.
        d = d[0] if d[0, 1] >= 0 else -d[0]
        return float(np.degrees(np.arctan2(d[0], d[1])))

    def val(arr, direction=None):
        out = {'value': round(float(arr[0]) * s, 3)}
        if direction is not None:
            out['side'] = {'R': 'patient right', 'L': 'patient left', 'C': 'centered'}[str(direction[0])]
        return out

    return {
        'unit': unit,
        'midlines': {
            'msr': {'point': msr_pt[0].round(2).tolist(), 'direction': msr_dir[0].round(5).tolist(),
                    'tilt_from_vertical_deg': round(tilt_deg(msr_dir), 3)},
            'cg_ans': {'point': cg_pt[0].round(2).tolist(), 'direction': cg_dir[0].round(5).tolist(),
                       'tilt_from_vertical_deg': round(tilt_deg(cg_dir), 3)},
            'angle_between_deg': round(abs(tilt_deg(msr_dir) - tilt_deg(cg_dir)), 3),
        },
        'lowerface_deviation': val(lf['distance'], lf['direction']),
        'midface_asymmetry': val(mf['asymmetry'], mf['direction']),
        'dental_upper_midline_deviation': val(dental['upper_midline_dist'], dental['upper_midline_dir']),
        'dental_crown_midpoint_deviation': val(dental['crown_midpoint_dist'], dental['crown_midpoint_dir']),
        'occlusal_canting': val(cant['diff'], cant['direction']),
    }


def draw_overlay(gray: np.ndarray, coords: np.ndarray, show_names: bool) -> np.ndarray:
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = gray.shape
    r = max(3, int(round(max(h, w) / 400)))
    fs = max(h, w) / 2500

    # Midlines drawn across the full image height.
    for (pt, d), color in (
        (compute_msr(coords), (255, 200, 0)),     # MSR: cyan
        (compute_cg_ans(coords), (0, 200, 255)),  # Cg-ANS: orange
    ):
        p, d = pt[0], d[0]
        if abs(d[1]) > 1e-6:
            t0, t1 = (0 - p[1]) / d[1], (h - 1 - p[1]) / d[1]
            a, b = p + t0 * d, p + t1 * d
            cv2.line(vis, tuple(int(v) for v in a), tuple(int(v) for v in b), color, max(1, r // 2))

    for i, (x, y) in enumerate(coords):
        c = (int(round(x)), int(round(y)))
        cv2.circle(vis, c, r, (0, 0, 255), -1)
        label = f'{i} {LANDMARK_NAMES[i]}' if show_names else str(i)
        cv2.putText(vis, label, (c[0] + r + 2, c[1] - r), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (255, 255, 255), max(1, r // 3), cv2.LINE_AA)
    return vis


# ── Tracing-style overlay ───────────────────────────────────────────
# Every line passes through predicted landmarks only; no anatomy is invented.

TRACE_LINE = (255, 60, 20)     # blue (BGR)
TRACE_REF = (255, 160, 90)     # light blue: bilateral reference lines
TRACE_POINT = (60, 230, 60)    # green
TRACE_LABEL = (0, 230, 255)    # yellow

TRACE_LABELS = {
    0: 'Ag', 1: 'Ag', 2: 'ANS', 3: 'Me', 20: 'Cg', 21: 'ZMS', 22: 'ZMS',
    23: 'J', 24: 'J', 25: 'Za', 26: 'Za', 27: 'Ms', 28: 'Ms', 29: 'Lo', 30: 'Lo',
}

# (crown, root) per tooth.
TEETH = [(4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19)]

# Bilateral reference lines of PA analysis: latero-orbitale, zygoma, jugal, antegonion.
REF_PAIRS = [(29, 30), (25, 26), (24, 23), (0, 1)]


def catmull_rom(points: np.ndarray, samples: int = 24) -> np.ndarray:
    """Smooth curve through the given points (centripetal Catmull-Rom)."""
    p = np.vstack([2 * points[0] - points[1], points, 2 * points[-1] - points[-2]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        t0 = 0.0
        t1 = t0 + np.linalg.norm(p1 - p0) ** 0.5 + 1e-6
        t2 = t1 + np.linalg.norm(p2 - p1) ** 0.5 + 1e-6
        t3 = t2 + np.linalg.norm(p3 - p2) ** 0.5 + 1e-6
        for t in np.linspace(t1, t2, samples, endpoint=False):
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    out.append(points[-1])
    return np.array(out)


def draw_tracing(gray: np.ndarray, coords: np.ndarray, show_names: bool) -> np.ndarray:
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    size = max(gray.shape)
    lw = max(2, round(size / 600))
    r = max(3, round(size / 300))
    fs = max(0.4, size / 1600)
    aa = cv2.LINE_AA

    def pt(i):
        return tuple(int(round(v)) for v in coords[i])

    def polyline(idx, color, thickness, smooth=False):
        p = coords[idx].astype(np.float64)
        if smooth and len(p) >= 3:
            p = catmull_rom(p)
        cv2.polylines(vis, [np.round(p).astype(np.int32)], False, color, thickness, aa)

    # Bilateral reference lines (thin).
    for a, b in REF_PAIRS:
        cv2.line(vis, pt(a), pt(b), TRACE_REF, max(1, lw // 2), aa)

    # Mandibular border Ag(R) - Me - Ag(L).
    polyline([0, 3, 1], TRACE_LINE, lw, smooth=True)
    # Facial midline Cg - ANS - upper/lower dental midline - Me.
    polyline([20, 2, 31, 32, 3], TRACE_LINE, lw)
    # Lateral orbit - zygoma - jugal process, each side (straight: a spline
    # through three widely spaced points overshoots outside the skull).
    polyline([29, 25, 24], TRACE_LINE, lw)
    polyline([30, 26, 23], TRACE_LINE, lw)
    # Teeth: crown-root axes, plus the occlusal plane between upper first molars.
    for crown, root in TEETH:
        cv2.line(vis, pt(crown), pt(root), TRACE_LINE, lw + 1, aa)
    cv2.line(vis, pt(4), pt(10), TRACE_LINE, lw, aa)

    # MSR (perpendicular bisector of latero-orbitale), dashed over the full height.
    msr_pt, msr_dir = compute_msr(coords)
    p, d = msr_pt[0], msr_dir[0]
    if abs(d[1]) > 1e-6:
        a = p + (0 - p[1]) / d[1] * d
        b = p + (gray.shape[0] - 1 - p[1]) / d[1] * d
        n = 60
        for k in range(0, n, 2):
            s, e = a + (b - a) * k / n, a + (b - a) * (k + 1) / n
            cv2.line(vis, tuple(int(v) for v in s), tuple(int(v) for v in e), TRACE_REF, lw, aa)

    for i in range(len(coords)):
        cv2.circle(vis, pt(i), r, TRACE_POINT, -1, aa)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(coords)):
        label = LANDMARK_NAMES[i] if show_names else TRACE_LABELS.get(i)
        if not label:
            continue
        x, y = pt(i)
        (tw, th), _ = cv2.getTextSize(label, font, fs, max(1, lw // 2))
        # Labels sit outward from the midline; ZMS goes below its point, since it
        # lies close to latero-orbitale.
        outward_left = x < msr_pt[0][0]
        tx = x - r - 3 - tw if outward_left else x + r + 3
        ty = y + r + th + 3 if i in (21, 22) else y - r - 3
        cv2.putText(vis, label, (tx, ty), font, fs, (0, 0, 0), max(1, lw // 2) + 3, aa)
        cv2.putText(vis, label, (tx, ty), font, fs, TRACE_LABEL, max(1, lw // 2), aa)
    return vis


# ── App ─────────────────────────────────────────────────────────────

def create_app(weights_dir: Path, folds: list[int] | None, device: torch.device) -> FastAPI:
    state = {}

    @asynccontextmanager
    async def lifespan(_app):
        print(f'Loading models from {weights_dir} on {device}')
        state['ensemble'] = Ensemble(weights_dir, folds, device)
        yield

    app = FastAPI(
        title='AALNet API',
        description='33-landmark detection and asymmetry measurements on PA cephalograms. '
                    'Research use only; not for clinical diagnosis or treatment decisions.',
        lifespan=lifespan,
    )

    async def run(file: UploadFile):
        data = await file.read()
        gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise HTTPException(400, 'Could not decode the uploaded file as an image')
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        tensor, params = preprocess(gray)
        mean_lb, per_fold_lb = state['ensemble'].predict(tensor)
        coords = coords_to_original(mean_lb.astype(np.float64), params)
        per_fold = np.stack([coords_to_original(c.astype(np.float64), params) for c in per_fold_lb])
        return gray, params, coords, per_fold

    @app.get('/health')
    def health():
        ens = state['ensemble']
        return {'status': 'ok', 'device': str(ens.device), 'folds': list(ens.models)}

    @app.get('/landmarks')
    def landmarks():
        return [{'index': i, 'name': n} for i, n in enumerate(LANDMARK_NAMES)]

    @app.post('/predict')
    async def predict(
        file: UploadFile = File(..., description='Original PA cephalogram (JPG/PNG/TIFF)'),
        pixel_spacing: float | None = Form(None, description='mm per pixel of the uploaded image; '
                                                             'measurements are in px if omitted'),
    ):
        gray, params, coords, per_fold = await run(file)
        # Spread between fold models: a rough per-landmark uncertainty.
        spread = np.linalg.norm(per_fold - coords[None], axis=-1).mean(axis=0)
        s = pixel_spacing if pixel_spacing is not None else 1.0
        return {
            'image_size': {'width': gray.shape[1], 'height': gray.shape[0]},
            'roi_bbox': params['bbox'],
            'folds_used': list(state['ensemble'].models),
            'landmarks': [
                {'index': i, 'name': LANDMARK_NAMES[i],
                 'x': round(float(x), 2), 'y': round(float(y), 2),
                 'fold_spread': round(float(spread[i]) * s, 3)}
                for i, (x, y) in enumerate(coords)
            ],
            'measurements': measure(coords, pixel_spacing),
        }

    @app.post('/predict/overlay', response_class=Response,
              responses={200: {'content': {'image/jpeg': {}}}})
    async def predict_overlay(
        file: UploadFile = File(..., description='Original PA cephalogram (JPG/PNG/TIFF)'),
        show_names: bool = Form(False, description='Label points with full landmark names'),
        style: Literal['points', 'tracing'] = Form(
            'points', description='points: numbered landmarks + midlines; '
                                  'tracing: cephalometric tracing through the landmarks'),
    ):
        gray, _, coords, _ = await run(file)
        draw = draw_tracing if style == 'tracing' else draw_overlay
        vis = draw(gray, coords, show_names)
        ok, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(buf.tobytes(), media_type='image/jpeg')

    return app


def main():
    parser = argparse.ArgumentParser(description='AALNet inference API')
    parser.add_argument('--weights_dir', type=str, default='weights/lambda02',
                        help='Directory containing fold*/checkpoint_best.pth')
    parser.add_argument('--folds', type=int, nargs='+', default=None,
                        help='Folds to ensemble (default: all found)')
    parser.add_argument('--device', type=str, default=None,
                        help='cuda, mps or cpu (default: best available)')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    app = create_app(Path(args.weights_dir), args.folds, device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
