"""
ca_ceph/router.py
─────────────────
PA-cephalogram landmark detection + asymmetry measurements, ported from the
CA Ceph project (AALNet). Preprocessing, the fold ensemble, measurements and
overlays are reused unchanged from ca_ceph/aalnet/scripts/serve.py.

Models are loaded once at startup (see load_models, called from api.py). If
torch/timm or the weights are missing, the rest of the API still starts and
these endpoints return 503.

Environment:
    CA_CEPH_WEIGHTS_DIR  folder holding fold*/checkpoint_best.pth
                         (default: ca_ceph/weights, which ships fold0 only)
    CA_CEPH_FOLDS        comma-separated folds to ensemble, e.g. "0" or "0,1"
                         (default: every fold found; each fold is ~300 MB RAM)
    CA_CEPH_DEVICE       cpu | mps | cuda (default: best available)

Endpoints:
    POST /analyze/ca-ceph        multipart "file" (+ optional form field
                                 pixel_spacing, mm/px) → 33 landmarks +
                                 asymmetry measurements (JSON)
    POST /analyze/ca-ceph-image  multipart "file" (+ form fields style=
                                 points|tracing, show_names) → JPEG overlay

Research use only: not a medical device, not for clinical decisions.
"""

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from starlette.concurrency import run_in_threadpool

HERE = Path(__file__).resolve().parent
WEIGHTS_DIR = Path(os.environ.get("CA_CEPH_WEIGHTS_DIR", HERE / "weights"))
MAX_UPLOAD_BYTES = 30 * 1024 * 1024

router = APIRouter(tags=["ca-ceph"])

STATE: dict = {"ensemble": None, "error": None}


def _device():
    import torch

    if os.environ.get("CA_CEPH_DEVICE"):
        return torch.device(os.environ["CA_CEPH_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_models() -> None:
    """Load the fold ensemble. Never raises: on failure the reason is kept
    in STATE["error"] and the endpoints answer 503."""
    try:
        from .aalnet.scripts.serve import Ensemble

        folds_env = os.environ.get("CA_CEPH_FOLDS", "").strip()
        folds = [int(f) for f in folds_env.split(",") if f.strip()] or None
        device = _device()
        print(f"[ca-ceph] loading folds {folds or 'all'} from {WEIGHTS_DIR} on {device}")
        STATE.update(ensemble=Ensemble(WEIGHTS_DIR, folds, device), error=None)
    except Exception as e:  # noqa: BLE001
        STATE["error"] = f"{type(e).__name__}: {e}"
        print(f"[ca-ceph] WARNING: models not ready: {STATE['error']}")


def status() -> dict:
    """Summary for the main /health endpoint."""
    ens = STATE["ensemble"]
    return {
        "device": str(ens.device) if ens else None,
        "folds": list(ens.models) if ens else [],
        "error": STATE["error"],
    }


async def _run(file: UploadFile):
    """Upload → (gray image, preprocessing params, mean coords, per-fold coords),
    coords in original-image pixels."""
    if STATE["ensemble"] is None:
        raise HTTPException(503, f"ca-ceph models not loaded: {STATE['error']}")

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image larger than 30 MB")
    # Inference takes 10-20 s on CPU: run it in a worker thread so the event
    # loop keeps serving every other route meanwhile. Ensemble.predict holds
    # its own lock, so CA Ceph requests still run one at a time (bounded RAM).
    return await run_in_threadpool(_infer, data)


def _infer(data: bytes):
    from .aalnet.scripts.serve import preprocess
    from .aalnet.utils.coord_utils import coords_to_original

    ens = STATE["ensemble"]
    gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise HTTPException(400, "could not decode the uploaded file as an image")
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    tensor, params = preprocess(gray)
    mean_lb, per_fold_lb = ens.predict(tensor)
    coords = coords_to_original(mean_lb.astype(np.float64), params)
    per_fold = np.stack([coords_to_original(c.astype(np.float64), params)
                         for c in per_fold_lb])
    return gray, params, coords, per_fold


@router.post("/analyze/ca-ceph")
async def analyze_ca_ceph(
    file: UploadFile = File(..., description="Original PA cephalogram (JPG/PNG/TIFF)"),
    pixel_spacing: Optional[float] = Form(
        None, description="mm per pixel of the uploaded image; "
                          "measurements are in px if omitted"),
):
    from .aalnet.scripts.serve import LANDMARK_NAMES, measure

    gray, params, coords, per_fold = await _run(file)
    # Spread between fold models: a rough per-landmark uncertainty
    # (always 0 when only one fold is loaded).
    spread = np.linalg.norm(per_fold - coords[None], axis=-1).mean(axis=0)
    s = pixel_spacing if pixel_spacing is not None else 1.0
    return {
        "image_size": {"width": gray.shape[1], "height": gray.shape[0]},
        "roi_bbox": params["bbox"],
        "folds_used": list(STATE["ensemble"].models),
        "landmarks": [
            {"index": i, "name": LANDMARK_NAMES[i],
             "x": round(float(x), 2), "y": round(float(y), 2),
             "fold_spread": round(float(spread[i]) * s, 3)}
            for i, (x, y) in enumerate(coords)
        ],
        "measurements": measure(coords, pixel_spacing),
    }


@router.post("/analyze/ca-ceph-image", response_class=Response,
             responses={200: {"content": {"image/jpeg": {}}}})
async def analyze_ca_ceph_image(
    file: UploadFile = File(..., description="Original PA cephalogram (JPG/PNG/TIFF)"),
    show_names: bool = Form(False, description="Label points with full landmark names"),
    style: str = Form("points", pattern="^(points|tracing)$",
                      description="points: numbered landmarks + midlines; "
                                  "tracing: cephalometric tracing through the landmarks"),
):
    gray, _, coords, _ = await _run(file)
    jpeg = await run_in_threadpool(_render, gray, coords, show_names, style)
    return Response(jpeg, media_type="image/jpeg")


def _render(gray, coords, show_names: bool, style: str) -> bytes:
    from .aalnet.scripts.serve import draw_overlay, draw_tracing

    draw = draw_tracing if style == "tracing" else draw_overlay
    ok, buf = cv2.imencode(".jpg", draw(gray, coords, show_names),
                           [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(500, "jpeg encode failed")
    return buf.tobytes()
