"""
gingivitis/router.py
────────────────────
Intraoral-photo inference, ported from BinaModel_project/src/api/app.py:
upload one intraoral photo, get every trained YOLO model's findings back.

Models are listed in gingivitis/models.yaml and loaded once at startup
(see load_models, called from api.py). If torch/ultralytics or the weights
are missing, the rest of the API still starts and these endpoints return 503.

Endpoints:
    POST /analyze-gingivitis        multipart "file" → findings JSON
        ?models=gingivitis,recession   run a subset of the registry
        ?conf=0.25                     override every model's threshold
        ?annotate=true                 add base64 JPEG with boxes drawn
        ?masks=true                    add polygons from seg models
        ?labels=short|full|none        tag style on the annotated image
    POST /analyze-gingivitis-image  same params → annotated JPEG
                                    (masks on by default)
"""

import base64
import io
import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

HERE = Path(__file__).resolve().parent
API_CFG = Path(os.environ.get("GINGIVITIS_CONFIG", HERE / "models.yaml"))
WEIGHTS_DIR = Path(os.environ.get("GINGIVITIS_WEIGHTS_DIR", HERE))
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

router = APIRouter(tags=["gingivitis"])


def _resolve_device(device: str) -> str:
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LoadedModel:
    """One YOLO checkpoint plus its API settings."""

    def __init__(self, key: str, spec: dict, device: str):
        from ultralytics import YOLO

        weights = WEIGHTS_DIR / spec["weights"]
        if not weights.exists():
            raise FileNotFoundError(f"[{key}] weights not found: {weights}")
        self.key = key
        self.weights = weights
        self.model = YOLO(str(weights))
        self.task = self.model.task
        self.names = self.model.names
        self.imgsz = spec.get("imgsz", 640)
        self.conf = spec.get("conf", 0.25)
        self.iou = spec.get("iou", 0.45)
        self.enabled = spec.get("enabled", True)
        self.device = device
        keep = spec.get("classes")
        self.class_ids = (
            [i for i, n in self.names.items() if n in keep] if keep else None
        )
        # Ultralytics predictors are not thread-safe; FastAPI runs sync
        # endpoints in a threadpool, so serialize per model.
        self.lock = threading.Lock()

    def warmup(self) -> None:
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.predict(dummy, self.conf)

    def predict(self, img_bgr: np.ndarray, conf: float):
        with self.lock:
            return self.model.predict(
                source=img_bgr, imgsz=self.imgsz, conf=conf, iou=self.iou,
                classes=self.class_ids, device=self.device, verbose=False,
            )[0]

    def info(self) -> dict:
        kept = self.class_ids if self.class_ids is not None else list(self.names)
        return {
            "weights": self.weights.name,
            "task": self.task,
            "classes": [self.names[i] for i in kept],
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "enabled": self.enabled,
        }


STATE: dict = {"models": {}, "device": None, "anatomy": set(), "error": None}


def load_models() -> None:
    """Load + warm every model in models.yaml. Never raises: on failure the
    reason is kept in STATE["error"] and the endpoints answer 503."""
    try:
        with open(API_CFG, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        device = _resolve_device(str(cfg.get("device", "auto")))
        models = {}
        for key, spec in cfg["models"].items():
            try:
                m = LoadedModel(key, spec, device)
                m.warmup()
                models[key] = m
                print(f"[gingivitis] loaded [{key}] {m.task} "
                      f"{list(m.names.values())} on {device}")
            except FileNotFoundError as e:
                print(f"[gingivitis] skipped {e}")
        if not models:
            raise RuntimeError(f"no models could be loaded from {API_CFG}")
        STATE.update(models=models, device=device,
                     anatomy=set(cfg.get("anatomy_classes", [])), error=None)
    except Exception as e:  # noqa: BLE001
        STATE["error"] = f"{type(e).__name__}: {e}"
        print(f"[gingivitis] WARNING: models not ready: {STATE['error']}")


def status() -> dict:
    """Summary for the main /health endpoint."""
    return {
        "device": STATE["device"],
        "models": {k: m.info() for k, m in STATE["models"].items()},
        "error": STATE["error"],
    }


def _decode_image(data: bytes) -> np.ndarray:
    """Bytes → BGR array, honoring EXIF rotation from phone cameras."""
    try:
        pil = Image.open(io.BytesIO(data))
        pil = ImageOps.exif_transpose(pil).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "file is not a readable image")
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _detections(model: LoadedModel, result, w: int, h: int,
                with_masks: bool) -> list:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    polys = None
    if with_masks and result.masks is not None:
        polys = result.masks.xy

    out = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        det = {
            "class": model.names[clss[i]],
            "confidence": round(float(confs[i]), 4),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "bbox_norm": [round(x1 / w, 5), round(y1 / h, 5),
                          round(x2 / w, 5), round(y2 / h, 5)],
            "model": model.key,
        }
        if polys is not None:
            det["polygon"] = [[round(float(x), 1), round(float(y), 1)]
                              for x, y in polys[i]]
        out.append(det)
    return out


_PALETTE = [(56, 56, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
            (10, 249, 72), (23, 204, 146), (134, 219, 61), (211, 188, 0)]

# Fixed colors (BGR) and one-letter tags so the same condition always
# looks the same across images; unknown classes fall back to the palette.
_CLASS_STYLE = {
    "gingivitis":    ((60, 60, 230), "G"),
    "caries":        ((0, 165, 255), "C"),
    "plaque":        ((0, 215, 255), "P"),
    "recession":     ((200, 90, 220), "R"),
    "discoloration": ((180, 130, 70), "D"),
    "ulcer":         ((80, 200, 60), "U"),
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _style(name: str, i: int) -> tuple:
    return _CLASS_STYLE.get(
        name, (_PALETTE[i % len(_PALETTE)], name[:2].capitalize()))


def _text_tag(canvas, text, x, y, color, scale, tw, above=False) -> None:
    """Text on a solid color background, top-left corner at (x, y).
    above=True sits the tag on top of y (just outside a box) when it fits."""
    (w, h), base = cv2.getTextSize(text, _FONT, scale, tw)
    pad = max(2, int(h * 0.3))
    if above and y - (h + base + 2 * pad) >= 0:
        y -= h + base + 2 * pad
    H, W = canvas.shape[:2]
    x = min(max(x, 0), max(W - w - 2 * pad, 0))
    y = min(max(y, 0), max(H - h - base - 2 * pad, 0))
    cv2.rectangle(canvas, (x, y), (x + w + 2 * pad, y + h + base + 2 * pad),
                  color, -1)
    cv2.putText(canvas, text, (x + pad, y + pad + h), _FONT, scale,
                (255, 255, 255), tw, cv2.LINE_AA)


def _annotate(img_bgr: np.ndarray, dets: list, labels: str = "short") -> bytes:
    """Draw boxes (and mask polygons, when present) → JPEG bytes.

    labels: "short" → "G 64%" tags + legend strip, "full" → "gingivitis 64%",
            "none"  → boxes only (legend still shown).
    """
    canvas = img_bgr.copy()
    H, W = canvas.shape[:2]
    thick = max(2, round(max(H, W) / 500))
    scale = max(0.35, max(H, W) / 2600)
    tw = max(1, round(scale * 1.5))

    styles: dict = {}
    for d in dets:
        if d["class"] not in styles:
            styles[d["class"]] = _style(d["class"], len(styles))

    overlay = canvas.copy()
    for d in dets:
        if d.get("polygon"):
            pts = np.array(d["polygon"], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], styles[d["class"]][0])
    canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)

    for d in dets:
        color, short = styles[d["class"]]
        x1, y1, x2, y2 = (int(v) for v in d["bbox"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thick)
    # Tags drawn after all boxes so no box line crosses a tag.
    if labels != "none":
        for d in dets:
            color, short = styles[d["class"]]
            name = short if labels == "short" else d["class"]
            x1, y1 = int(d["bbox"][0]), int(d["bbox"][1])
            _text_tag(canvas, f'{name} {d["confidence"] * 100:.0f}%',
                      x1, y1, color, scale, tw, above=True)

    if styles:
        canvas = _legend(canvas, dets, styles, scale, tw)
    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def _legend(canvas, dets, styles, scale, tw) -> np.ndarray:
    """Append a white strip below the image: "[G] gingivitis ×12  [C] ..."."""
    counts = defaultdict(int)
    for d in dets:
        counts[d["class"]] += 1
    (_, th), base = cv2.getTextSize("Ag", _FONT, scale, tw)
    strip_h = th + base + int(th * 1.6)
    strip = np.full((strip_h, canvas.shape[1], 3), 255, dtype=np.uint8)
    x, y_text = int(th * 0.6), int(th * 0.8) + th
    for name, (color, short) in styles.items():
        (sw, _), _ = cv2.getTextSize(short, _FONT, scale, tw)
        _text_tag(strip, short, x, int(th * 0.4), color, scale, tw)
        x += sw + int(th * 1.2)
        text = f"{name} x{counts[name]}"
        cv2.putText(strip, text, (x, y_text), _FONT, scale, (40, 40, 40),
                    tw, cv2.LINE_AA)
        (tw_px, _), _ = cv2.getTextSize(text, _FONT, scale, tw)
        x += tw_px + int(th * 1.5)
    return np.vstack([canvas, strip])


def _run(file: UploadFile, models: Optional[str], conf: Optional[float],
         masks: bool) -> tuple:
    """Run the selected models; return (response, image, finding dets)."""
    registry = STATE["models"]
    if not registry:
        raise HTTPException(503, f"gingivitis models not loaded: {STATE['error']}")

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image larger than 20 MB")
    img = _decode_image(data)
    h, w = img.shape[:2]

    if models:
        keys = [k.strip() for k in models.split(",") if k.strip()]
        unknown = [k for k in keys if k not in registry]
        if unknown:
            raise HTTPException(400, f"unknown models {unknown}; "
                                     f"available: {list(registry)}")
    else:
        keys = [k for k, m in registry.items() if m.enabled]

    t0 = time.perf_counter()
    all_dets, timings = [], {}
    for key in keys:
        m = registry[key]
        t = time.perf_counter()
        result = m.predict(img, conf if conf is not None else m.conf)
        all_dets.extend(_detections(m, result, w, h, masks))
        timings[key] = round((time.perf_counter() - t) * 1000, 1)

    anatomy_names = STATE["anatomy"]
    grouped = defaultdict(list)
    anatomy = defaultdict(int)
    for d in all_dets:
        if d["class"] in anatomy_names:
            anatomy[d["class"]] += 1
        else:
            grouped[d["class"]].append(d)

    # Every condition any loaded model can detect appears in the response,
    # so the client sees explicit "detected: false" rather than a missing key.
    possible = {registry[k].names[i] for k in keys
                for i in (registry[k].class_ids or registry[k].names)}
    conditions = {}
    for name in sorted(possible - anatomy_names):
        dets = sorted(grouped.get(name, []), key=lambda d: -d["confidence"])
        conditions[name] = {
            "detected": bool(dets),
            "count": len(dets),
            "max_confidence": dets[0]["confidence"] if dets else 0.0,
            "detections": dets,
        }

    response = {
        "image": {"filename": file.filename, "width": w, "height": h},
        "summary": {
            "conditions_found": [n for n, c in conditions.items() if c["detected"]],
            "total_findings": sum(c["count"] for c in conditions.values()),
        },
        "conditions": conditions,
        "anatomy": dict(anatomy),
        "models_run": keys,
        "timing_ms": {"per_model": timings,
                      "total": round((time.perf_counter() - t0) * 1000, 1)},
    }
    findings = [d for d in all_dets if d["class"] not in anatomy_names]
    return response, img, findings


_MODELS_Q = Query(None, description="comma-separated model keys")
_CONF_Q = Query(None, ge=0.0, le=1.0)
_LABELS_Q = Query("short", pattern="^(short|full|none)$",
                  description="tag style on the image: short | full | none")


@router.post("/analyze-gingivitis")
def analyze_gingivitis(
    file: UploadFile = File(...),
    models: Optional[str] = _MODELS_Q,
    conf: Optional[float] = _CONF_Q,
    annotate: bool = False,
    masks: bool = False,
    labels: str = _LABELS_Q,
):
    response, img, findings = _run(file, models, conf, masks)
    if annotate:
        response["annotated_image_jpeg_base64"] = base64.b64encode(
            _annotate(img, findings, labels)).decode("ascii")
    return response


@router.post("/analyze-gingivitis-image", response_class=Response,
             responses={200: {"content": {"image/jpeg": {}}}})
def analyze_gingivitis_image(
    file: UploadFile = File(...),
    models: Optional[str] = _MODELS_Q,
    conf: Optional[float] = _CONF_Q,
    masks: bool = True,
    labels: str = _LABELS_Q,
):
    """Same as /analyze-gingivitis, but returns the annotated JPEG directly.
    The findings summary is also sent in X-Conditions-Found / X-Total-Findings
    headers so clients can show it without a second request."""
    response, img, findings = _run(file, models, conf, masks)
    summary = response["summary"]
    return Response(
        content=_annotate(img, findings, labels),
        media_type="image/jpeg",
        headers={
            "X-Conditions-Found": ",".join(summary["conditions_found"]),
            "X-Total-Findings": str(summary["total_findings"]),
            "X-Inference-Ms": str(response["timing_ms"]["total"]),
        },
    )
