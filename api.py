# -*- coding: utf-8 -*-
"""
REST API for the U-Net panoramic-x-ray teeth segmentation model.

Run:
    .venv/Scripts/python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000
Then open  http://localhost:8000/docs

Endpoints
    GET  /health          -> {"status": "ok", ...}
    POST /segment         -> image/png  (binary teeth mask, original resolution)
    POST /overlay         -> image/png  (mask drawn over the input)
    POST /analyze         -> JSON: teeth_count + base64 mask / overlay / cca images
    POST /analyze-gingivitis        -> JSON: intraoral-photo findings (gingivitis,
                                       recession, caries, plaque), see gingivitis/
    POST /analyze-gingivitis-image  -> image/jpeg: the same findings drawn on the photo
    POST /analyze/lateral-ceph      -> JSON (or PNG with ?format=image): 29 lateral-ceph
                                       landmarks (HRNet), see lateral_ceph/
    POST /analyze-face              -> JSON (or PNG with ?format=image): frontal-photo
                                       landmarks + reference lines, see face_analysis/
    POST /analyze-face/recompute    -> JSON: lines re-derived from edited landmarks
"""
import io
import os
import base64
import importlib.util

import numpy as np
from PIL import Image
import cv2

# The model was saved with Keras 2 (TF 2.10). TF >= 2.16 ships Keras 3, which
# cannot deserialise that legacy .h5 (e.g. Conv2DTranspose's `groups` arg). When
# tf-keras is installed this routes tf.keras back to Keras 2 so the file loads.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# By default the API runs inference on CPU: a single 512x512 forward pass is
# ~1 s, and it keeps the 8 GB GPU free for training. Set API_USE_GPU=1 to use it.
if os.environ.get("API_USE_GPU", "0") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    _spec = importlib.util.find_spec("nvidia")
    if _spec and _spec.submodule_search_locations:
        _root = list(_spec.submodule_search_locations)[0]
        for _sub in os.listdir(_root):
            _bin = os.path.join(_root, _sub, "bin")
            if os.path.isdir(_bin):
                os.add_dll_directory(_bin)
                os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from fastapi import FastAPI, File, UploadFile, Query, HTTPException  # noqa: E402
from fastapi.responses import Response, JSONResponse  # noqa: E402

from cca import cca_analysis  # noqa: E402
import gingivitis  # noqa: E402
import lateral_ceph  # noqa: E402
import face_analysis  # noqa: E402

HERE = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(HERE, "dental_xray_seg.h5"))
INPUT_SIZE = (512, 512)

for _g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:  # noqa: BLE001
        pass

app = FastAPI(title="Teeth Segmentation API",
              description="U-Net semantic segmentation of teeth in panoramic dental X-rays.",
              version="1.0.0")
app.include_router(gingivitis.router)
app.include_router(lateral_ceph.router)
app.include_router(face_analysis.router)

_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"model file not found: {MODEL_PATH} (train it with train.py)")
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model


# --------------------------------------------------------------------------- #
def _read_image(raw: bytes):
    """bytes -> (gray_uint8 HxW, bgr_uint8 HxWx3) at original resolution."""
    try:
        pil = Image.open(io.BytesIO(raw))
        pil.load()
    except Exception:
        raise HTTPException(status_code=400, detail="could not decode image")
    gray = np.array(pil.convert("L"))
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray, bgr


def _predict_mask(gray: np.ndarray, threshold: float) -> np.ndarray:
    """Return a binary uint8 mask (0/255) at the ORIGINAL resolution of `gray`."""
    h, w = gray.shape
    small = cv2.resize(gray, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    x = small.astype("float32")[None, ..., None] / 255.0
    prob = get_model().predict(x, verbose=0)[0, :, :, 0]
    prob_full = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return ((prob_full > threshold) * 255).astype("uint8")


def _png(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise HTTPException(status_code=500, detail="png encode failed")
    return buf.tobytes()


def _overlay(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color = np.zeros_like(bgr)
    color[mask > 0] = (0, 0, 255)
    return cv2.addWeighted(bgr, 1.0, color, 0.4, 0)


# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": os.path.basename(MODEL_PATH),
        "model_loaded": _model is not None,
        "gpu": [d.name for d in tf.config.list_physical_devices("GPU")],
        "input_size": INPUT_SIZE,
        "gingivitis": gingivitis.router_status(),
        "lateral_ceph": lateral_ceph.router_status(),
        "face": face_analysis.router_status(),
    }


@app.post("/segment")
async def segment(file: UploadFile = File(...),
                  threshold: float = Query(0.5, ge=0.0, le=1.0)):
    gray, _ = _read_image(await file.read())
    mask = _predict_mask(gray, threshold)
    return Response(content=_png(mask), media_type="image/png")


@app.post("/overlay")
async def overlay(file: UploadFile = File(...),
                  threshold: float = Query(0.5, ge=0.0, le=1.0)):
    gray, bgr = _read_image(await file.read())
    mask = _predict_mask(gray, threshold)
    return Response(content=_png(_overlay(bgr, mask)), media_type="image/png")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...),
                  threshold: float = Query(0.5, ge=0.0, le=1.0),
                  erode_iteration: int = Query(3, ge=0, le=10),
                  open_iteration: int = Query(2, ge=0, le=10),
                  area_threshold: int = Query(2000, ge=0),
                  include_images: bool = Query(True)):
    gray, bgr = _read_image(await file.read())
    mask = _predict_mask(gray, threshold)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    try:
        cca_img, teeth_count = cca_analysis(
            bgr, mask_bgr, erode_iteration, open_iteration, area_threshold)
    except Exception as e:  # noqa: BLE001
        cca_img, teeth_count = bgr, None
        cca_error = str(e)
    else:
        cca_error = None

    coverage = float((mask > 0).mean())
    out = {
        "teeth_count": teeth_count,
        "mask_coverage_fraction": round(coverage, 4),
        "threshold": threshold,
        "image_size": {"width": gray.shape[1], "height": gray.shape[0]},
        "cca_error": cca_error,
    }
    if include_images:
        b64 = lambda a: base64.b64encode(_png(a)).decode()  # noqa: E731
        out["mask_png_b64"] = b64(mask)
        out["overlay_png_b64"] = b64(_overlay(bgr, mask))
        out["cca_png_b64"] = b64(cca_img)
    return JSONResponse(out)


@app.on_event("startup")
def _warm():
    try:
        get_model()
        get_model().predict(np.zeros((1, *INPUT_SIZE, 1), "float32"), verbose=0)
        print("[api] model warm")
    except Exception as e:  # noqa: BLE001
        print(f"[api] WARNING: model not ready: {e}")
    gingivitis.load_models()
    lateral_ceph.load_models()
    face_analysis.load_models()
