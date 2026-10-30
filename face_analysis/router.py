"""
face_analysis/router.py
───────────────────────
Frontal facial photo analysis (MediaPipe FaceLandmarker → named soft-tissue
landmarks + midsagittal / horizontal / lip-canting / jaw-taper lines), ported
from orthosage_python2_api's /analyze-face routes (previously py1.ekict.com).
Paths, params and response bodies are unchanged; facial_analysis.py is
copied as-is.

The landmarker is created once at startup (see load_models, called from
api.py) so a missing system library shows up in /health instead of on the
first request. On headless Linux MediaPipe needs libGLESv2 / libEGL:
    apt-get install -y libgles2 libegl1

Endpoints:
    POST /analyze-face             multipart "file"
        ?format=json   (default) {"width", "height", "landmarks": [...], "lines": [...]}
        ?format=image  PNG with the landmarks and lines drawn on it
    POST /analyze-face/recompute   JSON {"landmarks": [{key, x, y}], "height"}
                                   → {"lines": [...]} (after the clinician drags points)
"""

import io
import threading
from typing import List

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image
from pydantic import BaseModel

from . import facial_analysis

router = APIRouter(tags=["face"])

STATE: dict = {"ready": False, "error": None, "lock": threading.Lock()}


def load_models() -> None:
    """Create the MediaPipe landmarker. Never raises: on failure the reason
    is kept in STATE["error"] and /analyze-face answers 503."""
    try:
        facial_analysis._get_landmarker()
        STATE.update(ready=True, error=None)
        print("[face] MediaPipe FaceLandmarker ready")
    except Exception as e:  # noqa: BLE001
        STATE["error"] = f"{type(e).__name__}: {e}"
        print(f"[face] WARNING: landmarker not ready: {STATE['error']}")


def status() -> dict:
    """Summary for the main /health endpoint."""
    return {"ready": STATE["ready"], "error": STATE["error"]}


class LandmarkPoint(BaseModel):
    key: str
    x: float
    y: float


class RecomputeRequest(BaseModel):
    landmarks: List[LandmarkPoint]
    height: float


@router.post("/analyze-face")
def analyze_face(file: UploadFile = File(...),
                 format: str = Query("json", pattern="^(json|image)$")):
    if not STATE["ready"]:
        raise HTTPException(503, f"face landmarker not loaded: {STATE['error']}")
    raw = file.file.read()
    try:
        # The landmarker is a shared singleton; MediaPipe graphs are not
        # safe to call from several threadpool workers at once.
        with STATE["lock"]:
            result = facial_analysis.analyze_face(raw)
    except ValueError as e:  # no face found
        raise HTTPException(422, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"invalid image: {e}")

    if format == "image":
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        points = {lm["key"]: {"x": lm["x"], "y": lm["y"]} for lm in result["landmarks"]}
        annotated = facial_analysis.draw_analysis(img, points, result["lines"])
        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        return Response(buf.getvalue(), media_type="image/png")

    return JSONResponse(result)


@router.post("/analyze-face/recompute")
def analyze_face_recompute(body: RecomputeRequest):
    """Recompute reference lines/angles from landmarks the clinician has
    dragged on the canvas, without re-running face detection."""
    landmarks = [lm.model_dump() for lm in body.landmarks]
    lines = facial_analysis.recompute_lines(landmarks, body.height)
    return JSONResponse({"lines": lines})
