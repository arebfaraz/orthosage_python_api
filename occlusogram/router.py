"""
occlusogram/router.py
─────────────────────
Intraoral photo → clinical view + FDI-numbered tooth instances (+ per-tooth
gingivitis grade on frontal views), ported from the orthosage_occlusogram
project (IOP-Compass API, its POST /analyze). pipeline.py, contours.py,
gingivitis.py, postprocessing.py and postprocessing.yaml are copied
unchanged except for package-relative imports. Request fields and response
body are unchanged; only the path is new (/analyze is the X-ray route here).

Models are loaded once at startup (see load_models, called from api.py). If
the weights are missing, the rest of the API still starts and this endpoint
returns 503. gingivitis.pt is optional: without it grading is reported as
skipped, as in the original.

Environment:
    OCCLUSOGRAM_WEIGHTS_DIR        folder holding view_classifier.pt, maskrcnn.pt,
                                   gingivitis.pt (default: occlusogram/weights)
    OCCLUSOGRAM_DEVICE             auto | cpu | cuda (default: auto)
    OCCLUSOGRAM_INFERENCE_LONG_SIDE  long side the segmenter sees (default: 2048)

Endpoint:
    POST /analyze/occlusogram   multipart "file" (+ optional form fields view,
                                mirror_acquisition, gingivitis) → JSON
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

HERE = Path(__file__).resolve().parent
WEIGHTS_DIR = Path(os.environ.get("OCCLUSOGRAM_WEIGHTS_DIR", HERE / "weights"))
GINGIVITIS_WEIGHTS = WEIGHTS_DIR / "gingivitis.pt"
#: The gingivitis detector was trained on frontal photographs only.
GINGIVITIS_VIEWS = ("frontal",)
MAX_UPLOAD_BYTES = 40 * 1024 * 1024
ACCEPTED_TYPES = ("image/png", "image/jpeg", "image/jpg", "image/webp")

log = logging.getLogger("occlusogram")
router = APIRouter(tags=["occlusogram"])

STATE: dict = {"pipeline": None, "gingivitis": None, "error": None}


def load_models() -> None:
    """Load the view classifier, Mask R-CNN and (optional) gingivitis YOLO.
    Never raises: on failure the reason is kept in STATE["error"] and the
    endpoint answers 503."""
    try:
        from .pipeline import Pipeline, Settings

        settings = Settings(
            classifier_weights=WEIGHTS_DIR / "view_classifier.pt",
            maskrcnn_weights=WEIGHTS_DIR / "maskrcnn.pt",
            postprocessing_config=HERE / "postprocessing.yaml",
            device=os.environ.get("OCCLUSOGRAM_DEVICE", "auto"),
            inference_long_side=int(os.environ.get("OCCLUSOGRAM_INFERENCE_LONG_SIDE", 2048)),
        )
        missing = settings.missing()
        if missing:
            raise FileNotFoundError(f"missing required file(s): {', '.join(missing)}")
        pipeline = Pipeline(settings)
        gingivitis_model = None
        if GINGIVITIS_WEIGHTS.is_file():
            from .gingivitis import GingivitisModel

            gingivitis_model = GingivitisModel(GINGIVITIS_WEIGHTS, device=pipeline.device)
        else:
            print(f"[occlusogram] no gingivitis weights at {GINGIVITIS_WEIGHTS}; grading disabled")
        STATE.update(pipeline=pipeline, gingivitis=gingivitis_model, error=None)
        print(f"[occlusogram] models loaded on {pipeline.device}")
    except Exception as e:  # noqa: BLE001
        STATE["error"] = f"{type(e).__name__}: {e}"
        print(f"[occlusogram] WARNING: models not ready: {STATE['error']}")


def status() -> dict:
    """Summary for the main /health endpoint."""
    p = STATE["pipeline"]
    return {
        "device": p.device if p else None,
        "model_loaded": p is not None,
        "gingivitis_model_loaded": STATE["gingivitis"] is not None,
        "error": STATE["error"],
    }


def _form_flag(raw: Optional[str], default: bool) -> bool:
    """Read a boolean form field, tolerating the spellings HTML forms send."""
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# Plain `def`: FastAPI runs it in a worker thread, so the 2-5 s of CPU
# inference doesn't block other routes. Pipeline.lock serialises model calls.
@router.post("/analyze/occlusogram")
def analyze_occlusogram(
    file: UploadFile = File(..., description="One intraoral photograph (PNG, JPEG or WebP)"),
    view: Optional[str] = Form(
        None, description="Skip classification and use this view: frontal, left_buccal, "
                          "right_buccal, upper_occlusal, lower_occlusal"),
    mirror_acquisition: Optional[str] = Form(
        None, description="Photograph was taken through an intraoral mirror "
                          "(flipped before the models see it). Default true."),
    gingivitis: Optional[str] = Form(
        None, description="Grade gingivitis per tooth on frontal views. Default true."),
):
    from .gingivitis import is_issue, grade_rank
    from .pipeline import VIEW_CLASSES, VIEW_LABELS

    pipeline = STATE["pipeline"]
    if pipeline is None:
        raise HTTPException(503, f"occlusogram models not loaded: {STATE['error']}")

    # Parsed by hand: FastAPI reads a missing bool form field as false (checkbox
    # semantics), which would silently skip the flip the models were trained with.
    mirror_acquisition = _form_flag(mirror_acquisition, default=True)
    gingivitis = _form_flag(gingivitis, default=True)
    if file.content_type not in ACCEPTED_TYPES:
        raise HTTPException(415, f"unsupported type {file.content_type!r}; "
                                 f"accepted: {', '.join(ACCEPTED_TYPES)}")
    raw = file.file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"image is {len(raw) / 1e6:.1f} MB, over the "
                                 f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB limit")
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, f"could not decode {file.filename!r} as an image")
    if view is not None and view not in VIEW_CLASSES:
        raise HTTPException(400, f"unknown view {view!r}; expected one of {list(VIEW_CLASSES)}")

    height, width = image.shape[:2]
    original = image
    if mirror_acquisition:
        # The models were trained in the mirrored convention; contours are reflected
        # back inside segment(), so the response is in the uploaded image's frame.
        image = cv2.flip(image, 1)

    start = time.perf_counter()
    if view is None:
        classification = pipeline.classify(image)
    else:
        classification = {"view": view, "view_label": VIEW_LABELS[view],
                          "confidence": None, "probabilities": None}
    classify_s = time.perf_counter() - start

    segmentation = pipeline.segment(image, classification["view"], flipped=mirror_acquisition)
    segmentation["timings"] = {"classify": round(classify_s, 3), **segmentation["timings"]}

    gingivitis_result = _grade_gingivitis(original, classification["view"],
                                          segmentation["instances"], gingivitis)
    if gingivitis_result["ran"]:
        segmentation["timings"]["gingivitis"] = gingivitis_result.pop("seconds")
    issues = [
        {"fdi": d["fdi"], "type": "gingivitis", "grade": d["grade"], "label": d["label"],
         "confidence": d["confidence"], "bbox": d["bbox"]}
        for d in gingivitis_result.get("detections", [])
        if is_issue(d["grade"])
    ]
    issues.sort(key=lambda i: (-grade_rank(i["grade"]), i["fdi"] is None, i["fdi"] or 0))

    log.info("%s: %s (%s) -> %d tooth instance(s), %d issue(s)", file.filename,
             classification["view"], classification["confidence"],
             len(segmentation["instances"]), len(issues))
    return {
        "filename": file.filename,
        "width": width,
        "height": height,
        "mirror_acquisition": mirror_acquisition,
        "classification": classification,
        "segmenter": "mask-rcnn",
        "tooth_count": len(segmentation["instances"]),
        "issues": issues,
        "gingivitis": gingivitis_result,
        **segmentation,
    }


def _grade_gingivitis(image_bgr: np.ndarray, view: str, instances: list, requested: bool) -> dict:
    """Run the gingivitis detector and attach its grades to ``instances`` in place.

    ``image_bgr`` is the photograph as uploaded (never mirrored), so the boxes share
    the instances' coordinate frame.
    """
    from .gingivitis import GRADE_LABELS, grade_rank, match_to_instances

    model = STATE["gingivitis"]
    skipped = None
    if not requested:
        skipped = "not requested"
    elif model is None:
        skipped = "gingivitis model not installed"
    elif view not in GINGIVITIS_VIEWS:
        skipped = f"only graded on {', '.join(GINGIVITIS_VIEWS)} views; this is {view}"
    if skipped:
        for inst in instances:
            inst["gingivitis"] = None
        return {"ran": False, "skipped_reason": skipped}

    start = time.perf_counter()
    with STATE["pipeline"].lock:
        detections, jaws = model.predict(image_bgr)
    match_to_instances(detections, instances)
    worst = max((d["grade"] for d in detections), key=grade_rank, default=None)
    return {
        "ran": True,
        "skipped_reason": None,
        "seconds": round(time.perf_counter() - start, 3),
        "model": model.weights.name,
        "teeth_graded": len(detections),
        "worst_grade": worst,
        "worst_label": GRADE_LABELS[worst] if worst else None,
        "detections": sorted(detections, key=lambda d: (d["fdi"] is None, d["fdi"] or 0)),
        "jaw_regions": jaws,
    }
