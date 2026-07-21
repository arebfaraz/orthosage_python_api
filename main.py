import base64
import io
import os
import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image


logger = logging.getLogger(__name__)

from inference import OrthodonticAIService
from landmarks import (
    ImageType,
    LandmarkDetectionError,
    LandmarkDetectionService,
    LandmarkServiceConfig,
    ModelNotConfiguredError,
    NoFaceDetectedError,
    UnsupportedImageTypeError,
    build_default_factory,
)

app = FastAPI(title="Webceph AI Digitization Service")

ai_service = OrthodonticAIService(weights_path=os.getenv("MODEL_WEIGHTS_PATH"))

# Only LATERAL_CEPH has a trained checkpoint today; every other weights path
# is read from an env var so a newly trained model can be deployed by setting
# one variable, with no code change (see landmarks/factory.py).
landmark_service = LandmarkDetectionService(
    factory=build_default_factory(
        LandmarkServiceConfig(
            weights_paths={
                ImageType.LATERAL_CEPH: os.getenv("MODEL_WEIGHTS_PATH"),
                ImageType.PA_CEPH: os.getenv("PA_CEPH_WEIGHTS_PATH"),
                ImageType.ORTHOPAN: os.getenv("ORTHOPAN_WEIGHTS_PATH"),
                ImageType.RIGHT_INTRAORAL: os.getenv("RIGHT_INTRAORAL_WEIGHTS_PATH"),
                ImageType.FRONT_INTRAORAL: os.getenv("FRONT_INTRAORAL_WEIGHTS_PATH"),
                ImageType.LEFT_INTRAORAL: os.getenv("LEFT_INTRAORAL_WEIGHTS_PATH"),
                ImageType.UPPER_OCCLUSAL: os.getenv("UPPER_OCCLUSAL_WEIGHTS_PATH"),
                ImageType.LOWER_OCCLUSAL: os.getenv("LOWER_OCCLUSAL_WEIGHTS_PATH"),
            }
        )
    )
)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/image-digitization")
async def image_digitization(image: UploadFile = File(...)):
    # Validate uploaded file
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=422,
            detail="Uploaded file must be an image.",
        )

    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=422,
            detail="Uploaded image is empty.",
        )

    # Validate image
    try:
        with Image.open(io.BytesIO(image_bytes)) as probe:
            width, height = probe.size
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Could not read the uploaded image.",
        )

    try:
        landmarks = ai_service.run_inference(image_bytes, width, height)
        marked_image_bytes = ai_service.annotate_image(image_bytes, landmarks)

        return {
            "landmarks": landmarks,
            "marked_image": (
                "data:image/png;base64,"
                + base64.b64encode(marked_image_bytes).decode("ascii")
            ),
        }

    except Exception:
        logger.exception("Image digitization failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to process image.",
        )

@app.post("/landmarks")
async def detect_landmarks(image_type: str = Form(...), image: UploadFile = File(...)):
    """Type-aware landmark endpoint covering all 12 record image types
    (lateral/PA ceph, panoramic, facial photos, intraoral, occlusal) --
    see landmarks/types.py:ImageType for the accepted `image_type` values.
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="Uploaded file must be an image.")

    image_bytes = await image.read()

    try:
        result = landmark_service.detect(image_bytes, image_type)
    except UnsupportedImageTypeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except NoFaceDetectedError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LandmarkDetectionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"image_type": image_type, "landmarks": result.as_dicts()}
