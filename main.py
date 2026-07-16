import base64
import io
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from inference import OrthodonticAIService

app = FastAPI(title="Webceph AI Digitization Service")

ai_service = OrthodonticAIService(weights_path=os.getenv("MODEL_WEIGHTS_PATH"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/image-digitization")
async def image_digitization(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="Uploaded file must be an image.")

    image_bytes = await image.read()

    try:
        with Image.open(io.BytesIO(image_bytes)) as probe:
            width, height = probe.size
    except Exception:
        raise HTTPException(status_code=422, detail="Could not read the uploaded image.")

    landmarks = ai_service.run_inference(image_bytes, width, height)
    marked_image_bytes = ai_service.annotate_image(image_bytes, landmarks)
    marked_image = "data:image/png;base64," + base64.b64encode(marked_image_bytes).decode("ascii")

    return {"landmarks": landmarks, "marked_image": marked_image}
