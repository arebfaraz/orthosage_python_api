import io

import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from model import OrthosageLandmarkRegressor

LANDMARK_LABELS = [
    "Sella (S)",
    "Nasion (N)",
    "A-Point (A)",
    "B-Point (B)",
    "Pogonion (Pog)",
]


class OrthodonticAIService:
    def __init__(self, weights_path: str | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = OrthosageLandmarkRegressor(num_landmarks=len(LANDMARK_LABELS))

        if weights_path:
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def run_inference(self, image_bytes: bytes, original_width: int, original_height: int) -> list[dict]:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = torch.sigmoid(self.model(input_tensor))

        # predictions are normalized [0, 1] coordinates; flat -> (num_landmarks, 2)
        coords = predictions.view(-1, 2).cpu().numpy()

        landmarks = []
        for label, (norm_x, norm_y) in zip(LANDMARK_LABELS, coords):
            landmarks.append({
                "label": label,
                "x": round(float(norm_x) * original_width, 1),
                "y": round(float(norm_y) * original_height, 1),
            })
        return landmarks

    def annotate_image(self, image_bytes: bytes, landmarks: list[dict]) -> bytes:
        """Draw the detected landmarks directly onto the uploaded image and return PNG bytes."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(image)
        radius = max(4, round(min(image.width, image.height) * 0.006))

        for point in landmarks:
            x, y = point["x"], point["y"]
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(20, 184, 166), outline=(255, 255, 255), width=2)

            label = point["label"]
            text_bbox = draw.textbbox((0, 0), label)
            text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            text_x, text_y = x + radius + 4, y - radius - text_h - 4
            draw.rectangle((text_x - 3, text_y - 2, text_x + text_w + 3, text_y + text_h + 2), fill=(15, 23, 42))
            draw.text((text_x, text_y), label, fill=(255, 255, 255))

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
