import io

import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from model import HRNetW32, LANDMARK_LABELS

INPUT_SIZE = 768


class OrthodonticAIService:
    def __init__(self, weights_path: str | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HRNetW32()

        if weights_path:
            # weights_only=True is safe by construction: it refuses to unpickle
            # anything but plain tensors, so it can't execute arbitrary code
            # even if the checkpoint file were tampered with.
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def run_inference(self, image_bytes: bytes, original_width: int, original_height: int) -> list[dict]:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            heatmaps = self.model(input_tensor)[0]  # (num_landmarks, heatmap_h, heatmap_w)

        heatmap_h, heatmap_w = heatmaps.shape[1], heatmaps.shape[2]

        landmarks = []
        for label, heatmap in zip(LANDMARK_LABELS, heatmaps):
            peak_index = torch.argmax(heatmap)
            heatmap_y = (peak_index // heatmap_w).item()
            heatmap_x = (peak_index % heatmap_w).item()

            # heatmap space -> model input space -> original image space
            input_x = (heatmap_x + 0.5) * (INPUT_SIZE / heatmap_w)
            input_y = (heatmap_y + 0.5) * (INPUT_SIZE / heatmap_h)
            x = input_x * (original_width / INPUT_SIZE)
            y = input_y * (original_height / INPUT_SIZE)

            landmarks.append({
                "label": label,
                "x": round(float(x), 1),
                "y": round(float(y), 1),
                "confidence": round(float(heatmap.max()), 3),
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
