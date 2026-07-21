import io

import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from model import HRNetW32, LANDMARK_LABELS

INPUT_SIZE = 768

# Groups of landmarks that form one continuous anatomical structure, each
# traced as its own smooth curve rather than one straight line through every
# detected point in arbitrary order.
TRACING_CHAINS = [
    ["Sella (S)", "Nasion (N)"],
    ["Nasion (N)", "A-Point (A)", "B-Point (B)", "Pogonion (Pog)", "Gnathion (Gn)", "Menton (Me)"],
    ["Articulare (Ar)", "Gonion (Go)", "Menton (Me)"],
    ["Posterior Nasal Spine (PNS)", "Anterior Nasal Spine (ANS)"],
    ["Orbitale (Or)", "Porion (Po)"],
    ["Subnasale (Sn)", "Upper Lip", "Lower Lip", "Soft Tissue Pogonion"],
    ["Upper Incisor Tip (U1)", "Lower Incisor Tip (L1)"],
]


def _catmull_rom_point(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    x = 0.5 * (
        2 * p1[0] + (p2[0] - p0[0]) * t
        + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
        + (3 * p1[0] - p0[0] - 3 * p2[0] + p3[0]) * t3
    )
    y = 0.5 * (
        2 * p1[1] + (p2[1] - p0[1]) * t
        + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
        + (3 * p1[1] - p0[1] - 3 * p2[1] + p3[1]) * t3
    )
    return (x, y)


def _bow_points(p0, p1, bow_ratio=0.12, samples=20):
    """Sample a quadratic-bezier arc between two points so a 2-point chain
    (e.g. Sella-Nasion) still reads as a curve instead of a straight line."""
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    length = (dx ** 2 + dy ** 2) ** 0.5 or 1
    bow = length * bow_ratio
    cx = (p0[0] + p1[0]) / 2 + (-dy / length) * bow
    cy = (p0[1] + p1[1]) / 2 + (dx / length) * bow

    curve = []
    for step in range(samples + 1):
        t = step / samples
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * cx + t ** 2 * p1[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * cy + t ** 2 * p1[1]
        curve.append((x, y))
    return curve


def _smooth_curve(points, samples_per_segment=16):
    """Densely sample a Catmull-Rom spline through points so the drawn
    polyline reads as a smooth anatomical curve instead of straight segments."""
    if len(points) == 2:
        return _bow_points(points[0], points[1])
    if len(points) < 2:
        return points

    padded = [points[0], *points, points[-1]]
    curve = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        for step in range(samples_per_segment):
            curve.append(_catmull_rom_point(p0, p1, p2, p3, step / samples_per_segment))
    curve.append(points[-1])
    return curve


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

        if len(landmarks) > 1:
            line_width = max(1, round(radius * 0.3))
            point_map = {point["label"]: (point["x"], point["y"]) for point in landmarks}
            for chain in TRACING_CHAINS:
                chain_points = [point_map[label] for label in chain if label in point_map]
                if len(chain_points) < 2:
                    continue
                curve = _smooth_curve(chain_points)
                draw.line(curve, fill=(19, 20, 228), width=line_width, joint="curve")

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
