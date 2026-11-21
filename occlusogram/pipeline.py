"""One intraoral photograph in, its clinical view and FDI tooth instances out.

Extracted from the IOP-Compass viewer backend (``app/backend/pipeline.py``) and kept
behaviourally identical for a single image:

1. view classification -- ResNet18, plain argmax (the five-image constrained
   assignment does not apply to one photograph);
2. Mask R-CNN R50-FPN on the full image, downscaled to ``inference_long_side``;
3. the frozen post-processing rules (``postprocessing.yaml``);
4. masks -> polygons, scaled back to the uploaded image's coordinates.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from .contours import mask_to_polygons, mirror_polygons_x, polygons_area, polygons_bbox, polygons_centroid
from .postprocessing import PostProcessParams, postprocess_instances

log = logging.getLogger(__name__)

VIEW_CLASSES = (
    "frontal",
    "left_buccal",
    "right_buccal",
    "upper_occlusal",
    "lower_occlusal",
)
VIEW_LABELS = {
    "frontal": "Frontal",
    "left_buccal": "Left buccal",
    "right_buccal": "Right buccal",
    "upper_occlusal": "Upper occlusal",
    "lower_occlusal": "Lower occlusal",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Settings:
    classifier_weights: Path
    maskrcnn_weights: Path
    postprocessing_config: Path
    device: str = "cpu"
    #: Long side the segmenter sees; the benchmark's ``inference_long_side``.
    inference_long_side: int = 2048
    classifier_image_size: int = 256

    def missing(self) -> list[str]:
        return [
            str(p)
            for p in (self.classifier_weights, self.maskrcnn_weights, self.postprocessing_config)
            if not p.is_file()
        ]


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return requested


def target_size(width: int, height: int, long_side: int | None) -> tuple[int, int]:
    """``(width, height)`` scaled so the long side is at most ``long_side``; never upscales."""
    if not long_side or max(width, height) <= long_side:
        return width, height
    scale = long_side / max(width, height)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


# ----------------------------------------------------------------------- models
def load_classifier(path: Path, device: str) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(VIEW_CLASSES))
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    return model.eval().to(device)


class MaskRcnnPredictor:
    """Mask R-CNN R50-FPN with heads for the checkpoint's FDI label set."""

    def __init__(self, checkpoint: Path, device: str):
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        config = state.get("config") or {}
        self.fdi_labels: tuple[int, ...] = tuple(state["fdi_labels"])
        self.index_to_fdi = {i + 1: fdi for i, fdi in enumerate(self.fdi_labels)}
        self.score_threshold = float(config.get("score_threshold", 0.05))
        self.mask_threshold = float(config.get("mask_threshold", 0.5))

        model = maskrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            trainable_backbone_layers=int(config.get("trainable_backbone_layers", 3)),
            box_detections_per_img=int(config.get("max_detections", 64)),
        )
        num_classes = len(self.fdi_labels) + 1  # background + one class per FDI code
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, 256, num_classes)
        model.load_state_dict(state["model_state"])
        self.model = model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray) -> tuple[list[np.ndarray], list[int | None], list[float]]:
        tensor = (
            torch.from_numpy(image_bgr[:, :, ::-1].copy())
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
            .to(self.device)
        )
        out = self.model([tensor])[0]
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        masks = out["masks"].cpu().numpy()[:, 0]
        keep = scores >= self.score_threshold
        out_masks, out_fdis, out_scores = [], [], []
        for mask, label, score in zip(masks[keep], labels[keep], scores[keep]):
            binary = mask >= self.mask_threshold
            if not binary.any():
                continue
            out_masks.append(binary)
            out_fdis.append(self.index_to_fdi.get(int(label)))
            out_scores.append(float(score))
        return out_masks, out_fdis, out_scores


# --------------------------------------------------------------------- pipeline
class Pipeline:
    """Holds the loaded models. Model calls are serialised by :attr:`lock`."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = resolve_device(settings.device)
        self.lock = threading.Lock()
        self.post_params = PostProcessParams.from_yaml(settings.postprocessing_config)
        log.info("loading view classifier from %s", settings.classifier_weights)
        self.classifier = load_classifier(settings.classifier_weights, self.device)
        log.info("loading Mask R-CNN from %s", settings.maskrcnn_weights)
        self.segmenter = MaskRcnnPredictor(settings.maskrcnn_weights, self.device)

    def classify(self, image_bgr: np.ndarray) -> dict:
        size = self.settings.classifier_image_size
        resized = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        tensor = (tensor - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)

        with self.lock, torch.no_grad():
            logits = self.classifier(tensor.unsqueeze(0).to(self.device))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        index = int(probs.argmax())
        return {
            "view": VIEW_CLASSES[index],
            "view_label": VIEW_LABELS[VIEW_CLASSES[index]],
            "confidence": round(float(probs[index]), 4),
            "probabilities": {v: round(float(p), 4) for v, p in zip(VIEW_CLASSES, probs)},
        }

    def segment(self, image_bgr: np.ndarray, view: str, flipped: bool = False) -> dict:
        """Segment ``image_bgr`` and return instances in its original coordinates.

        ``flipped`` says the image was mirrored on ingest (an intraoral-mirror
        acquisition); contours are reflected back so the caller always gets the
        uploaded photograph's frame.
        """
        height, width = image_bgr.shape[:2]
        work_w, work_h = target_size(width, height, self.settings.inference_long_side)
        work = (
            image_bgr
            if (work_w, work_h) == (width, height)
            else cv2.resize(image_bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)
        )

        timings: dict[str, float] = {}
        with self.lock:
            start = time.perf_counter()
            masks, fdis, scores = self.segmenter.predict(work)
            timings["segment"] = time.perf_counter() - start

            # The full image is the region: nothing is excluded.
            start = time.perf_counter()
            masks, fdis, scores, post_stats = postprocess_instances(
                masks,
                fdis,
                scores,
                roi_mask=np.ones((work_h, work_w), dtype=np.uint8),
                exclusion_mask=np.zeros((work_h, work_w), dtype=np.uint8),
                params=self.post_params,
                view_label=view,
            )
            timings["postprocess"] = time.perf_counter() - start

        scale_x = width / float(work_w)
        scale_y = height / float(work_h)
        instances = []
        for index, (mask, fdi, score) in enumerate(zip(masks, fdis, scores)):
            contours = mask_to_polygons(mask, scale_x, scale_y)
            if not contours:
                continue
            if flipped:
                contours = mirror_polygons_x(contours, width)
            x0, y0, x1, y1 = polygons_bbox(contours)
            cx, cy = polygons_centroid(contours)
            instances.append(
                {
                    "instance_id": f"i{index}",
                    "fdi": int(fdi) if fdi is not None else None,
                    "score": round(float(score), 4),
                    "contours": contours,
                    "area": round(polygons_area(contours), 1),
                    "bbox": [round(v, 1) for v in (x0, y0, x1, y1)],
                    "centroid": [round(cx, 1), round(cy, 1)],
                }
            )
        instances.sort(key=lambda i: (i["fdi"] is None, i["fdi"] or 0))

        return {
            "instances": instances,
            "timings": {k: round(v, 3) for k, v in timings.items()},
            "stats": post_stats.to_dict(),
        }
