"""Per-tooth gingivitis grading, attached to the FDI tooth instances.

A YOLO detector (trained in orthosage_python2_api, ``train_gingivitis.py``) boxes the
gum line of each front tooth and grades it.  It knows nothing of FDI numbers, so each
box is matched to the IOP-Compass tooth instance its box overlaps most.

The detector was trained on **frontal** photographs only (retractors in, teeth
together), so it is run on frontal views only.

Class names are read from the weights rather than hard-coded, because the training
runs differ: exp2/exp4 have ``gingivitis_0_healthy`` .. ``gingivitis_4``; exp5 merges
healthy and level 1 into ``gingivitis_low``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from ultralytics import YOLO

#: Minimum box IoU between a gingivitis box and a tooth instance for them to be
#: matched.  Measured matches are 0.22-0.44 (the box covers gum as well as crown);
#: the next-best tooth is almost always below 0.08.
MATCH_MIN_IOU = 0.15

GRADE_LABELS = {
    "0": "healthy",
    "0-1": "healthy or mild",
    "1": "mild",
    "2": "moderate",
    "3": "severe",
    "4": "very severe",
}


def grade_from_class_name(name: str) -> str | None:
    """``gingivitis_3`` -> ``"3"``, ``gingivitis_low`` -> ``"0-1"``; ``None`` for non-tooth classes."""
    if not name.startswith("gingivitis"):
        return None
    if name.endswith("_low"):
        return "0-1"
    match = re.match(r"gingivitis_(\d)", name)
    return match.group(1) if match else None


def grade_rank(grade: str) -> int:
    """Highest level a grade can mean, for ordering (``"0-1"`` -> 1)."""
    return int(grade[-1])


def is_issue(grade: str) -> bool:
    """A grade that is gingivitis for certain: level 1 or worse, and not ``"0-1"``."""
    return grade not in ("0", "0-1")


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class GingivitisModel:
    def __init__(self, weights: Path, device: str = "cpu", conf: float = 0.25, iou: float = 0.5):
        self.weights = Path(weights)
        self.model = YOLO(str(self.weights))
        self.names: dict[int, str] = dict(self.model.names)
        # ultralytics names CUDA devices by index
        self.device = "0" if device.startswith("cuda") else device
        self.conf = conf
        self.iou = iou

    def predict(self, image_bgr: np.ndarray) -> tuple[list[dict], list[dict]]:
        """Gingivitis boxes and jaw-region boxes, in ``image_bgr``'s pixel coordinates."""
        result = self.model.predict(
            image_bgr, conf=self.conf, iou=self.iou, device=self.device, verbose=False
        )[0]
        teeth, jaws = [], []
        for box in result.boxes:
            name = self.names[int(box.cls.item())]
            bbox = [round(v, 1) for v in box.xyxy[0].tolist()]
            confidence = round(float(box.conf.item()), 4)
            if name.startswith("jaw_"):
                jaws.append({"region": name.removeprefix("jaw_"), "bbox": bbox, "confidence": confidence})
                continue
            grade = grade_from_class_name(name)
            if grade is None:
                continue
            teeth.append(
                {"fdi": None, "grade": grade, "label": GRADE_LABELS[grade], "confidence": confidence, "bbox": bbox}
            )
        return teeth, jaws


def match_to_instances(detections: list[dict], instances: list[dict]) -> None:
    """Set ``fdi`` on each detection and ``gingivitis`` on each instance, in place.

    Greedy one-to-one on box IoU, best pairs first.
    """
    pairs = sorted(
        (
            (box_iou(det["bbox"], inst["bbox"]), d, i)
            for d, det in enumerate(detections)
            for i, inst in enumerate(instances)
        ),
        reverse=True,
    )
    used_d, used_i = set(), set()
    for iou, d, i in pairs:
        if iou < MATCH_MIN_IOU:
            break
        if d in used_d or i in used_i:
            continue
        used_d.add(d)
        used_i.add(i)
        detections[d]["fdi"] = instances[i]["fdi"]
        instances[i]["gingivitis"] = {
            "grade": detections[d]["grade"],
            "label": detections[d]["label"],
            "confidence": detections[d]["confidence"],
        }
    for inst in instances:
        inst.setdefault("gingivitis", None)
