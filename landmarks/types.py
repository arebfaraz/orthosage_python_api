"""Core value objects and the ImageType vocabulary shared by every detector.

ImageType values match the `key` column seeded in
Webceph/database/seeders/CephalometricImageSampleSeeder.php, so a record's
type string round-trips into this enum without translation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ImageType(str, Enum):
    LATERAL_CEPH = "lateral_ceph"
    PA_CEPH = "pa_ceph"
    ORTHOPAN = "orthopan"
    FRONTAL_PHOTO = "frontal_photo"
    SMILE_PHOTO = "smile_photo"
    PHOTO_45 = "45_photo"
    LATERAL_PHOTO = "lateral_photo"
    RIGHT_INTRAORAL = "right_intraoral"
    FRONT_INTRAORAL = "front_intraoral"
    LEFT_INTRAORAL = "left_intraoral"
    UPPER_OCCLUSAL = "upper_occlusal"
    LOWER_OCCLUSAL = "lower_occlusal"


@dataclass(frozen=True)
class Landmark:
    name: str
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class LandmarkResult:
    image_type: ImageType
    landmarks: list[Landmark]

    def as_dicts(self) -> list[dict]:
        return [
            {"label": lm.name, "x": round(lm.x, 1), "y": round(lm.y, 1), "confidence": round(lm.confidence, 3)}
            for lm in self.landmarks
        ]
