"""Public facade: the one entry point calling code (a FastAPI route, a CLI,
a test) should use. It knows nothing about torch, mediapipe, or heatmaps --
only how to load bytes into an image and hand off to the detector registered
for the requested image type.
"""

from __future__ import annotations

import io

from PIL import Image, UnidentifiedImageError

from .errors import InvalidImageError, UnsupportedImageTypeError
from .factory import LandmarkDetectorFactory, build_default_factory
from .types import ImageType, LandmarkResult


class LandmarkDetectionService:
    def __init__(self, factory: LandmarkDetectorFactory | None = None):
        self._factory = factory or build_default_factory()

    def detect(self, image_bytes: bytes, image_type: ImageType | str) -> LandmarkResult:
        resolved_type = self._resolve_image_type(image_type)
        image = self._load_image(image_bytes)
        detector = self._factory.create(resolved_type)
        return detector.detect(image)

    @staticmethod
    def _resolve_image_type(image_type: ImageType | str) -> ImageType:
        if isinstance(image_type, ImageType):
            return image_type
        try:
            return ImageType(image_type)
        except ValueError as exc:
            raise UnsupportedImageTypeError(image_type) from exc

    @staticmethod
    def _load_image(image_bytes: bytes) -> Image.Image:
        try:
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError as exc:
            raise InvalidImageError("Could not read the uploaded image.") from exc
