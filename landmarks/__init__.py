"""Type-aware landmark detection for every Webceph record image.

    from landmarks import ImageType, LandmarkDetectionService

    service = LandmarkDetectionService()
    result = service.detect(image_bytes, ImageType.LATERAL_CEPH)
    result.as_dicts()  # [{"label": "Sella (S)", "x": 512.3, "y": 210.7, "confidence": 0.94}, ...]

To register a checkpoint for an image type that doesn't have one yet (see
factory.HEATMAP_TYPE_INPUT_SIZES for which types those are):

    from landmarks import ImageType, LandmarkDetectionService, LandmarkServiceConfig, build_default_factory

    config = LandmarkServiceConfig(weights_paths={ImageType.PA_CEPH: "/path/to/pa_ceph.pth"})
    service = LandmarkDetectionService(factory=build_default_factory(config))
"""

from .errors import (
    InvalidImageError,
    LandmarkDetectionError,
    ModelNotConfiguredError,
    NoFaceDetectedError,
    UnsupportedImageTypeError,
)
from .factory import LandmarkDetectorFactory, LandmarkServiceConfig, build_default_factory
from .service import LandmarkDetectionService
from .types import ImageType, Landmark, LandmarkResult

__all__ = [
    "ImageType",
    "Landmark",
    "LandmarkResult",
    "LandmarkDetectionError",
    "UnsupportedImageTypeError",
    "ModelNotConfiguredError",
    "InvalidImageError",
    "NoFaceDetectedError",
    "LandmarkDetectorFactory",
    "LandmarkServiceConfig",
    "build_default_factory",
    "LandmarkDetectionService",
]
