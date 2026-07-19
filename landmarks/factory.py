"""Composition root: wires concrete detectors behind the LandmarkDetector
interface and registers one per ImageType. Adding a new image type, or
swapping a NotConfiguredModelBackend for a freshly trained checkpoint, means
changing this file only -- every other module in the package is closed for
modification (Open/Closed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .backends import HRNetModelBackend, LandmarkModelBackend, NotConfiguredModelBackend
from .detector import LandmarkDetector
from .detectors.facial_photo import FaceMeshFacialLandmarkDetector
from .detectors.radiograph import HeatmapLandmarkDetector
from .errors import UnsupportedImageTypeError
from .preprocessing import TorchvisionPreprocessor
from .schema import LANDMARK_SCHEMAS
from .types import ImageType

RADIOGRAPH_INPUT_SIZE = 768
INTRAORAL_INPUT_SIZE = 512

FACIAL_PHOTO_TYPES = {
    ImageType.FRONTAL_PHOTO,
    ImageType.SMILE_PHOTO,
    ImageType.PHOTO_45,
    ImageType.LATERAL_PHOTO,
}

# Every non-photo type is served by HeatmapLandmarkDetector; this maps each to
# the input resolution its (real or future) model expects.
HEATMAP_TYPE_INPUT_SIZES: dict[ImageType, int] = {
    ImageType.LATERAL_CEPH: RADIOGRAPH_INPUT_SIZE,
    ImageType.PA_CEPH: RADIOGRAPH_INPUT_SIZE,
    ImageType.ORTHOPAN: RADIOGRAPH_INPUT_SIZE,
    ImageType.RIGHT_INTRAORAL: INTRAORAL_INPUT_SIZE,
    ImageType.FRONT_INTRAORAL: INTRAORAL_INPUT_SIZE,
    ImageType.LEFT_INTRAORAL: INTRAORAL_INPUT_SIZE,
    ImageType.UPPER_OCCLUSAL: INTRAORAL_INPUT_SIZE,
    ImageType.LOWER_OCCLUSAL: INTRAORAL_INPUT_SIZE,
}


@dataclass
class LandmarkServiceConfig:
    """Where to find a trained checkpoint for each heatmap-based image type.

    Leave an entry as None (the default) until a model has actually been
    trained for it -- the factory then wires a NotConfiguredModelBackend that
    fails loudly instead of returning fabricated coordinates. Today only
    LATERAL_CEPH has a real checkpoint (ai-service/weights/hrnet_w32_ceph19_state_dict.pth).
    """

    weights_paths: dict[ImageType, str | None] = field(
        default_factory=lambda: {image_type: None for image_type in HEATMAP_TYPE_INPUT_SIZES}
    )


class LandmarkDetectorFactory:
    """Registry + lazy-singleton cache of LandmarkDetector instances.

    Caching matters here: constructing a detector loads a torch model (or
    spins up a MediaPipe graph), which is too expensive to redo per request.
    """

    def __init__(self):
        self._builders: dict[ImageType, Callable[[], LandmarkDetector]] = {}
        self._instances: dict[ImageType, LandmarkDetector] = {}

    def register(self, image_type: ImageType, builder: Callable[[], LandmarkDetector]) -> None:
        self._builders[image_type] = builder
        self._instances.pop(image_type, None)

    def create(self, image_type: ImageType) -> LandmarkDetector:
        if image_type not in self._instances:
            try:
                builder = self._builders[image_type]
            except KeyError as exc:
                raise UnsupportedImageTypeError(image_type) from exc
            self._instances[image_type] = builder()
        return self._instances[image_type]


def _build_heatmap_backend(image_type: ImageType, weights_path: str | None) -> LandmarkModelBackend:
    if not weights_path:
        return NotConfiguredModelBackend(image_type)
    return HRNetModelBackend(num_landmarks=len(LANDMARK_SCHEMAS[image_type]), weights_path=weights_path)


def build_default_factory(config: LandmarkServiceConfig | None = None) -> LandmarkDetectorFactory:
    config = config or LandmarkServiceConfig()
    factory = LandmarkDetectorFactory()

    for image_type, input_size in HEATMAP_TYPE_INPUT_SIZES.items():
        weights_path = config.weights_paths.get(image_type)
        factory.register(
            image_type,
            lambda image_type=image_type, input_size=input_size, weights_path=weights_path: HeatmapLandmarkDetector(
                image_type=image_type,
                landmark_names=LANDMARK_SCHEMAS[image_type],
                preprocessor=TorchvisionPreprocessor(input_size=input_size),
                backend=_build_heatmap_backend(image_type, weights_path),
            ),
        )

    for image_type in FACIAL_PHOTO_TYPES:
        factory.register(
            image_type,
            lambda image_type=image_type: FaceMeshFacialLandmarkDetector(
                image_type=image_type,
                landmark_names=LANDMARK_SCHEMAS[image_type],
            ),
        )

    return factory
