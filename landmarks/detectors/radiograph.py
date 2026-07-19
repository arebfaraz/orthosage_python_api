"""Generic detector for any heatmap-regression model: lateral/PA cephalograms,
panoramic radiographs, and (once trained) intraoral/occlusal views. Behaviour
is fixed here; what varies per image type -- schema, backend, input size --
is injected through the constructor (Open/Closed, Single Responsibility).
"""

from __future__ import annotations

from PIL import Image

from ..backends import LandmarkModelBackend
from ..detector import LandmarkDetector
from ..preprocessing import ImagePreprocessor
from ..types import ImageType, Landmark, LandmarkResult


class HeatmapLandmarkDetector(LandmarkDetector):
    def __init__(
        self,
        image_type: ImageType,
        landmark_names: list[str],
        preprocessor: ImagePreprocessor,
        backend: LandmarkModelBackend,
    ):
        self._image_type = image_type
        self._landmark_names = landmark_names
        self._preprocessor = preprocessor
        self._backend = backend

    def detect(self, image: Image.Image) -> LandmarkResult:
        original_width, original_height = image.size
        input_tensor = self._preprocessor(image)
        heatmaps = self._backend.predict_heatmaps(input_tensor)

        if heatmaps.shape[0] != len(self._landmark_names):
            raise ValueError(
                f"Model for '{self._image_type}' produced {heatmaps.shape[0]} heatmaps "
                f"but the schema defines {len(self._landmark_names)} landmark names."
            )

        input_size = self._preprocessor.input_size
        heatmap_h, heatmap_w = heatmaps.shape[1], heatmaps.shape[2]

        landmarks = []
        for name, heatmap in zip(self._landmark_names, heatmaps):
            peak_index = heatmap.argmax().item()
            heatmap_y = peak_index // heatmap_w
            heatmap_x = peak_index % heatmap_w

            # heatmap space -> model input space -> original image space
            input_x = (heatmap_x + 0.5) * (input_size / heatmap_w)
            input_y = (heatmap_y + 0.5) * (input_size / heatmap_h)
            x = input_x * (original_width / input_size)
            y = input_y * (original_height / input_size)

            landmarks.append(Landmark(name=name, x=float(x), y=float(y), confidence=float(heatmap.max())))

        return LandmarkResult(image_type=self._image_type, landmarks=landmarks)
