"""Pluggable model backends. A backend turns a preprocessed input tensor into
per-landmark heatmaps; everything else (preprocessing, decoding heatmaps into
image-space coordinates, naming) lives in the detector, not here.

This indirection (Dependency Inversion) is what lets the same
HeatmapLandmarkDetector serve lateral cephalograms today (a real trained
HRNet-W32 checkpoint, ai-service/weights/hrnet_w32_ceph19_state_dict.pth) and
PA cephalograms / panoramic / intraoral / occlusal views tomorrow, once a
checkpoint has actually been trained for them, without changing detector code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from model import HRNET_W32_CFG, HRNetW32

from .errors import ModelNotConfiguredError
from .types import ImageType


class LandmarkModelBackend(ABC):
    """Strategy interface: preprocessed tensor in, per-landmark heatmaps out."""

    @abstractmethod
    def predict_heatmaps(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """input_tensor: (1, 3, H, W). Returns (num_landmarks, heatmap_h, heatmap_w)."""


class HRNetModelBackend(LandmarkModelBackend):
    """Wraps the HRNet-W32 heatmap-regression architecture with a trained checkpoint.

    NUM_JOINTS is derived from the schema's landmark count so the same
    architecture can back any image type once a matching checkpoint exists --
    only the final 1x1 conv layer's channel count depends on it.
    """

    def __init__(self, num_landmarks: int, weights_path: str, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = {**HRNET_W32_CFG, "NUM_JOINTS": num_landmarks}
        self.model = HRNetW32(cfg)

        # weights_only=True is safe by construction: it refuses to unpickle
        # anything but plain tensors, so it can't execute arbitrary code even
        # if the checkpoint file were tampered with.
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_heatmaps(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(input_tensor.to(self.device))[0]


class NotConfiguredModelBackend(LandmarkModelBackend):
    """Placeholder for image types with no trained model yet.

    Fails loudly and explicitly on first use instead of guessing coordinates,
    so a missing model can never be mistaken for a working (if inaccurate) one.
    """

    def __init__(self, image_type: ImageType):
        self._image_type = image_type

    def predict_heatmaps(self, input_tensor: torch.Tensor) -> torch.Tensor:
        raise ModelNotConfiguredError(self._image_type)
