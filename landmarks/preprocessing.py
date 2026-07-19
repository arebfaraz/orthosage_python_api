"""Image preprocessing strategy used ahead of any heatmap-regression backend."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from PIL import Image
from torchvision import transforms


class ImagePreprocessor(ABC):
    @property
    @abstractmethod
    def input_size(self) -> int:
        """Side length (pixels) the backend's model was trained on."""

    @abstractmethod
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Returns a (1, 3, input_size, input_size) tensor ready for a model backend."""


class TorchvisionPreprocessor(ImagePreprocessor):
    """Resize + normalize, matching standard ImageNet-pretrained backbone stats."""

    def __init__(self, input_size: int, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._input_size = input_size
        self._transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ])

    @property
    def input_size(self) -> int:
        return self._input_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self._transform(image.convert("RGB")).unsqueeze(0)
