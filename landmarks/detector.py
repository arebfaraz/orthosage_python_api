"""The single interface every concrete detector implements. Callers (the
service facade, a FastAPI route, a test) depend only on this abstraction --
Dependency Inversion -- so new image types or backends can be added without
touching calling code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

from .types import LandmarkResult


class LandmarkDetector(ABC):
    @abstractmethod
    def detect(self, image: Image.Image) -> LandmarkResult:
        """Runs on a single decoded RGB image and returns its named landmarks."""
