"""Exceptions raised by the landmarks package. Kept distinct from generic
ValueError/KeyError so a FastAPI route (or any other caller) can map each
one to the right HTTP status without string-matching error messages.
"""

from __future__ import annotations


class LandmarkDetectionError(Exception):
    """Base class for all errors raised by the landmarks package."""


class UnsupportedImageTypeError(LandmarkDetectionError):
    def __init__(self, image_type):
        super().__init__(f"No landmark detector is registered for image type '{image_type}'.")
        self.image_type = image_type


class ModelNotConfiguredError(LandmarkDetectionError):
    def __init__(self, image_type):
        super().__init__(
            f"'{image_type}' requires a trained landmark model, but none is configured. "
            "Supply a weights path via LandmarkServiceConfig once a model has been "
            "trained for this image type -- this error exists so a missing model can "
            "never be silently mistaken for a working one."
        )
        self.image_type = image_type


class InvalidImageError(LandmarkDetectionError):
    """Raised when the uploaded bytes cannot be decoded as an image."""


class NoFaceDetectedError(LandmarkDetectionError):
    """Raised when a facial-photo detector cannot find a face to landmark."""
