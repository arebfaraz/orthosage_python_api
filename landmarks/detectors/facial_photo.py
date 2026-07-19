"""Facial-photo detector: derives soft-tissue, cephalometric-style landmarks
from Google's MediaPipe Face Mesh (468 3D face landmarks).

Honesty about accuracy: MediaPipe locates the mesh vertices themselves very
reliably, but the *mapping* from a mesh vertex to a named clinical point
(e.g. "Glabella (G)") is a community-referenced anthropometric approximation,
not a model trained on clinician-labeled cephalometric points. Two
consequences worth knowing before relying on this for measurement:

1. Points that sit in a soft-tissue concavity rather than on a mesh vertex
   (soft-tissue A'/B') are interpolated from neighbouring vertices, not
   directly detected.
2. MediaPipe Face Mesh does not model the ear, so tragus-based points (used
   for Frankfort-plane references in some photographic protocols) cannot be
   derived from it at all -- they are simply not in any schema here.

Validate index-to-landmark mapping against your own labeled photos before
using this for anything clinical.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..detector import LandmarkDetector
from ..errors import NoFaceDetectedError
from ..types import ImageType, Landmark, LandmarkResult

# Indices into MediaPipe's 468-point canonical face model. Assumes a
# non-mirrored photo, so "right"/"left" below are the subject's anatomical
# right/left (opposite sides of the image in a frontal shot).
_GLABELLA = 9
_SOFT_TISSUE_NASION = 168
_PRONASALE = 4
_SUBNASALE = 2
_UPPER_LIP = 0
_LOWER_LIP = 17
_SOFT_TISSUE_POGONION = 152
_SOFT_TISSUE_MENTON = 175
_RIGHT_OUTER_CANTHUS = 33
_RIGHT_INNER_CANTHUS = 133
_LEFT_INNER_CANTHUS = 362
_LEFT_OUTER_CANTHUS = 263
_RIGHT_ALAR_BASE = 98
_LEFT_ALAR_BASE = 327
_RIGHT_CHEILION = 61
_LEFT_CHEILION = 291


class FaceMeshLandmarkMapper:
    """Maps a named clinical point to a MediaPipe mesh vertex (or an
    interpolation of a few vertices). Single Responsibility: this class
    knows anatomy-to-mesh-vertex mapping and nothing about image IO,
    mediapipe's API, or model loading -- swapping in a better-calibrated
    mapping later means replacing this one class.
    """

    _DIRECT: dict[str, int] = {
        "Glabella (G)": _GLABELLA,
        "Soft Tissue Nasion (Na')": _SOFT_TISSUE_NASION,
        "Pronasale (Pn)": _PRONASALE,
        "Subnasale (Sn)": _SUBNASALE,
        "Upper Lip (UL)": _UPPER_LIP,
        "Lower Lip (LL)": _LOWER_LIP,
        "Soft Tissue Pogonion (Pog')": _SOFT_TISSUE_POGONION,
        "Soft Tissue Menton (Me')": _SOFT_TISSUE_MENTON,
        "Right Outer Canthus": _RIGHT_OUTER_CANTHUS,
        "Left Outer Canthus": _LEFT_OUTER_CANTHUS,
        "Right Inner Canthus": _RIGHT_INNER_CANTHUS,
        "Left Inner Canthus": _LEFT_INNER_CANTHUS,
        "Right Alar Base": _RIGHT_ALAR_BASE,
        "Left Alar Base": _LEFT_ALAR_BASE,
        "Right Cheilion": _RIGHT_CHEILION,
        "Left Cheilion": _LEFT_CHEILION,
        "Upper Lip Low Point": _UPPER_LIP,
        "Lower Lip High Point": _LOWER_LIP,
    }

    _MIDPOINTS: dict[str, tuple[int, int]] = {
        "Soft Tissue A-Point (A')": (_SUBNASALE, _UPPER_LIP),
        "Soft Tissue B-Point (B')": (_LOWER_LIP, _SOFT_TISSUE_POGONION),
        "Facial Midline (Sn-Me')": (_SUBNASALE, _SOFT_TISSUE_MENTON),
    }

    def point(self, name: str, mesh_points: np.ndarray) -> tuple[float, float]:
        if name in self._DIRECT:
            x, y = mesh_points[self._DIRECT[name]]
            return float(x), float(y)
        if name in self._MIDPOINTS:
            a_index, b_index = self._MIDPOINTS[name]
            a, b = mesh_points[a_index], mesh_points[b_index]
            return float((a[0] + b[0]) / 2), float((a[1] + b[1]) / 2)
        raise KeyError(f"No Face Mesh mapping is defined for landmark '{name}'.")


class FaceMeshFacialLandmarkDetector(LandmarkDetector):
    """Detects soft-tissue landmarks on frontal/smile/45-degree/lateral
    photos using MediaPipe Face Mesh + a FaceMeshLandmarkMapper. Which
    landmarks come out is entirely driven by `landmark_names` (from
    schema.py), so the same class serves every facial photo view.
    """

    def __init__(
        self,
        image_type: ImageType,
        landmark_names: list[str],
        mapper: FaceMeshLandmarkMapper | None = None,
    ):
        self._image_type = image_type
        self._landmark_names = landmark_names
        self._mapper = mapper or FaceMeshLandmarkMapper()
        self._face_mesh = self._build_face_mesh()

    @staticmethod
    def _build_face_mesh():
        import mediapipe as mp

        return mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    def detect(self, image: Image.Image) -> LandmarkResult:
        rgb = np.array(image.convert("RGB"))
        height, width = rgb.shape[:2]
        result = self._face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            raise NoFaceDetectedError(f"No face was detected in the supplied '{self._image_type}' image.")

        face = result.multi_face_landmarks[0]
        mesh_points = np.array([(landmark.x * width, landmark.y * height) for landmark in face.landmark])

        landmarks = []
        for name in self._landmark_names:
            x, y = self._mapper.point(name, mesh_points)
            landmarks.append(Landmark(name=name, x=x, y=y, confidence=1.0))

        return LandmarkResult(image_type=self._image_type, landmarks=landmarks)
