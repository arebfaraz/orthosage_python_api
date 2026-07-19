"""Named landmark schemas: the single source of truth for which points each
image type produces, and in what order. Order matters for heatmap-based
detectors -- it must match the channel order the underlying model was
trained with -- which is why LATERAL_CEPH_LANDMARKS is imported from model.py
rather than redeclared here.

Adding a new image type is a two-line change: a schema list here, and one
registration line in factory.py -- no existing detector code changes
(Open/Closed).
"""

from __future__ import annotations

from model import LANDMARK_LABELS as LATERAL_CEPH_LANDMARKS

from .types import ImageType

# --- Radiographs -----------------------------------------------------------
# LATERAL_CEPH has a real trained checkpoint (ai-service/weights/hrnet_w32_ceph19_state_dict.pth).
# PA_CEPH and ORTHOPAN schemas below reflect standard analyses (Ricketts/Grummons
# PA points; panoramic anatomic landmarks) but have no trained checkpoint yet --
# see backends.NotConfiguredModelBackend.

PA_CEPH_LANDMARKS = [
    "Crista Galli (CG)",
    "Nasion (N)",
    "Anterior Nasal Spine (ANS)",
    "Menton (Me)",
    "Zygomatic Right (Z-R)",
    "Zygomatic Left (Z-L)",
    "Antegonion Right (AG-R)",
    "Antegonion Left (AG-L)",
    "Jugal Right (J-R)",
    "Jugal Left (J-L)",
    "Latero-orbitale Right (LO-R)",
    "Latero-orbitale Left (LO-L)",
    "Upper Dental Midline",
    "Lower Dental Midline",
]

ORTHOPAN_LANDMARKS = [
    "Condyle Right",
    "Condyle Left",
    "Coronoid Process Right",
    "Coronoid Process Left",
    "Gonion Right",
    "Gonion Left",
    "Mental Foramen Right",
    "Mental Foramen Left",
    "Mandibular Canal Anterior Right",
    "Mandibular Canal Anterior Left",
    "Maxillary Sinus Floor Right",
    "Maxillary Sinus Floor Left",
    "Occlusal Plane Anterior",
    "Dental Midline",
]

# --- Facial photos -----------------------------------------------------------
# Derived from MediaPipe Face Mesh by detectors/facial_photo.py. Point names
# mirror the soft-tissue profile landmarks (G, Na', Pn, Sn, A', UL, LL, B',
# Pog', Me') used in standard lateral cephalometric soft-tissue analysis.

_FACIAL_SOFT_TISSUE_PROFILE = [
    "Glabella (G)",
    "Soft Tissue Nasion (Na')",
    "Pronasale (Pn)",
    "Subnasale (Sn)",
    "Soft Tissue A-Point (A')",
    "Upper Lip (UL)",
    "Lower Lip (LL)",
    "Soft Tissue B-Point (B')",
    "Soft Tissue Pogonion (Pog')",
    "Soft Tissue Menton (Me')",
]

FRONTAL_PHOTO_LANDMARKS = [
    "Glabella (G)",
    "Soft Tissue Nasion (Na')",
    "Right Outer Canthus",
    "Left Outer Canthus",
    "Right Inner Canthus",
    "Left Inner Canthus",
    "Right Alar Base",
    "Left Alar Base",
    "Subnasale (Sn)",
    "Right Cheilion",
    "Left Cheilion",
    "Soft Tissue Menton (Me')",
    "Facial Midline (Sn-Me')",
]

SMILE_PHOTO_LANDMARKS = [
    "Right Cheilion",
    "Left Cheilion",
    "Upper Lip Low Point",
    "Lower Lip High Point",
    "Subnasale (Sn)",
    "Soft Tissue Menton (Me')",
    "Facial Midline (Sn-Me')",
]

PHOTO_45_LANDMARKS = list(_FACIAL_SOFT_TISSUE_PROFILE)

LATERAL_PHOTO_LANDMARKS = list(_FACIAL_SOFT_TISSUE_PROFILE)

# --- Intraoral / occlusal photos --------------------------------------------
# No trained checkpoint yet -- see backends.NotConfiguredModelBackend. Schemas
# reflect the reference points typically digitized for arch-form/midline/
# occlusion analysis from intraoral photography.

RIGHT_INTRAORAL_LANDMARKS = [
    "Upper Canine Cusp Tip",
    "Upper First Molar Mesiobuccal Cusp",
    "Lower Canine Cusp Tip",
    "Lower First Molar Mesiobuccal Cusp",
    "Buccal Overjet Reference",
]

FRONT_INTRAORAL_LANDMARKS = [
    "Upper Dental Midline",
    "Lower Dental Midline",
    "Upper Central Incisor Right Mesial Contact",
    "Upper Central Incisor Left Mesial Contact",
    "Overbite Reference (Upper Incisor Edge)",
    "Overbite Reference (Lower Incisor Edge)",
]

LEFT_INTRAORAL_LANDMARKS = [
    "Upper Canine Cusp Tip",
    "Upper First Molar Mesiobuccal Cusp",
    "Lower Canine Cusp Tip",
    "Lower First Molar Mesiobuccal Cusp",
    "Buccal Overjet Reference",
]

UPPER_OCCLUSAL_LANDMARKS = [
    "Midpalatal Raphe Midline",
    "Incisive Papilla",
    "Right Canine Cusp Tip",
    "Left Canine Cusp Tip",
    "Right First Molar Mesiobuccal Cusp",
    "Left First Molar Mesiobuccal Cusp",
]

LOWER_OCCLUSAL_LANDMARKS = [
    "Lingual Midline",
    "Right Canine Cusp Tip",
    "Left Canine Cusp Tip",
    "Right First Molar Mesiobuccal Cusp",
    "Left First Molar Mesiobuccal Cusp",
]

LANDMARK_SCHEMAS: dict[ImageType, list[str]] = {
    ImageType.LATERAL_CEPH: LATERAL_CEPH_LANDMARKS,
    ImageType.PA_CEPH: PA_CEPH_LANDMARKS,
    ImageType.ORTHOPAN: ORTHOPAN_LANDMARKS,
    ImageType.FRONTAL_PHOTO: FRONTAL_PHOTO_LANDMARKS,
    ImageType.SMILE_PHOTO: SMILE_PHOTO_LANDMARKS,
    ImageType.PHOTO_45: PHOTO_45_LANDMARKS,
    ImageType.LATERAL_PHOTO: LATERAL_PHOTO_LANDMARKS,
    ImageType.RIGHT_INTRAORAL: RIGHT_INTRAORAL_LANDMARKS,
    ImageType.FRONT_INTRAORAL: FRONT_INTRAORAL_LANDMARKS,
    ImageType.LEFT_INTRAORAL: LEFT_INTRAORAL_LANDMARKS,
    ImageType.UPPER_OCCLUSAL: UPPER_OCCLUSAL_LANDMARKS,
    ImageType.LOWER_OCCLUSAL: LOWER_OCCLUSAL_LANDMARKS,
}
