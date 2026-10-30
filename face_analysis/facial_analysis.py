"""
Frontal Facial Photo Analysis

Separate from the lateral-cephalogram HRNet pipeline (predict.py / app.py's
/predict). This module takes a frontal face photo and returns a set of
*named, individually movable* landmarks plus a few derived reference lines
(midsagittal, horizontal/interpupillary, lip canting), so a frontend canvas
can render the auto-detected points, let the clinician drag any point to
correct it, and re-derive the lines/angles from the edited positions.

Detection uses MediaPipe's Tasks API (`FaceLandmarker`, mediapipe>=0.10) —
478 points: the 468-point face mesh plus 10 iris points. The .task model
bundle is downloaded once and cached under assets/face_landmarker.task.
Mesh indices are geometric, not clinically labelled, so LANDMARKS below is
a best-effort mapping onto conventional soft-tissue landmark names (pupils/
iris centers are exact; the rest are close approximations meant to be
nudged by a human on the canvas, not taken as ground truth).

Usage:
    from facial_analysis import analyze_face, recompute_lines

    result = analyze_face(image_bytes)
    # result["landmarks"] -> [{"key","label","group","x","y"}, ...]
    # result["lines"]     -> [{"key","label","points":[x1,y1,x2,y2],"angle_deg","offset_px"}, ...]

    # after the user drags points on the canvas:
    edited = recompute_lines(landmarks_dict, width, height)
"""

import math
import os
import ssl
import urllib.request
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

# Must be set before mediapipe is imported: its graph runtime otherwise tries
# to initialize an OpenGL ES context for image-conversion calculators even
# when the model delegate is CPU, which needs libGLESv2.so.2 (a Mesa/EGL
# library) — not present on a headless server with no GPU/X11 and often not
# installable without root. This env var skips that GPU init path entirely.
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # pragma: no cover
    _SSL_CONTEXT = None

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "face_landmarker.task")


# ──────────────────────────────────────────────────────────────────────────
# Landmark definitions
#
# key       : stable identifier used in JSON / by the frontend
# label     : human-readable name shown next to the point
# group     : which reference line(s) this point feeds
# mp_index  : index into MediaPipe FaceMesh's 468 points (+468-477 iris,
#             only present when refine_landmarks=True)
# ──────────────────────────────────────────────────────────────────────────
LANDMARKS = [
    # Midline (feeds the midsagittal line)
    {"key": "glabella",         "label": "Glabella",              "group": "midline", "mp_index": 9},
    {"key": "nasion",           "label": "Nasion",                "group": "midline", "mp_index": 6},
    {"key": "pronasale",        "label": "Nasal Tip (Pronasale)", "group": "midline", "mp_index": 1},
    {"key": "subnasale",        "label": "Subnasale",             "group": "midline", "mp_index": 2},
    {"key": "philtrum",         "label": "Philtrum / Cupid's Bow", "group": "midline", "mp_index": 0},
    {"key": "stomion",          "label": "Stomion",               "group": "midline", "mp_index": 13},
    {"key": "labrale_inferius", "label": "Labrale Inferius",      "group": "midline", "mp_index": 17},
    {"key": "menton",           "label": "Menton",                "group": "midline", "mp_index": 152},

    # Eyes (feed the horizontal reference line)
    {"key": "pupil_r",       "label": "Right Pupil",         "group": "horizontal", "mp_index": 468},
    {"key": "pupil_l",       "label": "Left Pupil",          "group": "horizontal", "mp_index": 473},
    {"key": "canthus_r_out", "label": "Right Outer Canthus", "group": "eye", "mp_index": 33},
    {"key": "canthus_r_in",  "label": "Right Inner Canthus", "group": "eye", "mp_index": 133},
    {"key": "canthus_l_in",  "label": "Left Inner Canthus",  "group": "eye", "mp_index": 362},
    {"key": "canthus_l_out", "label": "Left Outer Canthus",  "group": "eye", "mp_index": 263},

    # Nose width
    {"key": "alare_r", "label": "Right Alare", "group": "nose", "mp_index": 129},
    {"key": "alare_l", "label": "Left Alare",  "group": "nose", "mp_index": 358},

    # Mouth corners (feed the lip canting line)
    {"key": "cheilion_r", "label": "Right Mouth Corner (Cheilion)", "group": "lip", "mp_index": 61},
    {"key": "cheilion_l", "label": "Left Mouth Corner (Cheilion)",  "group": "lip", "mp_index": 291},

    # Jaw / face width (feed the lower-face taper lines)
    {"key": "zygion_r", "label": "Right Zygion (cheekbone width)", "group": "jaw", "mp_index": 234},
    {"key": "zygion_l", "label": "Left Zygion (cheekbone width)",  "group": "jaw", "mp_index": 454},
    {"key": "gonion_r", "label": "Right Gonion (jaw angle)",       "group": "jaw", "mp_index": 58},
    {"key": "gonion_l", "label": "Left Gonion (jaw angle)",        "group": "jaw", "mp_index": 288},
]

LANDMARK_BY_KEY = {lm["key"]: lm for lm in LANDMARKS}

# Points used to fit the midsagittal (vertical midline) line, top → bottom.
MIDLINE_KEYS = [
    "glabella", "nasion", "pronasale", "subnasale",
    "philtrum", "stomion", "labrale_inferius", "menton",
]

# ──────────────────────────────────────────────────────────────────────────
# Reference-line definitions (computed from the landmarks above)
# ──────────────────────────────────────────────────────────────────────────
LINE_DEFS = [
    {"key": "midsagittal",  "label": "Midsagittal Line",     "type": "vertical_fit", "keys": MIDLINE_KEYS, "color": "#f5a623"},
    {"key": "horizontal_ref", "label": "Horizontal Reference Line", "type": "pair", "keys": ["pupil_r", "pupil_l"], "color": "#2ecc71"},
    {"key": "lip_canting",  "label": "Lip Canting Line",     "type": "pair", "keys": ["cheilion_r", "cheilion_l"], "color": "#2ecc71"},
    {"key": "jaw_taper_r",  "label": "Right Jaw Taper",      "type": "pair", "keys": ["zygion_r", "gonion_r", "menton"], "color": "#e74c3c"},
    {"key": "jaw_taper_l",  "label": "Left Jaw Taper",       "type": "pair", "keys": ["zygion_l", "gonion_l", "menton"], "color": "#e74c3c"},
]


def _ensure_model_downloaded() -> str:
    if not os.path.exists(_MODEL_PATH):
        os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
        print(f"[facial_analysis] Downloading face landmark model to {_MODEL_PATH} ...")
        with urllib.request.urlopen(_MODEL_URL, context=_SSL_CONTEXT) as resp:
            data = resp.read()
        tmp_path = _MODEL_PATH + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(data)
        os.replace(tmp_path, _MODEL_PATH)  # atomic — avoids a half-written file on crash
    return _MODEL_PATH


_LANDMARKER = None  # lazily-created singleton, reused across requests


def _get_landmarker():
    global _LANDMARKER
    if mp is None:
        raise RuntimeError(
            "mediapipe is not installed. Run: pip install mediapipe"
        )
    if _LANDMARKER is None:
        from mediapipe.tasks.python import BaseOptions, vision

        base_options = BaseOptions(
            model_asset_path=_ensure_model_downloaded(),
            delegate=BaseOptions.Delegate.CPU,  # GPU/Metal delegate crashes on some macOS setups
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            running_mode=vision.RunningMode.IMAGE,
        )
        _LANDMARKER = vision.FaceLandmarker.create_from_options(options)
    return _LANDMARKER


def detect_landmarks(img: Image.Image) -> dict:
    """Run MediaPipe FaceLandmarker and return {key: {x, y}} in pixel coords."""
    w, h = img.size
    img_np = np.array(img.convert("RGB"))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)

    result = _get_landmarker().detect(mp_image)

    if not result.face_landmarks:
        raise ValueError("No face detected in the uploaded photo.")

    mesh = result.face_landmarks[0]  # list of NormalizedLandmark, len 478

    points = {}
    for lm in LANDMARKS:
        mp_lm = mesh[lm["mp_index"]]
        points[lm["key"]] = {"x": mp_lm.x * w, "y": mp_lm.y * h}
    return points


# ──────────────────────────────────────────────────────────────────────────
# Line / angle geometry
# ──────────────────────────────────────────────────────────────────────────

def _fit_vertical_line(pts):
    """Least-squares fit x = m*y + c through a set of near-vertical points.
    Returns (m, c, y_min, y_max)."""
    ys = np.array([p["y"] for p in pts], dtype=np.float64)
    xs = np.array([p["x"] for p in pts], dtype=np.float64)
    m, c = np.polyfit(ys, xs, 1)
    return float(m), float(c), float(ys.min()), float(ys.max())


def _line_angle_deg(p1, p2):
    """Angle of the line p1→p2 relative to horizontal, in degrees.
    Positive = right side (p2) higher than left side (p1)."""
    dx = p2["x"] - p1["x"]
    dy = p2["y"] - p1["y"]
    return math.degrees(math.atan2(-dy, dx))  # image y grows downward


def compute_lines(points: dict, height: int) -> list:
    """Given {key: {x,y}}, compute each line in LINE_DEFS. Points that are
    missing (e.g. user deleted one) are skipped for that line."""
    lines = []

    for spec in LINE_DEFS:
        pts = [points[k] for k in spec["keys"] if k in points]
        if len(pts) < 2:
            continue

        if spec["type"] == "vertical_fit":
            m, c, y0, y1 = _fit_vertical_line(pts)
            x0, x1 = m * 0 + c, m * height + c
            tilt_deg = math.degrees(math.atan(m))
            # Deviation of the lowest midline point (chin) from the fitted line
            chin = pts[-1]
            expected_x = m * chin["y"] + c
            offset_px = chin["x"] - expected_x
            lines.append({
                "key": spec["key"],
                "label": spec["label"],
                "color": spec["color"],
                "endpoints": [x0, 0, x1, height],
                "tilt_deg": round(tilt_deg, 2),
                "offset_px": round(float(offset_px), 1),
            })

        elif spec["type"] == "pair":
            p1, p2 = pts[0], pts[-1]
            angle = _line_angle_deg(p1, p2)
            lines.append({
                "key": spec["key"],
                "label": spec["label"],
                "color": spec["color"],
                "endpoints": [p1["x"], p1["y"], p2["x"], p2["y"]],
                "angle_deg": round(angle, 2),
            })

    return lines


# ──────────────────────────────────────────────────────────────────────────
# Drawing (optional annotated preview, same style as predict.py)
# ──────────────────────────────────────────────────────────────────────────

def draw_analysis(img: Image.Image, points: dict, lines: list) -> Image.Image:
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)

    for line in lines:
        x0, y0, x1, y1 = line["endpoints"]
        draw.line([(x0, y0), (x1, y1)], fill=line["color"], width=2)
        label = line.get("angle_deg", line.get("tilt_deg"))
        if label is not None:
            draw.text((x1 + 4, y1 - 10), f"{line['label']}: {label}°", fill=line["color"])

    for key, p in points.items():
        r = 4
        draw.ellipse([(p["x"] - r, p["y"] - r), (p["x"] + r, p["y"] + r)],
                     fill="#2b6cff", outline="white", width=1)

    return annotated


# ──────────────────────────────────────────────────────────────────────────
# Top-level entry points
# ──────────────────────────────────────────────────────────────────────────

def analyze_face(raw_bytes: bytes) -> dict:
    """Full pipeline: detect + compute lines. Returns a JSON-serializable dict."""
    img = Image.open(BytesIO(raw_bytes)).convert("RGB")
    w, h = img.size

    points = detect_landmarks(img)
    lines = compute_lines(points, h)

    landmarks_out = [
        {
            "key": lm["key"],
            "label": lm["label"],
            "group": lm["group"],
            "x": round(points[lm["key"]]["x"], 1),
            "y": round(points[lm["key"]]["y"], 1),
            "movable": True,
        }
        for lm in LANDMARKS
    ]

    return {
        "width": w,
        "height": h,
        "landmarks": landmarks_out,
        "lines": lines,
    }


def recompute_lines(landmarks: list, height: int) -> list:
    """Recompute lines from a (possibly user-edited) landmark list, e.g.
    [{"key": "pupil_r", "x": 120.0, "y": 88.5}, ...] sent back by the
    frontend after the clinician drags a point on the canvas."""
    points = {lm["key"]: {"x": lm["x"], "y": lm["y"]} for lm in landmarks}
    return compute_lines(points, height)
