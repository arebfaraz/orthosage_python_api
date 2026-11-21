"""Instance masks to polygons, in the uploaded image's coordinates.

Taken from the IOP-Compass viewer backend (``app/backend/contours.py``), keeping only
what the API needs. Polygons use the flat ``[x0, y0, x1, y1, ...]`` layout.
"""

from __future__ import annotations

import cv2
import numpy as np

# Douglas-Peucker tolerance as a fraction of the contour perimeter.
SIMPLIFY_EPS_FRAC = 0.002

# Contours below this many points, or this area in original-resolution pixels, are
# decoder speckle rather than a tooth outline.
MIN_POLYGON_POINTS = 3
MIN_POLYGON_AREA_PX = 12.0


def mask_to_polygons(mask: np.ndarray, scale_x: float = 1.0, scale_y: float = 1.0) -> list[list[float]]:
    """Outer contours of ``mask``, scaled into the original image's coordinates."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if not binary.any():
        return []

    found, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []
    for contour in found:
        if len(contour) < MIN_POLYGON_POINTS:
            continue
        eps = SIMPLIFY_EPS_FRAC * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, eps, True)
        if len(simplified) < MIN_POLYGON_POINTS:
            simplified = contour
        points = simplified.reshape(-1, 2).astype(np.float64)
        points[:, 0] *= scale_x
        points[:, 1] *= scale_y
        if polygon_area(points) < MIN_POLYGON_AREA_PX:
            continue
        polygons.append([round(float(v), 2) for v in points.reshape(-1)])
    return polygons


def mirror_polygons_x(polygons: list[list[float]], width: int) -> list[list[float]]:
    """Reflect ``polygons`` horizontally within a ``width``-wide image."""
    limit = float(width - 1)
    out: list[list[float]] = []
    for flat in polygons:
        mirrored = list(flat)
        mirrored[0::2] = [round(limit - float(x), 2) for x in flat[0::2]]
        out.append(mirrored)
    return out


def flat_to_points(flat: list[float]) -> np.ndarray:
    """``[x0, y0, x1, y1, ...]`` to an ``(n, 2)`` array, dropping a trailing odd value."""
    values = np.asarray(flat, dtype=np.float64).reshape(-1)
    return values[: len(values) - len(values) % 2].reshape(-1, 2)


def polygon_area(points: np.ndarray) -> float:
    """Absolute shoelace area of a closed polygon."""
    if len(points) < MIN_POLYGON_POINTS:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) / 2.0)


def polygons_area(polygons: list[list[float]]) -> float:
    """Total area of a multi-contour instance."""
    return float(sum(polygon_area(flat_to_points(p)) for p in polygons))


def polygons_bbox(polygons: list[list[float]]) -> tuple[float, float, float, float]:
    """``(x_min, y_min, x_max, y_max)`` over every contour of an instance."""
    points = np.concatenate(
        [flat_to_points(p) for p in polygons if len(p) >= 2 * MIN_POLYGON_POINTS]
        or [np.zeros((1, 2))]
    )
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def polygons_centroid(polygons: list[list[float]]) -> tuple[float, float]:
    """Area-weighted centroid, falling back to the vertex mean for degenerate input."""
    total_area = 0.0
    cx = cy = 0.0
    vertices = []
    for flat in polygons:
        points = flat_to_points(flat)
        vertices.append(points)
        area = polygon_area(points)
        if area <= 0:
            continue
        centre = points.mean(axis=0)
        cx += centre[0] * area
        cy += centre[1] * area
        total_area += area
    if total_area > 0:
        return cx / total_area, cy / total_area
    if vertices:
        stacked = np.concatenate(vertices)
        if len(stacked):
            return float(stacked[:, 0].mean()), float(stacked[:, 1].mean())
    return 0.0, 0.0
