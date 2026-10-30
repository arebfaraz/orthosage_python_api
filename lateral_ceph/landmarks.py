"""
Landmark names, colors and coordinate helpers for the lateral-ceph HRNet,
copied unchanged from orthosage_python2_api (data/dataset.py, predict.py,
utils/metrics.py) so the API output matches py1.ekict.com/predict exactly.
"""

import torch

LANDMARK_SYMBOLS = [
    'A', 'ANS', 'Ar', 'B', 'Co', 'Gn', 'Go',
    'LIA', 'LIT', 'LMT', 'LPM', 'Li', 'Ls',
    'Me', 'N', "N`", 'Or', 'PNS', 'Pn', 'Po',
    'Pog', "Pog`", 'R', 'S', 'Sn',
    'UIA', 'UIT', 'UMT', 'UPM',
]
NUM_LANDMARKS = 29

# Short-form labels for API consumers (OrthoSage landmark naming scheme).
LANDMARK_LABELS = {
    'A': 'A-Point', 'ANS': 'ANS', 'Ar': 'Articulare', 'B': 'B-Point',
    'Co': 'Condylion', 'Gn': 'Gnathion', 'Go': 'Gonion',
    'LIA': 'L1RootTip', 'LIT': 'L1IncisalTip',
    'LMT': 'Lower Molar Cusp Tip', 'LPM': 'Lower 2nd PM Cusp Tip',
    'Li': 'LabraleInferius', 'Ls': 'LabraleSuperius',
    'Me': 'Menton', 'N': 'Nasion', "N`": 'SoftTissueNa',
    'Or': 'Orbitale', 'PNS': 'PNS', 'Pn': 'Pronasale', 'Po': 'Porion',
    'Pog': 'Pogonion', "Pog`": 'SoftTissuePog', 'R': 'RamusDown',
    'S': 'Sella', 'Sn': 'Subnasale',
    'UIA': 'U1RootTip', 'UIT': 'U1IncisalTip',
    'UMT': 'Upper Molar Cusp Tip', 'UPM': 'Upper 2nd PM Cusp Tip',
}

# Distinct colors per landmark (hex)
COLORS = [
    '#FF5722', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0',
    '#00BCD4', '#E91E63', '#8BC34A', '#FFC107', '#607D8B',
    '#F44336', '#3F51B5', '#009688', '#FFEB3B', '#795548',
    '#03A9F4', '#8D6E63', '#9E9E9E', '#CDDC39', '#673AB7',
    '#FF6F00', '#1B5E20', '#880E4F', '#0D47A1', '#BF360C',
    '#004D40', '#311B92', '#B71C1C', '#33691E',
]


def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def decode_heatmaps(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Decode heatmaps to landmark coordinates using argmax.

    Args:
        heatmaps: (B, N, H, W)

    Returns:
        coords: (B, N, 2) in heatmap (x, y) space
    """
    B, N, H, W = heatmaps.shape
    flat = heatmaps.view(B, N, -1)
    idx = flat.argmax(dim=-1)              # (B, N)
    x = (idx % W).float()
    y = (idx // W).float()
    return torch.stack([x, y], dim=-1)    # (B, N, 2)


def heatmap_to_image_coords(heatmap_coords: torch.Tensor, imgsz: int) -> torch.Tensor:
    """
    Scale heatmap coordinates (at stride 4) to image-space coordinates.

    Args:
        heatmap_coords: (B, N, 2) or (N, 2)
        imgsz: input image size (both dims assumed equal)

    Returns:
        image_coords: same shape, scaled by 4
    """
    return heatmap_coords * 4.0


def image_to_original_coords(
    img_coords: torch.Tensor,
    orig_sizes: torch.Tensor,
    imgsz: int,
) -> torch.Tensor:
    """
    Scale coordinates from resized image space back to original image space.

    Args:
        img_coords: (B, N, 2) in [0, imgsz]
        orig_sizes: (B, 2) = [orig_h, orig_w]
        imgsz: training image size

    Returns:
        orig_coords: (B, N, 2) in original pixel space
    """
    scale_w = orig_sizes[:, 1:2] / imgsz     # (B, 1)
    scale_h = orig_sizes[:, 0:1] / imgsz     # (B, 1)
    scale = torch.stack([scale_w, scale_h], dim=-1)  # (B, 1, 2)
    return img_coords * scale

