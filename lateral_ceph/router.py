"""
lateral_ceph/router.py
──────────────────────
Lateral-cephalogram landmark detection (29 landmarks, HRNet), ported from
orthosage_python2_api's POST /predict (previously served on py1.ekict.com).
Request params and response body are unchanged; only the path is new.

The model is loaded once at startup (see load_models, called from api.py).
If the weights are missing, the rest of the API still starts and this
endpoint returns 503.

Environment:
    LATERAL_CEPH_WEIGHTS  checkpoint path (default: lateral_ceph/weights/best.pt,
                          i.e. py1's runs/train/full_run/best.pt)
    LATERAL_CEPH_DEVICE   cpu | cuda (default: cuda if available, else cpu)

Endpoint:
    POST /analyze/lateral-ceph   multipart "file"
        ?format=json   (default) {"landmarks": {sym: {x, y, label}}, "width", "height"}
        ?format=image  PNG with the landmarks drawn on it
"""

import io
import os
import threading
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
WEIGHTS = Path(os.environ.get("LATERAL_CEPH_WEIGHTS", HERE / "weights" / "best.pt"))

router = APIRouter(tags=["lateral-ceph"])

STATE: dict = {"model": None, "device": None, "imgsz": 512, "error": None,
               "lock": threading.Lock()}


def load_models() -> None:
    """Load the HRNet checkpoint. Never raises: on failure the reason is kept
    in STATE["error"] and the endpoint answers 503."""
    try:
        import torch

        from .hrnet import build_hrnet
        from .landmarks import NUM_LANDMARKS

        if not WEIGHTS.exists():
            raise FileNotFoundError(f"weights not found: {WEIGHTS}")
        device = torch.device(os.environ.get("LATERAL_CEPH_DEVICE")
                              or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Our own training checkpoint ({"model", "args", ...}), so a full
        # unpickle is fine; weights_only=True rejects its non-tensor entries.
        ckpt = torch.load(WEIGHTS, map_location=device, weights_only=False)
        train_args = ckpt.get("args", {}) or {}
        model_name = train_args.get("model", "hrnet_w32")
        model = build_hrnet(model_name, num_landmarks=NUM_LANDMARKS)
        model.load_state_dict(ckpt["model"])
        STATE.update(model=model.to(device).eval(), device=device,
                     imgsz=train_args.get("imgsz", 512), error=None)
        print(f"[lateral-ceph] loaded {model_name} from {WEIGHTS} on {device}")
    except Exception as e:  # noqa: BLE001
        STATE["error"] = f"{type(e).__name__}: {e}"
        print(f"[lateral-ceph] WARNING: model not ready: {STATE['error']}")


def status() -> dict:
    """Summary for the main /health endpoint."""
    return {
        "weights": WEIGHTS.name,
        "device": str(STATE["device"]) if STATE["device"] else None,
        "model_loaded": STATE["model"] is not None,
        "error": STATE["error"],
    }


@router.post("/analyze/lateral-ceph")
def analyze_lateral_ceph(file: UploadFile = File(...),
                         format: str = Query("json", pattern="^(json|image)$")):
    import torch

    from .landmarks import (COLORS, LANDMARK_LABELS, LANDMARK_SYMBOLS,
                            decode_heatmaps, heatmap_to_image_coords,
                            hex_to_rgb, image_to_original_coords)

    model = STATE["model"]
    if model is None:
        raise HTTPException(503, f"lateral-ceph model not loaded: {STATE['error']}")

    try:
        img = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"invalid image: {e}")

    device, imgsz = STATE["device"], STATE["imgsz"]
    orig_w, orig_h = img.size
    img_resized = img.resize((imgsz, imgsz), Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with STATE["lock"], torch.no_grad():
        hm_pred = model(tensor)

    hm_coords = decode_heatmaps(hm_pred.float().cpu())
    img_coords = heatmap_to_image_coords(hm_coords, imgsz)
    orig_size = torch.tensor([[orig_h, orig_w]], dtype=torch.float32)
    orig_coords = image_to_original_coords(img_coords, orig_size, imgsz)[0].numpy()

    if format == "image":
        draw = ImageDraw.Draw(img)
        for i in range(len(orig_coords)):
            color = hex_to_rgb(COLORS[i % len(COLORS)])
            px, py = float(orig_coords[i, 0]), float(orig_coords[i, 1])
            r = 6
            draw.ellipse([(px - r, py - r), (px + r, py + r)], fill=color,
                         outline="white", width=1)
            draw.text((px + r + 2, py - 6), LANDMARK_SYMBOLS[i], fill=color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(buf.getvalue(), media_type="image/png")

    landmarks = {
        sym: {
            "x": float(orig_coords[i, 0]),
            "y": float(orig_coords[i, 1]),
            "label": LANDMARK_LABELS[sym],
        }
        for i, sym in enumerate(LANDMARK_SYMBOLS)
    }
    return JSONResponse({"landmarks": landmarks, "width": orig_w, "height": orig_h})
