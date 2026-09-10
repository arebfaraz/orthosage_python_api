# -*- coding: utf-8 -*-
"""
Local (Windows/GPU) training script for
"Semantic Segmentation of Teeth in Panoramic X-ray Image Using U-Net".

This is a standalone port of Main.ipynb (which was Colab-only). It:
  1. downloads the panoramic x-ray dataset (if missing),
  2. builds X (images) and Y (custom splitted masks) at 512x512,
  3. augments the training split,
  4. trains the U-Net from model.py,
  5. saves the model and reports a test F1 score.

Usage:
    python train.py                      # notebook defaults (200 epochs)
    python train.py --epochs 50 --batch-size 4
    python train.py --aug-rounds 2 --no-augment
"""

import os
import sys
import glob
import zipfile
import argparse

import numpy as np
from PIL import Image
from natsort import natsorted

# ---- Pillow >= 10 compatibility (Image.ANTIALIAS was removed) ----
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

HERE = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(HERE, "Data")
CUSTOM_MASKS_ZIP = os.path.join(HERE, "Custom_Masks", "splitted_masks.zip")


def _register_cuda_dlls():
    """TF 2.10 on Windows won't auto-find the nvidia-*-cu11 pip wheels; wire them
    onto the DLL search path before tensorflow is imported."""
    import importlib.util
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return
    nvidia_root = list(spec.submodule_search_locations)[0]
    for sub in os.listdir(nvidia_root):
        bin_dir = os.path.join(nvidia_root, sub, "bin")
        if os.path.isdir(bin_dir):
            os.add_dll_directory(bin_dir)
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


_register_cuda_dlls()


def log(msg):
    print(f"[train] {msg}", flush=True)


# ----------------------------------------------------------------------
# 1. Dataset
# ----------------------------------------------------------------------
# The dataset link baked into the original repo (an AWS S3 cache of the
# Mendeley "hxt48yk462" archive) now returns HTTP 403. The dataset author
# re-published the exact same 116 image/mask pairs as a HuggingFace dataset,
# which we use instead. Falls back to a local Data/Images + Custom_Masks
# layout if it already exists.
HF_PARQUET_URL = ("https://huggingface.co/datasets/SerdarHelli/"
                  "SegmentationOfTeethPanoramicXRayImages/resolve/main/"
                  "data/train-00000-of-00001-f8e051e737d32f09.parquet")
PARQUET_PATH = os.path.join(DATA_DIR, "teeth_segmentation.parquet")


def convert_one_channel(arr):
    if arr.ndim > 2:
        return arr[:, :, 0]
    return arr


def _download(url, dest):
    import urllib.request
    import shutil
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    log(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=180) as r:
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest + ".part", "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done/1e6:.1f}/{total/1e6:.1f} MB "
                          f"({100*done/total:.0f}%)", end="", flush=True)
        print()
    os.replace(dest + ".part", dest)
    log(f"saved -> {dest}")


def _resize_bin(pil_or_arr, resize_shape):
    if not isinstance(pil_or_arr, Image.Image):
        pil_or_arr = Image.fromarray(pil_or_arr)
    return np.asarray(pil_or_arr.convert("L").resize(resize_shape, Image.NEAREST))


def _recover_split_permutation(df):
    """The HF parquet rows are shuffled relative to the repo's 1..116 numbering.
    Match each HF label to a Custom_Masks/splitted_masks entry by IoU (Hungarian)
    so HF images can be paired with the *splitted* masks the paper trains on."""
    import io
    import json
    from natsort import natsorted as _ns
    from scipy.optimize import linear_sum_assignment

    cache = os.path.join(DATA_DIR, "hf_permutation.json")
    with zipfile.ZipFile(CUSTOM_MASKS_ZIP) as z:
        names = _ns([n for n in z.namelist() if n.lower().endswith(".png")])
        split_raw = [z.read(n) for n in names]

    if os.path.exists(cache):
        idx = json.load(open(cache))["hf_row_to_split_pos"]
        return split_raw, idx

    S = 128
    def small(arr):
        return (np.asarray(Image.open(io.BytesIO(arr)).convert("L")
                           .resize((S, S))) > 127).astype(np.float32)
    split = np.stack([small(r) for r in split_raw]).reshape(len(split_raw), -1)
    hf = np.stack([
        (np.asarray(Image.open(io.BytesIO(df.iloc[i]["label"]["bytes"]))
                    .convert("L").resize((S, S))) > 127).astype(np.float32)
        for i in range(len(df))]).reshape(len(df), -1)
    inter = hf @ split.T
    iou = inter / (hf.sum(1)[:, None] + split.sum(1)[None, :] - inter + 1e-6)
    rows, cols = linear_sum_assignment(-iou)
    idx = [0] * len(df)
    for r, c in zip(rows, cols):
        idx[r] = int(c)
    log(f"recovered HF->splitted permutation, mean IoU {iou[rows, cols].mean():.3f}")
    json.dump({"hf_row_to_split_pos": idx}, open(cache, "w"))
    return split_raw, idx


def load_dataset(resize_shape=(512, 512), masks="splitted"):
    """Return X (N,H,W,1) uint8 images and Y (N,H,W,1) uint8 masks.

    masks="splitted" : repo Custom_Masks (teeth separated -> enables tooth counting)
    masks="hf"       : the filled teeth masks bundled in the HF parquet
    """
    import io

    # -- Preferred: local folder layout (Data/Images + Custom_Masks/Masks) ----
    images_dir = os.path.join(DATA_DIR, "Images")
    if os.path.isdir(images_dir) and glob.glob(os.path.join(images_dir, "*")):
        return _load_from_folders(images_dir, resize_shape)

    # -- Otherwise: HuggingFace parquet -------------------------------------
    if not os.path.exists(PARQUET_PATH):
        _download(HF_PARQUET_URL, PARQUET_PATH)

    import pandas as pd
    df = pd.read_parquet(PARQUET_PATH)
    log(f"loaded parquet: {len(df)} image/mask pairs (masks={masks})")

    n = len(df)
    X = np.zeros((n, resize_shape[0], resize_shape[1], 1), dtype=np.uint8)
    Y = np.zeros((n, resize_shape[0], resize_shape[1], 1), dtype=np.uint8)

    split_raw = split_idx = None
    if masks == "splitted":
        split_raw, split_idx = _recover_split_permutation(df)

    for i in range(n):
        im = Image.open(io.BytesIO(df.iloc[i]["image"]["bytes"])).convert("L")
        X[i, :, :, 0] = np.asarray(im.resize(resize_shape, Image.ANTIALIAS))
        if masks == "splitted":
            lb = Image.open(io.BytesIO(split_raw[split_idx[i]]))
        else:
            lb = Image.open(io.BytesIO(df.iloc[i]["label"]["bytes"]))
        Y[i, :, :, 0] = _resize_bin(lb, resize_shape)
    return X, Y


def _load_from_folders(images_dir, resize_shape):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    img_files = natsorted([f for f in os.listdir(images_dir)
                           if f.lower().endswith(exts)])

    mask_dir = os.path.join(HERE, "Custom_Masks", "Masks")
    if not os.path.isdir(mask_dir) or not os.listdir(mask_dir):
        os.makedirs(mask_dir, exist_ok=True)
        with zipfile.ZipFile(CUSTOM_MASKS_ZIP) as z:
            z.extractall(mask_dir)
    mask_files = natsorted([f for f in os.listdir(mask_dir)
                            if f.lower().endswith(exts)])
    log(f"folder layout: {len(img_files)} images / {len(mask_files)} masks")

    n = min(len(img_files), len(mask_files))
    X = np.zeros((n, resize_shape[0], resize_shape[1], 1), dtype=np.uint8)
    Y = np.zeros((n, resize_shape[0], resize_shape[1], 1), dtype=np.uint8)
    for i in range(n):
        im = Image.open(os.path.join(images_dir, img_files[i]))
        im = im.resize(resize_shape, Image.ANTIALIAS)
        X[i, :, :, 0] = convert_one_channel(np.asarray(im))
        mk = Image.open(os.path.join(mask_dir, mask_files[i]))
        if mk.size != resize_shape:
            mk = mk.resize(resize_shape, Image.NEAREST)
        Y[i, :, :, 0] = convert_one_channel(np.asarray(mk))
    return X, Y


def _save_preview(X, Y, path, k=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = np.linspace(0, len(X) - 1, k).astype(int)
    fig, ax = plt.subplots(2, k, figsize=(3 * k, 6))
    for j, i in enumerate(idx):
        ax[0, j].imshow(X[i, :, :, 0], cmap="gray")
        ax[0, j].set_title(f"img {i}")
        ax[1, j].imshow(X[i, :, :, 0], cmap="gray")
        ax[1, j].imshow(Y[i, :, :, 0], cmap="jet", alpha=0.4)
        for a in (ax[0, j], ax[1, j]):
            a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
    log(f"wrote preview -> {path}")


# ----------------------------------------------------------------------
# 2. Augmentation
# ----------------------------------------------------------------------
def build_augmenter():
    try:
        import cv2
        import albumentations as A
    except Exception as e:  # noqa: BLE001
        log(f"albumentations unavailable ({e}); using numpy fallback augmentation")
        return None

    return A.Compose([
        A.OneOf([A.RandomCrop(width=512, height=512),
                 A.PadIfNeeded(min_height=512, min_width=512, p=0.5)], p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.325, scale_limit=0.15, rotate_limit=15,
                           border_mode=cv2.BORDER_CONSTANT, p=1),
        A.Rotate(15, p=0.5),
        A.Blur(blur_limit=3, p=0.5),
        A.GaussNoise(p=0.5),
        A.HorizontalFlip(p=0.25),
    ])


def numpy_augment(x, y):
    """Cheap fallback: flips + 90-degree-ish rotations + brightness jitter."""
    from scipy.ndimage import rotate
    if np.random.rand() < 0.5:
        x, y = x[:, ::-1], y[:, ::-1]
    if np.random.rand() < 0.5:
        ang = np.random.uniform(-15, 15)
        x = rotate(x, ang, reshape=False, order=1, mode="constant")
        y = rotate(y, ang, reshape=False, order=0, mode="constant")
    if np.random.rand() < 0.5:
        x = np.clip(x * np.random.uniform(0.75, 1.25), 0, 1)
    return x, y


def augment_training_set(x_train, y_train, rounds):
    aug = build_augmenter()
    x_all, y_all = [x_train], [y_train]
    base_x, base_y = np.copy(x_train), np.copy(y_train)
    for r in range(rounds):
        log(f"augmentation round {r + 1}/{rounds}")
        xa, ya = np.copy(base_x), np.copy(base_y)
        for i in range(len(base_x)):
            if aug is not None:
                res = aug(image=base_x[i], mask=base_y[i])
                xa[i], ya[i] = res["image"], res["mask"]
            else:
                xa[i], ya[i] = numpy_augment(base_x[i], base_y[i])
        x_all.append(xa)
        y_all.append(ya)
    return np.concatenate(x_all), np.concatenate(y_all)


# ----------------------------------------------------------------------
# 3. Main
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--aug-rounds", type=int, default=4)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--train-split", type=int, default=105)
    p.add_argument("--masks", choices=["splitted", "hf"], default="splitted",
                   help="'splitted' = repo Custom_Masks (teeth separated, enables "
                        "CCA tooth counting); 'hf' = filled masks from the parquet")
    p.add_argument("--output", default=os.path.join(HERE, "dental_xray_seg.h5"))
    args = p.parse_args()

    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    log(f"TensorFlow {tf.__version__}; GPUs visible: {gpus}")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:  # noqa: BLE001
            pass

    X, Y = load_dataset(resize_shape=(512, 512), masks=args.masks)
    log(f"X={X.shape}  Y={Y.shape}")

    # save a quick overlay so the image/mask pairing can be eyeballed
    try:
        _save_preview(X, Y, os.path.join(HERE, "data_preview.png"))
    except Exception as e:  # noqa: BLE001
        log(f"(preview skipped: {e})")

    X = np.float32(X / 255.0)
    Y = np.float32(Y / 255.0)

    s = args.train_split
    x_train, y_train = X[:s], Y[:s]
    x_test, y_test = X[s:], Y[s:]
    log(f"train={len(x_train)}  test={len(x_test)}")

    if not args.no_augment and args.aug_rounds > 0:
        x_train, y_train = augment_training_set(x_train, y_train, args.aug_rounds)
    log(f"training tensor: {x_train.shape}")

    from model import UNET
    model = UNET(input_shape=(512, 512, 1), last_activation="sigmoid")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    ckpt = os.path.join(HERE, "best_model.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="loss",
                                           save_best_only=True, verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(HERE, "training_log.csv")),
    ]

    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs,
              verbose=1, callbacks=callbacks)

    model.save(args.output)
    log(f"saved model -> {args.output}")

    if len(x_test):
        from sklearn.metrics import f1_score
        pred = model.predict(x_test, batch_size=args.batch_size)
        f1 = f1_score((pred > 0.25).astype(int).flatten(),
                      (y_test > 0.25).astype(int).flatten(), average="micro")
        log(f"test micro-F1 (thr=0.25): {f1:.4f}")


if __name__ == "__main__":
    sys.exit(main())
