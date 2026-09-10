"""Evaluate the splitted-mask model: segmentation F1 + CCA tooth-count accuracy,
and render a preview. Runs on CPU (keeps the GPU free)."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import f1_score

from train import load_dataset
from cca import cca_analysis

TEST_FROM = 105
model = tf.keras.models.load_model("dental_xray_seg.h5", compile=False)

X, Y = load_dataset((512, 512), masks="splitted")
Xf = np.float32(X / 255.0)
x_test, y_test = Xf[TEST_FROM:], Y[TEST_FROM:]

pred = model.predict(x_test, batch_size=4, verbose=0)

# --- segmentation score ---
f1 = f1_score((pred > 0.5).astype(int).flatten(),
              (y_test > 127).astype(int).flatten(), average="micro")
print(f"segmentation micro-F1 (thr 0.5): {f1:.4f}")


def count_components(binary_uint8, erode_iter=1, area_frac=0.0008):
    b = cv2.erode(binary_uint8, np.ones((3, 3), "uint8"), iterations=erode_iter)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    amin = area_frac * binary_uint8.size
    return int((stats[1:, cv2.CC_STAT_AREA] > amin).sum())


rows = []
for i in range(len(x_test)):
    gt_ct = count_components((y_test[i, :, :, 0] > 127).astype("uint8"))
    pm = (pred[i, :, :, 0] > 0.5).astype("uint8") * 255
    pred_ct = count_components((pm > 0).astype("uint8"))
    # CCA (as the API does it) on a 512-res mask
    bgr = cv2.cvtColor((x_test[i, :, :, 0] * 255).astype("uint8"), cv2.COLOR_GRAY2BGR)
    _, cca_ct = cca_analysis(bgr, cv2.cvtColor(pm, cv2.COLOR_GRAY2BGR),
                             erode_iteration=1, open_iteration=1,
                             area_threshold=int(0.0008 * pm.size), annotate=False)
    rows.append((gt_ct, pred_ct, cca_ct))
    print(f"  test {i:2d}: GT={gt_ct:2d}  pred-CC={pred_ct:2d}  CCA={cca_ct:2d}")

rows = np.array(rows)
mae = np.abs(rows[:, 0] - rows[:, 2]).mean()
print(f"tooth-count MAE (CCA vs GT): {mae:.2f}")

# --- preview ---
k = len(x_test)
fig, ax = plt.subplots(3, k, figsize=(2.4 * k, 7.5))
for i in range(k):
    pm = (pred[i, :, :, 0] > 0.5).astype("uint8") * 255
    bgr = cv2.cvtColor((x_test[i, :, :, 0] * 255).astype("uint8"), cv2.COLOR_GRAY2BGR)
    cca_img, ct = cca_analysis(bgr.copy(), cv2.cvtColor(pm, cv2.COLOR_GRAY2BGR),
                               erode_iteration=1, open_iteration=1,
                               area_threshold=int(0.0008 * pm.size))
    ax[0, i].imshow(x_test[i, :, :, 0], cmap="gray"); ax[0, i].set_title(f"test {i}")
    ax[1, i].imshow(pm, cmap="gray")
    ax[2, i].imshow(cv2.cvtColor(cca_img, cv2.COLOR_BGR2RGB)); ax[2, i].set_title(f"{ct} teeth")
    for r in range(3):
        ax[r, i].set_xticks([]); ax[r, i].set_yticks([])
ax[0, 0].set_ylabel("input"); ax[1, 0].set_ylabel("predicted mask"); ax[2, 0].set_ylabel("CCA + count")
fig.suptitle("Splitted-mask U-Net — held-out test set", fontsize=14)
fig.tight_layout()
fig.savefig("counting_preview.png", dpi=85)
print("wrote counting_preview.png")
