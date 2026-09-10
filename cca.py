# -*- coding: utf-8 -*-
"""
Connected-Component / morphological post-processing for the U-Net teeth mask.

This is a dependency-light rewrite of the repo's CCA_Analysis.py: the only thing
that file needed from `imutils` was `perspective.order_points`, which is vendored
here so the module imports cleanly on modern setuptools.
"""
import cv2
import numpy as np
from scipy.spatial import distance as dist


def order_points(pts):
    """imutils.perspective.order_points, vendored (tl, tr, br, bl)."""
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most
    d = dist.cdist(tl[np.newaxis], right_most, "euclidean")[0]
    (br, tr) = right_most[np.argsort(d)[::-1], :]
    return np.array([tl, tr, br, bl], dtype="float32")


def midpoint(a, b):
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def cca_analysis(orig_image, predict_image, erode_iteration=3, open_iteration=2,
                 area_threshold=2000, pixels_per_metric=1.0, annotate=True):
    """
    orig_image     : BGR uint8 image to draw on (any size, must match predict_image)
    predict_image  : BGR uint8 mask from the network, same size as orig_image
    returns        : (annotated_bgr_image, teeth_count)
    """
    kernel = np.ones((5, 5), dtype=np.uint8)
    sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    image = predict_image.copy()
    canvas = orig_image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=open_iteration)
    image = cv2.filter2D(image, -1, sharpen)
    image = cv2.erode(image, kernel, iterations=erode_iteration)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    labels = cv2.connectedComponents(thresh, connectivity=8)[1]

    teeth_count = 0
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = np.zeros(thresh.shape, dtype="uint8")
        mask[labels == label] = 255
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = cnts[0]
        if cv2.contourArea(c) <= area_threshold:
            continue
        teeth_count += 1

        if not annotate:
            continue

        box = cv2.boxPoints(cv2.minAreaRect(c))
        box = order_points(np.array(box, dtype="int").astype("float32"))
        color = [int(v) for v in np.random.choice(range(150), size=3)]
        cv2.drawContours(canvas, [box.astype("int")], 0, color, 2)
        (tl, tr, br, bl) = box
        (tltr_x, tltr_y) = midpoint(tl, tr)
        (blbr_x, blbr_y) = midpoint(bl, br)
        (tlbl_x, tlbl_y) = midpoint(tl, bl)
        (trbr_x, trbr_y) = midpoint(tr, br)
        for (px, py) in ((tltr_x, tltr_y), (blbr_x, blbr_y),
                         (tlbl_x, tlbl_y), (trbr_x, trbr_y)):
            cv2.circle(canvas, (int(px), int(py)), 5, (255, 0, 0), -1)
        cv2.line(canvas, (int(tltr_x), int(tltr_y)), (int(blbr_x), int(blbr_y)), color, 2)
        cv2.line(canvas, (int(tlbl_x), int(tlbl_y)), (int(trbr_x), int(trbr_y)), color, 2)
        d_a = dist.euclidean((tltr_x, tltr_y), (blbr_x, blbr_y)) * pixels_per_metric
        d_b = dist.euclidean((tlbl_x, tlbl_y), (trbr_x, trbr_y)) * pixels_per_metric
        cv2.putText(canvas, f"{d_a:.1f}px", (int(tltr_x - 15), int(tltr_y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(canvas, f"{d_b:.1f}px", (int(trbr_x + 10), int(trbr_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    return canvas, teeth_count
