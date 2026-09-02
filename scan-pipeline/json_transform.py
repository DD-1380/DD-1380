from collections import defaultdict

import cv2
import numpy as np

# returns all words in the page.
def all_words(page):
    for block in page["blocks"]:
        for line in block["lines"]:
            yield from line["words"]

# returns the center of a word.
def center(word):
    (x0, y0), (x1, y1) = word["geometry"]
    return np.float32([(x0 + x1) / 2, (y0 + y1) / 2])


# Matches words between source and target by value,
# returns parallel arrays of x & y coordinates.
def matched_pairs(source, target):
    by_value = defaultdict(list)
    for w in all_words(target):
        by_value[w["value"]].append(w)
    src, tgt = [], []
    for w in all_words(source):
        cands = by_value.get(w["value"])
        if not cands:
            continue
        src_c = center(w)
        match = min(cands, key=lambda c: abs(center(c)[1] - src_c[1]))
        src.append(src_c)
        tgt.append(center(match))
    return np.float32(src), np.float32(tgt)

# converts normalized coordinates to pixel coordinates.
def norm_to_px(pts, hw):
    h, w = hw
    out = pts.copy()
    out[:, 0] *= w
    out[:, 1] *= h
    return out

# Takes two dicts of keypoint data and the scan shape,
def homography(source, target, scan_shape, out_shape=None):
    h_out, w_out = out_shape if out_shape is not None else source["dimensions"]
    src_pts, tgt_pts = matched_pairs(source, target)
    matrix, inliers = cv2.findHomography(
        norm_to_px(tgt_pts, scan_shape),
        norm_to_px(src_pts, (h_out, w_out)),
        cv2.RANSAC,
        5.0,
    )
    return matrix, inliers

# returns the corners of the page.
def block_corners(page):
    x0 = min(b["geometry"][0][0] for b in page["blocks"])
    y0 = min(b["geometry"][0][1] for b in page["blocks"])
    x1 = max(b["geometry"][1][0] for b in page["blocks"])
    y1 = max(b["geometry"][1][1] for b in page["blocks"])
    return np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

# Estimates a coarse page alignment and returns a 2x3 affine matrix. block_corners assumes that the outer
# bounds are a fair approximation of the page bounds. It may be worth making a more formal alignment.
# system for other documents. Although this works for dd-1380s
def affine_norm(source, target):
    return cv2.estimateAffinePartial2D(block_corners(source), block_corners(target))


# Applies an affine transformation and returns the transformed points.
def project_affine(points, matrix):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.transform(pts, matrix).reshape(-1, 2)
