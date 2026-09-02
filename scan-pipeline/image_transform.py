import os

import cv2

from json_transform import homography

WARP_SCALE = float(os.environ.get("WARP_SCALE", "4"))

# Applies a homography transformation to the scan image by using the source and target json objects.
# returns transformed image, scaled up from source
def warp_to_source(scan, source, target, scale: float = WARP_SCALE):
    h_out, w_out = source["dimensions"]
    h_out, w_out = round(h_out * scale), round(w_out * scale)
    matrix, _ = homography(source, target, scan.shape[:2], out_shape=(h_out, w_out))
    return cv2.warpPerspective(
        scan, matrix, (w_out, h_out),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
