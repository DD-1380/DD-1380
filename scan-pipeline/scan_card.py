import os

import cv2
import imutils
import numpy as np
from imutils import perspective

# Finds the card in a photo and flattens it. Same steps and output as
# docscan.doc.scan (the flattened card, removed background set to black,
# encoded as PNG), but much faster:
#   -  background model is loaded once, not once per photo
#   -  model sees a downscaled copy (it works at 320x320 internally, and the
#     mask is only used to find the card's outline)
#   - the full-size photo is warped once
#
# SCAN_MASK_MODEL picks the rembg model. docscan passes none, so recent rembg
# versions use "bria-rmbg" (1 GB, slow on CPU). "u2net" (170 MB) is much faster
# and only has to find the card's outline.

APPROX_POLY_DP_ACCURACY_RATIO = 0.02
OUTLINE_HEIGHT = 500.0   # docscan finds the outline at this height
MASK_SIZE = 1024         # longest side of the copy the background model sees
MASK_MODEL = os.environ.get("SCAN_MASK_MODEL", "u2net")

_session = None


def get_session():
    global _session
    if _session is None:
        from rembg import new_session
        _session = new_session(MASK_MODEL)
    return _session


# Background mask (0 = background, 255 = card) at the photo's full size.
def card_mask(image: np.ndarray) -> np.ndarray:
    from rembg import remove

    h, w = image.shape[:2]
    scale = min(1.0, MASK_SIZE / max(h, w))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = remove(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), session=get_session(), only_mask=True)
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)


# The card's four corners in full-size pixels, or None if no 4-sided outline is found.
def card_outline(mask: np.ndarray):
    ratio = mask.shape[0] / OUTLINE_HEIGHT
    small = imutils.resize(mask, height=int(OUTLINE_HEIGHT))
    _, small = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY)
    small = cv2.medianBlur(small, 15)
    contours = imutils.grab_contours(cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        polygon = cv2.approxPolyDP(contour, APPROX_POLY_DP_ACCURACY_RATIO * cv2.arcLength(contour, True), True)
        if len(polygon) == 4:
            return polygon.reshape(4, 2) * ratio
    return None


def scan(data: bytes) -> bytes | None:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    mask = card_mask(image)
    # Same as rembg's cutout: background fades to black.
    image = cv2.multiply(image, cv2.merge([mask, mask, mask]), scale=1 / 255)
    outline = card_outline(mask)
    if outline is not None:
        image = perspective.four_point_transform(image, outline)
    _, buf = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return buf.tobytes()
