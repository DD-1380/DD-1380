import base64
import json
import logging

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

BODY_CFG = {
    "min_blob_area": 15,
    "outline_dilate": 7,
    "merge_dilate": 7,
    "region_dilate": 5,
    "overlap_px": 10,
    "overlap_frac": 0.5,
}

_PREPARED_CACHE = {}


def _ellipse(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _decode_template(value):
    encoded = value.split(",", 1)[-1]
    data = base64.b64decode(encoded)
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)


def _view_definition(source_json, view_name):
    for key in (
        f"body_diagram_{view_name}",
        view_name,
    ):
        value = source_json.get(key)
        if isinstance(value, dict):
            return value
    return None


def _prepare(source_json):
    cache_key = json.dumps(
        {key: _view_definition(source_json, key) for key in ("front", "back")},
        sort_keys=True,
    )
    if cache_key in _PREPARED_CACHE:
        return _PREPARED_CACHE[cache_key]

    prepared = {}
    for view_name in ("front", "back"):
        view = _view_definition(source_json, view_name)
        if not view:
            continue
        bbox = view.get("bbox")
        regions = view.get("regions")
        template_b64 = view.get("template_png_b64")
        if not bbox or not regions or not template_b64:
            LOGGER.warning(
                "Skipping body diagram %s: bbox, regions, and template are required",
                view_name,
            )
            continue

        try:
            template = _decode_template(template_b64)
            if template is None:
                raise ValueError("template is not a readable image")
            region_masks = {}
            for region in regions:
                polygon = np.asarray(region["polygon"], dtype=np.int32)
                if view.get("coordinate_frame", "full_page_aligned") != "crop_local":
                    polygon = polygon - np.asarray(bbox[:2], dtype=np.int32)
                region_masks[region["name"]] = polygon
            prepared[view_name] = {
                "bbox": bbox,
                "template": template,
                "polygons": region_masks,
                "region_masks": _region_masks(region_masks, template.shape),
            }
        except (KeyError, TypeError, ValueError, base64.binascii.Error) as error:
            LOGGER.warning("Skipping body diagram %s: invalid source data: %s", view_name, error)

    _PREPARED_CACHE[cache_key] = prepared
    return prepared


def _region_masks(polygons, shape):
    masks = {}
    for name, polygon in polygons.items():
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        masks[name] = cv2.dilate(mask, _ellipse(BODY_CFG["region_dilate"]))
    return masks


def classify_view(crop_gray, template_gray, region_masks, cfg=BODY_CFG):
    if template_gray.shape != crop_gray.shape:
        LOGGER.warning("Resizing body diagram template from %s to %s", template_gray.shape, crop_gray.shape)
        template_gray = cv2.resize(
            template_gray, crop_gray.shape[::-1], interpolation=cv2.INTER_NEAREST,
        )

    _, template_ink = cv2.threshold(template_gray, 127, 255, cv2.THRESH_BINARY_INV)
    fat_template = cv2.dilate(template_ink, _ellipse(cfg["outline_dilate"]))
    blurred = cv2.GaussianBlur(crop_gray, (3, 3), 0)
    _, scan = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    ink = cv2.bitwise_and(scan, cv2.bitwise_not(fat_template))
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, _ellipse(3))
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, _ellipse(3))

    merged = cv2.dilate(ink, _ellipse(cfg["merge_dilate"]))
    count, labels = cv2.connectedComponents(merged, connectivity=8)
    labels_ink = np.where(ink > 0, labels, 0)
    area = np.bincount(labels_ink.ravel(), minlength=count)
    valid = area >= cfg["min_blob_area"]
    valid[0] = False

    results = {}
    for name, mask in region_masks.items():
        overlap = np.bincount(labels_ink[mask > 0], minlength=count)
        need = np.maximum(1, np.minimum(cfg["overlap_px"], cfg["overlap_frac"] * area))
        hit = valid & (overlap >= need)
        results[name] = {
            "marked": bool(hit.any()),
            "overlap_px": int(overlap[valid].max()) if valid.any() else 0,
        }
    return results


def process_body_diagram(aligned_page, source_json):
    prepared = _prepare(source_json)
    results = {}
    height, width = aligned_page.shape[:2]
    for view_name, view in prepared.items():
        x1, y1, x2, y2 = [int(round(value)) for value in view["bbox"]]
        left, top = max(0, x1), max(0, y1)
        right, bottom = min(width, x2), min(height, y2)
        if right <= left or bottom <= top:
            LOGGER.warning("Skipping body diagram %s: bbox is outside the aligned page", view_name)
            continue
        crop = aligned_page[top:bottom, left:right]
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        masks = view["region_masks"]
        if crop_gray.shape != view["template"].shape:
            masks = _region_masks(view["polygons"], crop_gray.shape)
        results[view_name] = classify_view(crop_gray, view["template"], masks)
    return results
