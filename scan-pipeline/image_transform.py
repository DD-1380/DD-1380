import os

import cv2
import numpy as np

from json_transform import homography

WARP_SCALE = float(os.environ.get("WARP_SCALE", "4"))
ALIGN_METHODS = ("doctr", "features")

def align_method(method: str | None = None) -> str:
    chosen = method if method is not None else os.environ.get("ALIGN_METHOD", "features")
    chosen = chosen.strip().lower()
    if chosen not in ALIGN_METHODS:
        raise ValueError(f"Alignment must be 'doctr' or 'features', not {chosen!r}.")
    return chosen


def load_blank_form(side: str | None = None) -> np.ndarray | None:
    names = [f"ALIGN_SOURCE_{side.upper()}"] if side else []
    names.append("ALIGN_SOURCE")
    for name in names:
        path = os.environ.get(name)
        if not path:
            continue
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read the blank form image at {path}.")
        return image
    return None


# Applies a homography transformation to the scan image by using the source and target json objects.
# returns transformed image, scaled up from source
def warp_to_source(scan, source, target, scale: float = WARP_SCALE):
    h_out, w_out = source["dimensions"]
    h_out, w_out = round(h_out * scale), round(w_out * scale)
    matrix, _ = homography(source, target, scan.shape[:2], out_shape=(h_out, w_out))
    return cv2.warpPerspective(
        scan, matrix, (w_out, h_out),
        # switched from INTER_LINEAR to INTER_CUBIC: sharper results when upscaling
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

def warp_by_features(scan, source_image, source, scale: float = WARP_SCALE):
    h_out, w_out = source["dimensions"]
    h_out, w_out = round(h_out * scale), round(w_out * scale)
    aligned = align_to_source(source_image, scan, out_shape=(h_out, w_out))
    blank = cv2.resize(_bgr(source_image), (w_out, h_out), interpolation=cv2.INTER_AREA)
    return settle_to_source(blank, aligned)


def align_to_source(source, modified, out_shape=None) -> np.ndarray:
    source_gray = _gray(source)
    modified_gray = _gray(modified)
    matrix = _form_homography(source_gray, modified_gray)
    height, width = source_gray.shape
    if out_shape is not None:
        out_h, out_w = out_shape
        onto_output = np.array(
            [[out_w / width, 0, 0], [0, out_h / height, 0], [0, 0, 1]],
            dtype=np.float64,
        )
        matrix = (onto_output @ matrix).astype(np.float32)
        height, width = out_h, out_w
    return cv2.warpPerspective(
        _bgr(modified),
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _form_homography(
    source: np.ndarray,
    modified: np.ndarray,
    work_height: int = 900,
    features: int = 8000,
    ratio: float = 0.75,
    ransac: float = 4.0,
    min_inliers: int = 30,
) -> np.ndarray:
    source_small, source_scale = _working_image(source, work_height)
    modified_small, modified_scale = _working_image(modified, work_height)
    orb = cv2.ORB_create(features)
    source_keys, source_desc = orb.detectAndCompute(source_small, None)
    modified_keys, modified_desc = orb.detectAndCompute(modified_small, None)
    if source_desc is None or modified_desc is None:
        raise RuntimeError("could not find form features to align")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(source_desc, modified_desc, k=2)
    good = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio * second.distance:
            good.append(first)
    if len(good) < 8:
        raise RuntimeError("could not match the scan to the blank form")

    source_points = np.float32([source_keys[match.queryIdx].pt for match in good])
    modified_points = np.float32([modified_keys[match.trainIdx].pt for match in good])
    homography_matrix, mask = cv2.findHomography(modified_points, source_points, cv2.RANSAC, ransac)
    if homography_matrix is None or mask is None or int(mask.sum()) < min_inliers:
        raise RuntimeError("could not estimate an alignment from the form")

    to_modified = np.diag([modified_scale, modified_scale, 1.0])
    from_source = np.diag([1.0 / source_scale, 1.0 / source_scale, 1.0])
    return (from_source @ homography_matrix @ to_modified).astype(np.float32)


def _working_image(image: np.ndarray, height: int) -> tuple[np.ndarray, float]:
    scale = min(1.0, height / image.shape[0])
    if scale == 1.0:
        small = image
    else:
        small = cv2.resize(
            image,
            (int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return (np.clip(small, 0.0, 1.0) * 255).astype(np.uint8), scale

def settle_to_source(
    source,
    modified,
    step: int = 40,
    shift_limit: float = 4.0,
    min_response: float = 0.12,
) -> np.ndarray:
    source_gray = _gray(source)
    modified_gray = _gray(modified)
    source_gray, modified_gray = _overlap(source_gray, modified_gray)
    height, width = source_gray.shape
    color = _bgr(modified)[:height, :width]
    if step < 8 or height < step * 2 or width < step * 2:
        return color

    rows = list(range(step // 2, height, step))
    cols = list(range(step // 2, width, step))
    coarse = np.zeros((len(rows), len(cols), 2), np.float32)
    for row_index, y in enumerate(rows):
        for col_index, x in enumerate(cols):
            y0, y1 = max(0, y - step), min(height, y + step)
            x0, x1 = max(0, x - step), min(width, x + step)
            patch_source = source_gray[y0:y1, x0:x1].astype(np.float32) * np.float32(255.0)
            patch_modified = modified_gray[y0:y1, x0:x1].astype(np.float32) * np.float32(255.0)
            (shift_x, shift_y), response = cv2.phaseCorrelate(patch_source, patch_modified)
            if not np.isfinite(response) or response < min_response:
                continue
            if not np.isfinite(shift_x) or not np.isfinite(shift_y):
                continue
            coarse[row_index, col_index] = (
                float(np.clip(shift_x, -shift_limit, shift_limit)),
                float(np.clip(shift_y, -shift_limit, shift_limit)),
            )

    field = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_LINEAR)
    field = np.clip(field, -shift_limit, shift_limit)
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
    return cv2.remap(
        color,
        grid_x + field[..., 0],
        grid_y + field[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[2] == 4:
            array = array[:, :, :3]
        if array.dtype != np.uint8:
            array = _as_uint8(array)
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    elif array.ndim != 2:
        raise ValueError("expected a grayscale image or a BGR image")
    return _unit_float(array)


def _bgr(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        if array.dtype != np.uint8:
            array = _as_uint8(array)
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = _as_uint8(array)
    return np.ascontiguousarray(array)


def _as_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(_unit_float(image), 0.0, 1.0) * 255).astype(np.uint8)


def _unit_float(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.float32) / np.float32(255.0)
    array = array.astype(np.float32, copy=False)
    peak = float(np.max(array)) if array.size else 0.0
    if peak > 1.0:
        array = array / np.float32(255.0)
    return np.clip(array, 0.0, 1.0)


def _overlap(source: np.ndarray, modified: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height = min(source.shape[0], modified.shape[0])
    width = min(source.shape[1], modified.shape[1])
    if height < 3 or width < 3:
        raise ValueError("images must be at least 3 pixels on each side")
    return source[:height, :width], modified[:height, :width]