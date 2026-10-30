import cv2
import numpy as np

FIELD_COLOR = (0, 128, 0)       # green
FIELD_IMAGE_COLOR = (0, 0, 255) # red

# this file covers the overlay of text. This does not actually transform json or image.
# this strictly renders OCR text on top of the image.

def display_label(value: str) -> str:
    for prefix in ("field_image_", "field_"):
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def _body_view(page, view_name):
    for key in (f"body_diagram_{view_name}", view_name):
        if isinstance(page.get(key), dict):
            return page[key]
    return None


def _label_position(polygon, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(distance.argmax()), distance.shape)
    return x, y


def overlay(page, image, checkboxResults, bodyResults=None):
    h, w = page["dimensions"]
    vis = image.copy()

    words = (
        word
        for block in page["blocks"]
        for line in block["lines"]
        for word in line["words"]
        if (word.get("value") or "").startswith("field")
    )

    for word in words:
        value = word["value"]
        (x0, y0), (x1, y1) = word["geometry"]
        px0, py0 = int(x0 * w), int(y0 * h)
        px1, py1 = int(x1 * w), int(y1 * h)

        color = FIELD_IMAGE_COLOR if value.startswith("field_image") else FIELD_COLOR
        label = display_label(value)
        label += f" ({checkboxResults[value]["label"]})" if value in checkboxResults else ""

        MIN_SCALE = 0.25
        MAX_SCALE = 0.45
        CHAR_WIDTH_PX = 7
        MIN_TEXT_Y = 12
        TEXT_Y_OFFSET = 2
        scale = max(MIN_SCALE, min(MAX_SCALE, (px1 - px0) / max(len(label) * CHAR_WIDTH_PX, 1)))
        text_y = max(MIN_TEXT_Y, py0 - TEXT_Y_OFFSET)

        cv2.rectangle(vis, (px0, py0), (px1, py1), color, 1)
        cv2.putText(vis, label, (px0, text_y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    for view_name, view_results in (bodyResults or {}).items():
        view = _body_view(page, view_name)
        if not view:
            continue
        bbox = view.get("bbox")
        if not bbox:
            continue
        x1, y1 = int(round(bbox[0])), int(round(bbox[1]))
        for region in view.get("regions", []):
            name = region["name"]
            polygon = np.asarray(region["polygon"], dtype=np.int32)
            if view.get("coordinate_frame", "full_page_aligned") != "crop_local":
                polygon += np.asarray([x1, y1], dtype=np.int32)
            result = view_results.get(name, {"marked": False})
            marked = result["marked"]
            color = (0, 0, 255) if marked else FIELD_COLOR
            fill = vis.copy()
            if marked:
                cv2.fillPoly(fill, [polygon], color)
                vis = cv2.addWeighted(fill, 0.35, vis, 0.65, 0)
            cv2.polylines(vis, [polygon], True, color, 2 if marked else 1)
            local_polygon = polygon - np.asarray([x1, y1], dtype=np.int32)
            px, py = _label_position(local_polygon, (
                max(1, int(round(bbox[3] - bbox[1]))),
                max(1, int(round(bbox[2] - bbox[0]))),
            ))
            label = "marked" if marked else "unmarked"
            anchor = (x1 + px, y1 + py)
            cv2.putText(vis, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(vis, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    return vis