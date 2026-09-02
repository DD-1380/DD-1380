import cv2

FIELD_COLOR = (0, 128, 0)       # green
FIELD_IMAGE_COLOR = (0, 0, 255) # red

# this file covers the overlay of text. This does not actually transform json or image.
# this strictly renders OCR text on top of the image.

def display_label(value: str) -> str:
    for prefix in ("field_image_", "field_"):
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def overlay(page, image):
    h, w = image.shape[:2]
    ref_h, ref_w = page["dimensions"]
    render_scale = w / ref_w if ref_w else 1.0
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

        MIN_SCALE = 0.25 * render_scale
        MAX_SCALE = 0.45 * render_scale
        CHAR_WIDTH_PX = 7 * render_scale
        MIN_TEXT_Y = 12 * render_scale
        TEXT_Y_OFFSET = 2 * render_scale
        scale = max(MIN_SCALE, min(MAX_SCALE, (px1 - px0) / max(len(label) * CHAR_WIDTH_PX, 1)))
        text_y = int(max(MIN_TEXT_Y, py0 - TEXT_Y_OFFSET))
        thickness = max(1, round(render_scale))

        cv2.rectangle(vis, (px0, py0), (px1, py1), color, thickness)
        cv2.putText(vis, label, (px0, text_y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    return vis