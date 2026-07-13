import json
import cv2
import numpy as np
from doctr.models import ocr_predictor, recognition_predictor

# Fields whose crops may contain multi-word or multi-line free text.
# These need detection first to properly locate/segment words before
# recognition. every other field is a single tight value and can skip
# detection entirely.
DETECTION_FIELDS = {"field_name", "field_unit", "field_other"}

recognition_model = None
detection_model = None

def get_recognition_model():
    global recognition_model
    if recognition_model is None:
        # recognition_predictor skips detection entirely. Every crop
        # gets a text prediction, since we already know it's a single
        # field/word crop and don't need docTR to relocate the text.
        recognition_model = recognition_predictor(pretrained=True)
    return recognition_model

def get_detection_model():
    global detection_model
    if detection_model is None:
        # Full detect + recognize pipeline, for crops that may contain
        # more than one word or wrap across a line.
        detection_model = ocr_predictor(pretrained=True)
    return detection_model

# Extract bounding boxes for all text fields from source.json.
def get_field_boxes(source: dict) -> dict:
    h, w = source["dimensions"]
    boxes = {}
    for block in source["blocks"]:
        for line in block["lines"]:
            for word in line["words"]:
                value = word["value"]
                if value.startswith("field_") and not value.startswith("field_image_"):
                    (x0, y0), (x1, y1) = word["geometry"]
                    boxes[value] = (
                        int(x0 * w), int(y0 * h),
                        int(x1 * w), int(y1 * h)
                    )
    return boxes

# Crop a region from the aligned image.
def crop(image: np.ndarray, box: tuple) -> np.ndarray:
    x0, y0, x1, y1 = box
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(image.shape[1], x1)
    y1 = min(image.shape[0], y1)
    return image[y0:y1, x0:x1]

# Run recognition-only on a batch of crops in one model call.
# Fast path: assumes each crop is a single word/value.
def ocr_batch_recognition(crops: list[np.ndarray]) -> list[str]:
    valid_idx = [i for i, c in enumerate(crops) if c.size > 0]
    if not valid_idx:
        return ["" for _ in crops]

    rgb_crops = [cv2.cvtColor(crops[i], cv2.COLOR_BGR2RGB) for i in valid_idx]
    predictions = get_recognition_model()(rgb_crops)  # List[Tuple[str, float]]

    texts = ["" for _ in crops]
    for i, (text, _confidence) in zip(valid_idx, predictions):
        texts[i] = text.strip()
    return texts

# Run full detect+recognize on a batch of crops in one model call.
# Slower path: needed for crops that may contain multiple words/lines.
def ocr_batch_detection(crops: list[np.ndarray]) -> list[str]:
    valid_idx = [i for i, c in enumerate(crops) if c.size > 0]
    if not valid_idx:
        return ["" for _ in crops]

    rgb_crops = [cv2.cvtColor(crops[i], cv2.COLOR_BGR2RGB) for i in valid_idx]
    result = get_detection_model()(rgb_crops)

    texts = ["" for _ in crops]
    for i, page in zip(valid_idx, result.pages):
        words = [
            word.value
            for block in page.blocks
            for line in block.lines
            for word in line.words
        ]
        texts[i] = " ".join(words).strip()
    return texts

# Takes aligned image and source JSON, returns extracted field dict.
def extract_fields(aligned_image: np.ndarray, source: dict) -> dict:
    boxes = get_field_boxes(source)
    field_keys = list(boxes.keys())

    recognition_keys = [k for k in field_keys if k not in DETECTION_FIELDS]
    detection_keys = [k for k in field_keys if k in DETECTION_FIELDS]

    output: dict[str, str] = {}

    if recognition_keys:
        crops = [crop(aligned_image, boxes[k]) for k in recognition_keys]
        texts = ocr_batch_recognition(crops)
        output.update(zip(recognition_keys, texts))

    if detection_keys:
        crops = [crop(aligned_image, boxes[k]) for k in detection_keys]
        texts = ocr_batch_detection(crops)
        output.update(zip(detection_keys, texts))

    output = {k: output[k] for k in field_keys}  # restore original order
    for k, v in output.items():
        print(f"{k}: '{v}'")
    return output

if __name__ == "__main__":
    import sys
    from pathlib import Path

    source_path = sys.argv[1]
    image_path = sys.argv[2]

    source = json.loads(Path(source_path).read_text())
    image = cv2.imread(image_path)

    result = extract_fields(image, source)
    print(json.dumps(result, indent=2))