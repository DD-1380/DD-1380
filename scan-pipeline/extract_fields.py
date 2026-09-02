import json
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from ocr_backends import ocr, workers

# Extract bounding boxes for all text fields from source.json.
def get_field_boxes(source: dict) -> dict:
    boxes = {}
    for block in source["blocks"]:
        for line in block["lines"]:
            for word in line["words"]:
                value = word["value"]
                if value.startswith("field_") and not value.startswith("field_image_"):
                    boxes[value] = word["geometry"]
    return boxes

# Crop a region from the aligned image.
def crop(image: np.ndarray, box: tuple) -> np.ndarray:
    h, w = image.shape[:2]
    (x0, y0), (x1, y1) = box
    x0 = max(0, int(x0 * w))
    y0 = max(0, int(y0 * h))
    x1 = min(w, int(x1 * w))
    y1 = min(h, int(y1 * h))
    return image[y0:y1, x0:x1]

# Takes aligned image and source JSON, returns extracted field dict.
# Output keys are the field_key names from source.json
def extract_fields(aligned_image: np.ndarray, source: dict) -> dict:
    boxes = get_field_boxes(source)
    field_keys = list(boxes.keys())
    crops = [crop(aligned_image, boxes[field_key]) for field_key in field_keys]

    with ThreadPoolExecutor(max_workers=workers()) as pool:
        texts = pool.map(ocr, crops, field_keys)

    output = {}
    for field_key, text in zip(field_keys, texts):
        output[field_key] = text
        print(f"{field_key}: '{text}'")
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