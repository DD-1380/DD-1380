import json
import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = None
model = None

def get_model():
    global processor, model
    if model is None:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
    return processor, model

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

# Run TrOCR recognition on a batch of crops in one model call.
def ocr_batch_recognition(crops: list[np.ndarray]) -> list[str]:
    valid_idx = [i for i, c in enumerate(crops) if c.size > 0]
    if not valid_idx:
        return ["" for _ in crops]

    proc, mdl = get_model()
    pil_crops = [
        Image.fromarray(cv2.cvtColor(crops[i], cv2.COLOR_BGR2RGB))
        for i in valid_idx
    ]

    pixel_values = proc(images=pil_crops, return_tensors="pt").pixel_values
    generated_ids = mdl.generate(pixel_values)
    texts_raw = proc.batch_decode(generated_ids, skip_special_tokens=True)

    texts = ["" for _ in crops]
    for i, text in zip(valid_idx, texts_raw):
        texts[i] = text.strip()
    return texts

# Takes aligned image and source JSON, returns extracted field dict.
def extract_fields(aligned_image: np.ndarray, source: dict) -> dict:
    boxes = get_field_boxes(source)
    field_keys = list(boxes.keys())

    crops = [crop(aligned_image, boxes[k]) for k in field_keys]
    texts = ocr_batch_recognition(crops)
    output = dict(zip(field_keys, texts))

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
