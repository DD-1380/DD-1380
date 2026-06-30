import json
import cv2
import numpy as np
from pathlib import Path
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# mapping from source.json field names to output JSON keys
FIELD_MAP = {
    "field_battle_roster": "BATTLE_ROSTER_NUMBER_PG1",
    "field_name": "NAME",
    "field_last_4": "SSN_LAST4",
    "field_service": "SERVICE",
    "field_unit": "UNIT",
    "field_allergies": "ALLERGIES",
    "field_other": "MECHANISM_OTHER",
    "field_r_arm_type": "TQ_RIGHT_ARM_TYPE",
    "field_l_arm_type": "TQ_LEFT_ARM_TYPE",
    "field_r_arm_time": "TQ_RIGHT_ARM_TIME",
    "field_l_arm_time": "TQ_LEFT_ARM_TIME",
    "field_r_leg_type": "TQ_RIGHT_LEG_TYPE",
    "field_l_leg_type": "TQ_LEFT_LEG_TYPE",
    "field_r_leg_time": "TQ_RIGHT_LEG_TIME",
    "field_l_leg_time": "TQ_LEFT_LEG_TIME",
    "field_time_1": "SS_TIME_1",
    "field_time_2": "SS_TIME_2",
    "field_time_3": "SS_TIME_3",
    "field_time_4": "SS_TIME_4",
    "field_pulse_rate_1": "SS_PULSE_1",
    "field_pulse_rate_2": "SS_PULSE_2",
    "field_pulse_rate_3": "SS_PULSE_3",
    "field_pulse_rate_4": "SS_PULSE_4",
    "field_blood_1": "SS_BP_1",
    "field_blood_2": "SS_BP_2",
    "field_blood_3": "SS_BP_3",
    "field_blood_4": "SS_BP_4",
    "field_respiratory_1": "SS_RESP_1",
    "field_respiratory_2": "SS_RESP_2",
    "field_respiratory_3": "SS_RESP_3",
    "field_respiratory_4": "SS_RESP_4",
    "field_pulse_ox_1": "SS_PULSEOX_1",
    "field_pulse_ox_2": "SS_PULSEOX_2",
    "field_pulse_ox_3": "SS_PULSEOX_3",
    "field_pulse_ox_4": "SS_PULSEOX_4",
    "field_avpu_1": "SS_AVPU_1",
    "field_avpu_2": "SS_AVPU_2",
    "field_avpu_3": "SS_AVPU_3",
    "field_avpu_4": "SS_AVPU_4",
    "field_pain_1": "SS_PAIN_1",
    "field_pain_2": "SS_PAIN_2",
    "field_pain_3": "SS_PAIN_3",
    "field_pain_4": "SS_PAIN_4",
}

model = None

def get_model():
    global model
    if model is None:
        model = ocr_predictor(pretrained=True)
    return model

#Extract bounding boxes for all text fields from source.json.
def get_field_boxes(source: dict) -> dict:

    manual_map_path = Path("field_box_map.json")
    if manual_map_path.exists():
        raw = json.loads(manual_map_path.read_text())
        return {k: tuple(v) for k, v in raw.items()}
    

    h, w = source["dimensions"]
    boxes = {}
    for block in source["blocks"]:
        for line in block["lines"]:
            for word in line["words"]:
                value = word["value"]
                if value.startswith("field_") and not value.startswith("field_image_"):
                    (x0, y0), (x1, y1) = word["geometry"]
                    X_SHIFT = 0
                    Y_SHIFT = 0
                    Y_PAD = 4
                    boxes[value] = (
                        int(x0 * w) + X_SHIFT, int(y0 * h) + Y_SHIFT - Y_PAD,
                        int(x1 * w) + X_SHIFT + 100, int(y1 * h) + Y_SHIFT + Y_PAD
                    )
    return boxes

#Crop a region from the aligned image and OCR it.
def crop_and_ocr(image: np.ndarray, box: tuple) -> str:
    x0, y0, x1, y1 = box
    pad = 4
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(image.shape[1], x1 + pad)
    y1 = min(image.shape[0], y1 + pad)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return ""
    _, buf = cv2.imencode(".jpg", crop)
    doc = DocumentFile.from_images(buf.tobytes())
    result = get_model()(doc)
    words = [
        word.value
        for page in result.pages
        for block in page.blocks
        for line in block.lines
        for word in line.words
    ]
    return " ".join(words).strip()

#Takes aligned image and source JSON, returns extracted field dict.
def extract_fields(aligned_image: np.ndarray, source: dict) -> dict:
    boxes = get_field_boxes(source)

    debug = aligned_image.copy()
    for field_key, (x0, y0, x1, y1) in boxes.items():
        cv2.rectangle(debug, (x0, y0), (x1, y1), (0, 255, 0), 1)
    cv2.imwrite("debug_boxes.jpg", debug)

    output = {}
    for field_key, box in boxes.items():
        output_key = FIELD_MAP.get(field_key, field_key)
        text = crop_and_ocr(aligned_image, box)
        output[output_key] = text if text else ""
        print(f"{output_key}: '{text}'")
    return output

if __name__ == "__main__":
    import sys
    source_path = sys.argv[1]
    image_path = sys.argv[2]

    source = json.loads(Path(source_path).read_text())
    image = cv2.imread(image_path)
    result = extract_fields(image, source)
    print(json.dumps(result, indent=2))