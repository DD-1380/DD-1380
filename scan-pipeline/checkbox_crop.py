from pathlib import Path

import tensorflow as tf
import numpy as np
from PIL import Image

_checkbox_model = None
_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "checkbox_CNN.keras"
_CLASS_NAMES = ["checked", "unchecked"]

def get_checkbox_model():
    global _checkbox_model
    if _checkbox_model is None:
        _checkbox_model = tf.keras.models.load_model(_MODEL_PATH)
    return _checkbox_model

def is_checkbox_field(word: str | None) -> bool:
    if not word:
        return False
    
    normalizedWord = word.strip().lower()
    if normalizedWord.startswith("field_image_"):
        return True
    
    return False

def crop_checkbox(word: dict, image, padding = 4):
    imgHeight, imgWidth = image.shape[:2]
    (x0, y0), (x1, y1) = word["geometry"]
    
    leftDim = max(0, int(round(x0 * imgWidth)) - padding)
    topDim = max(0, int(round(y0 * imgHeight)) - padding)
    rightDim = min(imgWidth, int(round(x1 * imgWidth)) + padding)
    bottomDim = min(imgHeight, int(round(y1 * imgHeight)) + padding)

    return image[topDim:bottomDim, leftDim:rightDim].copy()

def _crop_to_input(cropped_image):
    img = Image.fromarray(cropped_image).convert("L").resize((64, 64))
    return np.array(img).reshape(64, 64, 1)

def predict_checkbox(cropped_image, classNames = None):
    if classNames is None:
        classNames = _CLASS_NAMES
    imgArr = _crop_to_input(cropped_image).reshape(1, 64, 64, 1)
    prediction = float(get_checkbox_model().predict(imgArr, verbose=0)[0][0])
    predictionLabel = classNames[1] if prediction > 0.5 else classNames[0]
    return predictionLabel, prediction

def classify_checkbox(source: dict, image, checkboxWords = None):
    if not checkboxWords:
        return {}

    batch = np.stack([
        _crop_to_input(crop_checkbox(checkbox, image))
        for checkbox in checkboxWords
    ])
    predictions = get_checkbox_model().predict(batch, verbose=0)

    checkboxResults = {}
    for checkbox, prediction in zip(checkboxWords, predictions):
        score = float(prediction[0])
        checkboxResults[checkbox["value"]] = {
            "label": _CLASS_NAMES[1] if score > 0.5 else _CLASS_NAMES[0],
            "confidence": score,
        }
    return checkboxResults