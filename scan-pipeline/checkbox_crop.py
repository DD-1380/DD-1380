from pathlib import Path

import numpy as np
from PIL import Image

_checkbox_model = None
_tf = None
_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "checkbox_CNN.keras"
_CLASS_NAMES = ["checked", "unchecked"]

def _tensorflow():
    global _tf
    if _tf is None:
        import tensorflow as tf
        # The CNN is small. Leave the L4 to rembg and docTR. Importing
        # TensorFlow at startup loads its CUDA stub first and the next
        # CUDA session (the scan) segfaults.
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            pass
        _tf = tf
    return _tf

def get_checkbox_model():
    global _checkbox_model
    if _checkbox_model is None:
        _checkbox_model = _tensorflow().keras.models.load_model(_MODEL_PATH)
    return _checkbox_model

def _scores(batch):
    tf = _tensorflow()
    model = get_checkbox_model()
    x = tf.convert_to_tensor(batch, dtype=tf.float32)
    # The saved model nests RandomRotation / Zoom / Translation / Contrast
    # in an inner Sequential. predict() does not pass training=False into
    # that block, so those kernels run at inference and segfault.
    for layer in model.layers:
        if layer.__class__.__name__ in ("InputLayer", "Sequential"):
            continue
        x = layer(x, training=False)
    return np.asarray(x).reshape(-1)

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
    prediction = float(_scores(imgArr)[0])
    predictionLabel = classNames[1] if prediction > 0.5 else classNames[0]
    return predictionLabel, prediction

def classify_checkbox(source: dict, image, checkboxWords = None):
    if not checkboxWords:
        return {}

    usable = []
    inputs = []
    for checkbox in checkboxWords:
        cropped = crop_checkbox(checkbox, image)
        if cropped.size == 0:
            continue
        usable.append(checkbox)
        inputs.append(_crop_to_input(cropped))
    if not usable:
        return {}

    scores = _scores(np.stack(inputs))
    checkboxResults = {}
    for checkbox, score in zip(usable, scores):
        score = float(score)
        checkboxResults[checkbox["value"]] = {
            "label": _CLASS_NAMES[1] if score > 0.5 else _CLASS_NAMES[0],
            "confidence": score,
        }
    return checkboxResults