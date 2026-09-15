import tensorflow as tf
import numpy as np
from PIL import Image

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

def predict_checkbox(cropped_image, classNames = ["checked", "unchecked"]):
    model = tf.keras.models.load_model("../models/checkbox_CNN.keras")
    
    img = Image.fromarray(cropped_image).convert("L").resize((64, 64))
    imgArr = np.array(img).reshape(1, 64, 64, 1)
    prediction = model.predict(imgArr, verbose=0)[0][0]
    predictionLabel = classNames[1] if prediction > 0.5 else classNames[0]
    
    return predictionLabel, float(prediction)    

def classify_checkbox(source: dict, image, checkboxWords = []):
    checkboxResults = {}
    
    for checkbox in checkboxWords:
        crop = crop_checkbox(checkbox, image)
        
        label, confidence = predict_checkbox(crop)
        
        checkboxResults[checkbox["value"]] = {
            "label": label,
            "confidence": confidence,
        }
        
    return checkboxResults