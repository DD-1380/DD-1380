import numpy as np


def is_injury_field(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower().startswith("field_injury")


def crop_injury_field(image: np.ndarray, word: dict) -> np.ndarray:
    height, width = image.shape[:2]
    (x0, y0), (x1, y1) = word["geometry"]
    left = max(0, int(x0 * width))
    top = max(0, int(y0 * height))
    right = min(width, int(x1 * width))
    bottom = min(height, int(y1 * height))
    return image[top:bottom, left:right].copy()


def crop_injury_fields(image: np.ndarray, source: dict) -> dict[str, np.ndarray]:
    crops = {}
    for block in source["blocks"]:
        for line in block["lines"]:
            for word in line["words"]:
                if is_injury_field(word.get("value")):
                    crops[word["value"]] = crop_injury_field(image, word)
    return crops
