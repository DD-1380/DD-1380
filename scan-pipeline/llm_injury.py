import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from llm_ocr import crop_to_data_url, get_client, get_model

PROMPT = (
    "You are inspecting one body-region crop from a scanned DD Form 1380. "
    "The crop contains a printed body outline. A handwritten mark, usually an X, "
    "means that body region is injured. Return ONLY the word marked if a "
    "handwritten injury mark is visible, or unmarked if the crop shows only "
    "the printed outline. No extra words, quotes, markdown, or commentary."
)


def prompt_for(context: str | None = None) -> str:
    if not context:
        return PROMPT
    return (
        f"{PROMPT} This crop is form field '{context}'. Use the field name "
        "only as a hint for which body region this is; judge the mark from "
        "the image and never invent a mark that is not visible."
    )


def workers() -> int:
    return max(1, int(os.environ.get("OCR_LLM_WORKERS", "4")))


def _normalize(text: str) -> str:
    cleaned = text.strip().strip('"').strip("'").lower()
    first = cleaned.split(maxsplit=1)[0].strip(".,:;") if cleaned else ""
    if first in {"marked", "yes", "x", "true", "injured"}:
        return "marked"
    if first in {"unmarked", "no", "false", "none", "blank", "clear", ""}:
        return "unmarked"
    return cleaned


def classify_injury(image_crop: np.ndarray, context: str | None = None) -> str:
    if image_crop.size == 0:
        return "unmarked"

    response = get_client().chat.completions.create(
        model=get_model(),
        temperature=0,
        max_tokens=int(os.environ.get("OCR_LLM_MAX_TOKENS", "64")),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_for(context)},
                    {
                        "type": "image_url",
                        "image_url": {"url": crop_to_data_url(image_crop)},
                    },
                ],
            }
        ],
        extra_body={
            "reasoning_effort": os.environ.get("OCR_LLM_REASONING_EFFORT", "none"),
        },
    )
    text = response.choices[0].message.content or ""
    return _normalize(text)


def classify_injuries(crops: dict[str, np.ndarray]) -> dict[str, str]:
    field_keys = list(crops.keys())
    images = [crops[field_key] for field_key in field_keys]

    with ThreadPoolExecutor(max_workers=workers()) as pool:
        labels = pool.map(classify_injury, images, field_keys)

    output = {}
    for field_key, label in zip(field_keys, labels):
        output[field_key] = label
        print(f"{field_key}: '{label}'")
    return output


if __name__ == "__main__":
    import sys

    import cv2

    image = cv2.imread(sys.argv[1])
    context = sys.argv[2] if len(sys.argv) > 2 else None
    print(classify_injury(image, context))
