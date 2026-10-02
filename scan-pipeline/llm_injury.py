import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from llm_ocr import crop_to_data_url, get_client, get_model

PROMPT = (
    "You are inspecting one body-region crop from a scanned DD Form 1380. "
    "The crop shows a printed body outline. That printed outline is not an injury. "
    "A handwritten mark added on the region, usually an X, means it is injured. "
    "Return ONLY the word marked if a handwritten injury mark is visible, or unmarked "
    "if the crop shows only the printed outline. No extra words, quotes, markdown, or commentary."
)


def prompt_for(context: str | None = None) -> str:
    if not context:
        return PROMPT
    return (
        f"{PROMPT} This crop is form field '{context}'. Use the field name "
        "only to identify the body region. Never mark a region because of "
        "the field name, and never invent a mark that is not visible."
    )


def workers() -> int:
    return max(1, int(os.environ.get("OCR_LLM_WORKERS", "4")))


def _normalize(text: str) -> str:
    cleaned = text.strip().strip('"').strip("'")
    token = " ".join(cleaned.lower().split()).strip("`*.,:;")
    if token in {"unmarked", "not marked"}:
        return "unmarked"
    if token == "marked":
        return "marked"
    return cleaned


def _classify_data_url(data_url: str, context: str | None = None) -> str:
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
                        "image_url": {"url": data_url},
                    },
                ],
            }
        ],
        extra_body={
            "reasoning_effort": os.environ.get("OCR_LLM_REASONING_EFFORT", "none"),
        },
    )
    text = (response.choices[0].message.content or "").strip()
    return _normalize(text)


def _label_encoded(job: tuple[str, str | None]) -> str:
    context, data_url = job
    if data_url is None:
        return "unmarked"
    return _classify_data_url(data_url, context)


def classify_injury(image_crop: np.ndarray, context: str | None = None) -> str:
    if image_crop.size == 0:
        return "unmarked"
    return _classify_data_url(crop_to_data_url(image_crop), context)


def classify_injuries(crops: dict[str, np.ndarray]) -> dict[str, str]:
    get_client()
    jobs = []
    for field_key, image in crops.items():
        if image.size == 0:
            jobs.append((field_key, None))
        else:
            jobs.append((field_key, crop_to_data_url(image)))

    with ThreadPoolExecutor(max_workers=workers()) as pool:
        labels = pool.map(_label_encoded, jobs)

    output = {}
    for (field_key, _), label in zip(jobs, labels):
        output[field_key] = label
        print(f"{field_key}: '{label}'")
    return output


if __name__ == "__main__":
    import sys

    import cv2

    image = cv2.imread(sys.argv[1])
    context = sys.argv[2] if len(sys.argv) > 2 else None
    print(classify_injury(image, context))
