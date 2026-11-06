import base64
import os

import cv2
import numpy as np
from openai import OpenAI

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "gemma4:e4b"

PROMPT = (
    "You are an OCR engine. Read the text in this image crop of a scanned "
    "form field and return ONLY the exact text you see, with no extra "
    "words, quotes, markdown, or commentary. If the crop is blank or has "
    "no legible text, return an empty string."
)


def prompt_for(context: str | None = None) -> str:
    if not context:
        return PROMPT
    return (
        f"{PROMPT} This crop is form field '{context}'. Use the field name "
        "only as a hint for expected format; never invent or complete text "
        "that is not visible in the image. Text should be in english or numeric with no unicode characters."
    )


client = None


def get_client() -> OpenAI:
    global client
    if client is None:
        client = OpenAI(
            base_url=os.environ.get("OCR_LLM_BASE_URL", DEFAULT_BASE_URL),
            api_key=os.environ.get("OCR_LLM_API_KEY", "ollama"),
        )
    return client


def get_model() -> str:
    return os.environ.get("OCR_LLM_MODEL", DEFAULT_MODEL)


def crop_to_data_url(image_crop: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", image_crop)
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


HUMAN_INPUT = "human input requested"


def choose_reading(field_key: str, first: str, second: str) -> str:
    prompt = (
        "You are checking two OCR readings of one form field. "
        f"The field name is '{field_key}'. "
        "Decide whether the readings are similar, and which one looks more right for that field name. "
        "An empty reading means that reader saw no text. A blank value can be right when the field is empty. "
        "If one reading looks more right, or the readings are similar, reply with the better one using exactly one of these lines:\n"
        "CHOICE: A\n"
        "CHOICE: B\n"
        "A is the first reading and B is the second. "
        "If both readings look odd for this field name, reply with exactly:\n"
        "CHOICE: human input requested\n"
        "Return nothing else.\n"
        f"Reading A: {first}\n"
        f"Reading B: {second}"
    )
    response = get_client().chat.completions.create(
        model=get_model(),
        temperature=0,
        max_tokens=int(os.environ.get("OCR_LLM_MAX_TOKENS", "64")),
        messages=[{"role": "user", "content": prompt}],
        extra_body={
            "reasoning_effort": os.environ.get("OCR_LLM_REASONING_EFFORT", "none"),
        },
    )
    text = (response.choices[0].message.content or "").strip()
    folded = text.strip()
    marker = folded.lower().rfind("choice:")
    value = folded[marker + len("choice:"):].strip() if marker != -1 else folded
    value = value.splitlines()[0].strip().strip('"').strip("'")
    if value.lower() == HUMAN_INPUT:
        return HUMAN_INPUT
    if value.upper() == "A":
        return first
    if value.upper() == "B":
        return second
    if value == first or value == second:
        return value
    return HUMAN_INPUT


def ocr(image_crop: np.ndarray, context: str | None = None) -> str:
    if image_crop.size == 0:
        return ""

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
    text = (response.choices[0].message.content or "").strip()
    return text.strip('"').strip()


if __name__ == "__main__":
    import sys

    image = cv2.imread(sys.argv[1])
    context = sys.argv[2] if len(sys.argv) > 2 else None
    print(ocr(image, context))
