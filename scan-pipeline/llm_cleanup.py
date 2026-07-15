"""
Use an LLM (via the OpenAI API) to clean up raw OCR text per field, using
the expected data type from field_types.classify_field as context for
the correction.

Requires the OPENAI_API_KEY environment variable to be set.
"""

import json
import os

from openai import OpenAI

from field_types import classify_field

MODEL = "gpt-4o-mini" 

_TYPE_DESCRIPTIONS = {
    "alphanumeric": "letters and/or digits, no restriction",
    "alphabetic": "letters only (a name or word, no digits)",
    "numeric": "digits only",
    "numeric_or_special": "digits and/or special characters like - or /, no letters",
    "date": "a date, digits only (e.g. DDMMYY or similar numeric date format)",
    "time": "a time, digits only (e.g. HHMM 24-hour format)",
}


def _build_prompt(fields: dict[str, str]) -> str:
    lines = []
    for field_key, raw_text in fields.items():
        field_type = classify_field(field_key)
        description = _TYPE_DESCRIPTIONS.get(field_type, "no restriction")
        lines.append(
            f'- "{field_key}" (expected: {description}): raw OCR text = "{raw_text}"'
        )
    joined = "\n".join(lines)
    return (
        "The following are raw OCR results from a scanned military TCCC "
        "(DD-1380) card, one per form field. OCR sometimes misreads "
        "characters (e.g. O/0, l/1/I/), S/5, */A) or picks up stray noise.\n\n"
        "For each field, using the expected type as a guide, return your "
        "best-corrected value. If the raw text is empty or clearly "
        "unreadable, return an empty string rather than guessing content "
        "that isn't supported by the raw text. Do not invent values that "
        "aren't grounded in the raw OCR text.\n\n"
        f"{joined}"
    )


def _build_schema(field_keys: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {key: {"type": "string"} for key in field_keys},
        "required": field_keys,
        "additionalProperties": False,
    }


def clean_fields_with_llm(fields: dict[str, str], api_key: str | None = None) -> dict[str, str]:
    """Takes {field_key: raw_ocr_text}, returns {field_key: cleaned_text}.

    Sends every field in a single request (one batched call, not one per
    field) using structured outputs so the response shape is guaranteed
    to match the input keys.

    Reads the API key from the OPENAI_API_KEY environment variable unless
    api_key is passed explicitly.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "No OpenAI API key found. Set the OPENAI_API_KEY environment "
            "variable or pass api_key explicitly."
        )
    client = OpenAI(api_key=key)

    field_keys = list(fields.keys())
    prompt = _build_prompt(fields)
    schema = _build_schema(field_keys)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You clean up noisy OCR output for a structured form. "
                    "Return only what the raw text supports."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "cleaned_fields",
                "schema": schema,
                "strict": True,
            },
        },
    )

    return json.loads(response.choices[0].message.content)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    raw_fields = json.loads(Path(sys.argv[1]).read_text())
    cleaned = clean_fields_with_llm(raw_fields)
    print(json.dumps(cleaned, indent=2))
