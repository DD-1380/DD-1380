import json
from typing import Any

# Load the schema
with open("schema.json", "r") as f:
    SCHEMA = json.load(f)


def convert_to_badok(
    extracted_fields: dict[str, Any],
    document_id: int | None = None,
) -> dict[str, Any]:
    """
    Convert extracted OCR fields into the Badok JSON format.
    """

    values = {}

    # Initialize every field with its default value
    for field_name, metadata in SCHEMA.items():
        values[field_name] = metadata["default"]

    # Replace defaults with extracted OCR values
    for field_name, value in extracted_fields.items():

        if field_name not in SCHEMA:
            print(f"Warning: Unknown field '{field_name}'")
            continue

        values[field_name] = value

    return {
        "id": document_id,
        "values": values,
        "total_score": 0,
        "score_components": {}
    }