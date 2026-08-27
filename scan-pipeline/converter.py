import json
from pathlib import Path
from typing import Any


SCHEMA_PATH = Path(__file__).parent / "schema.json"

with open(SCHEMA_PATH, "r") as f:
    SCHEMA = json.load(f)


def default_value(field_type: str) -> Any:
    """
    Returns the default value for a field type.
    """

    if field_type == "bool":
        return False

    if field_type == "int":
        return 0

    if field_type == "float":
        return 0.0

    # Default for strings and unknown types
    return ""


def convert_to_badok(
    extracted_fields: dict[str, Any],
    document_id: int | None = None,
) -> dict[str, Any]:
    """
    Convert OCR extracted fields into the Badok dictionary.

    Parameters
    ----------
    extracted_fields:
        Dictionary returned by extract_fields().

    document_id:
        Optional document id.

    Returns
    -------
    dict
        Badok dictionary.
    """

    values = {}

   

    for field_name, metadata in SCHEMA.items():
        field_type = metadata.get("type", "str")
        values[field_name] = default_value(field_type)

    

    for field_name, value in extracted_fields.items():

        # Ignore fields that do not exist in the schema
        if field_name not in values:
            print(f"Warning: Unknown field '{field_name}'")
            continue

        values[field_name] = value

    

    badok = {
        "id": document_id,
        "values": values,
        "total_score": 0,
        "score_components": {}
    }

    return badok