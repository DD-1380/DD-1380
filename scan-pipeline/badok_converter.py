from typing import Dict, Any
import json

with open("field_map.json", "r") as f:
    FIELD_SCHEMA = json.load(f)


def convert_to_badok(
    extracted_fields: Dict[str, Any],
    document_id: int | None = None,
) -> Dict[str, Any]:

    values = {}

    # Init every Badok field with its default value
    for badok_name, metadata in FIELD_SCHEMA.items():
        values[badok_name] = metadata["default"]

    # defaults with OCR results
    for field_name, value in extracted_fields.items():
        if field_name in values:
            values[field_name] = value

    return {
        "id": document_id,
        "values": values,
        "total_score": 0,
        "score_components": {}
    } #noah, this assumes tthat the ocr engine already returns keys like: name date or any info text it got