from typing import Dict, Any 
import json

with open ("field_map.json", "r") as f:
    FIELD_MAP = json.load(f)

def convert_to_badok(
    extracted_fields: Dict[str,Any],
    document_id: int | None = None;
) -> Dict[str,Any]:
    values = {}

    for internal_name, value in extracted_fields.items():

        badok_name = FIELD_MAP.get(internal_name)

        if badok_name is None:
            continue

        values[badok_name] = value

    return{
    }

