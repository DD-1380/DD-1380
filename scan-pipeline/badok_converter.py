from typing import Dict, Any 
import json

with open ("field_map.json", "r") as f:
    FIELD_MAP = json.load(f)