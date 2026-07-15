"""
Field type classification for DD-1380 OCR extraction.

Maps a field_key (as annotated in source.json) to an expected data type,
based on the "Notes" column of the USUHS DD-1380 scoring document.
Mapping is keyed against the real field_key naming scheme observed in
source.json (field_battle_roster, field_pulse_rate_1, field_blood_1_sys,
etc.)

Valid types: "alphanumeric", "alphabetic", "numeric",
"numeric_or_special", "date", "time".
"""

# Exact field_key -> type
_EXACT_TYPES = {
    "field_battle_roster": "alphanumeric",
    "field_name": "alphabetic",
    "field_last_4": "numeric_or_special",
    "field_service": "alphanumeric",
    "field_unit": "alphanumeric",
    "field_allergies": "alphabetic",
    "field_other_1": "alphanumeric",
    "field_other_2": "alphanumeric",
}

# field_key prefix -> type, checked in order (first match wins)
_PREFIX_TYPES = [
    ("field_time_", "time"),
    ("field_pulse_rate_", "numeric"),
    ("field_respiratory_", "numeric"),
    ("field_pulse_ox_", "numeric_or_special"),  # doc: "Numeric or 'N/A'"
    ("field_avpu_", "alphanumeric"),
    ("field_pain_", "alphanumeric"),
    ("field_l_arm_time", "time"),
    ("field_r_arm_time", "time"),
    ("field_l_leg_time", "time"),
    ("field_r_leg_time", "time"),
    ("field_l_arm_type", "alphanumeric"),
    ("field_r_arm_type", "alphanumeric"),
    ("field_l_leg_type", "alphanumeric"),
    ("field_r_leg_type", "alphanumeric"),
]


def classify_field(field_key: str) -> str:
    """Classify a field_key into an expected data type.

    Falls back to "alphanumeric" (no restriction) for anything not
    explicitly listed here, including field_image_* checkbox fields,
    which this module doesn't attempt to type since they aren't OCR'd
    as text.
    """
    key = field_key.lower()

    if key in _EXACT_TYPES:
        return _EXACT_TYPES[key]

    if key.endswith("_sys") or key.endswith("_dia"):
        # Blood pressure, split into systolic/diastolic — each half is
        # just a number, even though the combined "120/80" field is
        # Alphanumeric in the scoring doc (because of the "/").
        return "numeric"

    for prefix, field_type in _PREFIX_TYPES:
        if key.startswith(prefix):
            return field_type

    return "alphanumeric"
