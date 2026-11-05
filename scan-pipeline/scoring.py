import re

# Scores a DD-1380 against the USUHS 30-item and 11-item checklists, using
# the pipeline's own field names (source.json). Takes the two dicts that
# process_document returns: OCR text per text field, and the checkbox CNN
# result per checkbox.
#
# A field missing from both dicts hasn't been extracted yet (no box in
# source.json, e.g. the back of the card), so items that depend on it are
# listed in "not_scored" instead of scoring 0. Fields below that aren't in
# source.json yet are named in the same style for when they're annotated.
#
# Rules follow the USUHS scoring document, except where all 15 graded cards
# in the USU/UCF spreadsheet consistently do otherwise:
#   - times like "1200L" count (the document says alphanumeric time = 0)
#   - a blank battle roster # / evac on the back scores 0 (the document says 1)
#   - vitals are scored individually (items 13-19), as on the revised checklist

TIME_PATTERN = re.compile(r"\d{3,4}")

EVAC = ["field_image_evac", "field_image_priority", "field_image_routine"]
EVAC_BACK = ["field_image_evac_back", "field_image_priority_back", "field_image_routine_back"]
GENDER = ["field_image_male", "field_image_female"]                       # not in source.json yet
MECHANISMS = [f"field_image_{m}" for m in
              ("artillery", "blunt", "burn", "fall", "grenade", "gsw", "ied", "landmine", "mvc", "rpg")]
TOURNIQUETS = [(f"field_{limb}_type", f"field_{limb}_time") for limb in ("r_arm", "l_arm", "r_leg", "l_leg")]
VITALS = {
    "ss_time": [f"field_time_{n}" for n in range(1, 5)],
    "ss_pulse": [f"field_pulse_rate_{n}" for n in range(1, 5)],
    "ss_bp": [f"field_blood_{n}" for n in range(1, 5)],
    "ss_resp": [f"field_respiratory_{n}" for n in range(1, 5)],
    "ss_pulse_ox": [f"field_pulse_ox_{n}" for n in range(1, 5)],
    "ss_avpu": [f"field_avpu_{n}" for n in range(1, 5)],
    "ss_pain": [f"field_pain_{n}" for n in range(1, 5)],
}
# Back of the card, not in source.json yet.
CIRCULATION = ["field_image_tq_extremity", "field_image_tq_junctional", "field_image_tq_truncal", "field_tq_type",
               "field_image_dressing_hemostatic", "field_image_dressing_pressure", "field_image_dressing_other",
               "field_dressing_type"]
AIRWAY = ["field_image_airway_intact", "field_image_airway_npa", "field_image_airway_cric",
          "field_image_airway_et_tube", "field_image_airway_sga", "field_airway_type"]
BREATHING = ["field_image_breathing_o2", "field_image_breathing_needle_d", "field_image_breathing_chest_seal",
             "field_image_breathing_chest_tube", "field_breathing_type"]
FLUID_ROWS = [[f"field_{kind}_{col}_{n}" for col in ("name", "volume", "route", "time")]
              for kind in ("fluid", "bloodproduct") for n in (1, 2)]
MED_ROWS = [[f"field_{kind}_{col}_{n}" for col in ("name", "dose", "route", "time")]
            for kind, rows in (("analgesic", 3), ("antibiotic", 2), ("othermed", 2)) for n in range(1, rows + 1)]
OTHER = ["field_image_combat_pill_pack", "field_image_eye_shield", "field_image_eye_shield_right",
         "field_image_eye_shield_left", "field_image_splint", "field_image_hypothermia", "field_hypothermia_type"]


# True/False if the field was extracted, None if it wasn't.
def filled(values: dict, key: str):
    value = values.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip() != ""


# Three-valued AND/OR: unknown only when the known fields can't decide it.
def all_of(results):
    results = list(results)
    if False in results:
        return False
    return None if None in results else True


def any_of(results):
    results = list(results)
    if True in results:
        return True
    return None if None in results else False


def any_filled(values: dict, keys) -> bool | None:
    return any_of(filled(values, k) for k in keys)


def is_time(values: dict, key: str):
    value = values.get(key)
    if value is None:
        return None
    return TIME_PATTERN.search(str(value)) is not None


def row_complete(values: dict, row) -> bool | None:
    return all_of(filled(values, k) for k in row)


def _not(result):
    return None if result is None else not result


# Item 11: a mechanism box, or "Other" text. "Other" checked with no text is 0.
def mechanism(values: dict):
    other_box, other_text = filled(values, "field_image_other"), filled(values, "field_other")
    if other_box and other_text is False:
        return False
    return any_of([any_filled(values, MECHANISMS), other_text])


# Any mark on the injury diagram (field_injury_*, from the body-diagram
# reader) or the TQ section of the front.
def front_marked(values: dict):
    injury_keys = [k for k in values if k.startswith("field_injury_")]
    tq_keys = [k for pair in TOURNIQUETS for k in pair]
    marked = any_filled(values, injury_keys + tq_keys)
    # No field_injury_* keys at all means the diagram hasn't been read yet.
    if marked is False and not injury_keys:
        return None
    return marked


# Item 12: the front is marked, and every TQ type has a time.
def injuries(values: dict):
    for tq_type, tq_time in TOURNIQUETS:
        if filled(values, tq_type) and filled(values, tq_time) is False:
            return False
    return front_marked(values)


# Circulation, breathing and other: blank is fine when the front shows no injury.
def treatment_or_uninjured(values: dict, keys):
    return any_of([any_filled(values, keys), _not(front_marked(values))])


def score_30(values: dict) -> dict:
    items = {
        "battle_roster_num": filled(values, "field_battle_roster"),
        "evac_category": any_filled(values, EVAC),
        "name": filled(values, "field_name"),
        "last_4": filled(values, "field_last_4"),
        "gender": any_filled(values, GENDER),
        "date": filled(values, "field_date"),                    # not in source.json yet
        "time": is_time(values, "field_time"),                   # not in source.json yet
        "service": filled(values, "field_service"),
        "unit": filled(values, "field_unit"),
        "allergies": filled(values, "field_allergies"),
        "mechanism_of_injury": mechanism(values),
        "injuries": injuries(values),
    }
    for name, keys in VITALS.items():
        check = is_time if name == "ss_time" else filled
        items[name] = any_of(check(values, k) for k in keys)
    items.update({
        "battle_roster_num_back": filled(values, "field_battle_roster_back"),
        "evac_category_back": any_filled(values, EVAC_BACK),
        "circulation": treatment_or_uninjured(values, CIRCULATION),
        "airway_treatment": any_filled(values, AIRWAY),
        "breathing": treatment_or_uninjured(values, BREATHING),
        "circulation_fluid": any_of(row_complete(values, row) for row in FLUID_ROWS),
        "meds": any_of(row_complete(values, row) for row in MED_ROWS),
        "other": treatment_or_uninjured(values, OTHER),
        "notes": filled(values, "field_notes"),
        "first_responder_name": filled(values, "field_first_responder_name"),
        "first_responder_last_4": filled(values, "field_first_responder_last_4"),
    })
    return items


def score_11(values: dict, items_30: dict) -> dict:
    patient_info = ["name", "last_4", "gender", "date", "time", "service", "unit", "allergies"]
    vitals_without_time = [k for k in VITALS if k != "ss_time"]
    return {
        "battle_roster_num": any_of([items_30["battle_roster_num"], items_30["battle_roster_num_back"]]),
        "evac_category": any_of([items_30["evac_category"], items_30["evac_category_back"]]),
        "patient_info": all_of(items_30[k] for k in patient_info),
        "mechanism_of_injury": items_30["mechanism_of_injury"],
        "injuries": items_30["injuries"],
        "signs_and_symptoms": any_of(items_30[k] for k in vitals_without_time),
        "treatment_info": any_of([any_filled(values, CIRCULATION + AIRWAY + BREATHING),
                                  items_30["circulation_fluid"]]),
        "meds": items_30["meds"],
        "other": items_30["other"],
        "notes": items_30["notes"],
        "first_responder_identity": all_of([items_30["first_responder_name"], items_30["first_responder_last_4"]]),
    }


# Items 3-9 of the 11-item checklist are the critical ones.
CRITICAL_11 = ["patient_info", "mechanism_of_injury", "injuries", "signs_and_symptoms",
               "treatment_info", "meds", "other"]


def as_str(result):
    return None if result is None else str(int(result))


def total(items: dict) -> int:
    return sum(1 for v in items.values() if v)


# Builds the report in the output format from the scoring document. Scores are
# strings, as in that document; null means the field hasn't been extracted.
def score_card(fields: dict, checkboxResults: dict) -> dict:
    values = dict(fields)
    values.update({k: r["label"] == "checked" for k, r in checkboxResults.items()})

    items_30 = score_30(values)
    items_11 = score_11(values, items_30)
    return {
        "total_score": str(total(items_30)),
        "total_grade": str(round(100 * total(items_30) / len(items_30))),
        "score_components": {k: as_str(v) for k, v in items_30.items()},
        "checklist_11_score": str(total(items_11)),
        "critical_score": str(total({k: items_11[k] for k in CRITICAL_11})),
        "checklist_11_components": {k: as_str(v) for k, v in items_11.items()},
        "not_scored": [k for k, v in items_30.items() if v is None],
        "grade_components": {k: as_str(filled(values, k)) for k in values},
    }
