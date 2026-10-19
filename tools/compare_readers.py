"""
Compares two readers' eval results and tests review policies that use the
second reader to check the first.

    python tools/compare_readers.py results_trocr.csv results_llm.csv

Takes two CSVs from eval_cards.py and matches them by card and field, so only
cards in both count. The first file is the primary reader, the second is the
checker.

Policies (a flagged field goes to a human reviewer):
  all differences  flag any field where the readings differ
  transcription    flag differences that survive ignoring punctuation, or
                   where the primary reading doesn't fit the field's format
  grading          flag only differences that could change the score: one
                   blank and one filled, or a time field where only one
                   passes the time check

Reports, per policy: flags per card, share of the primary reader's errors
caught, and accuracy if every flagged field is fixed. Writes all flagged
fields to disagreements.csv next to the first file.
"""
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scan-pipeline"))

from scoring import TIME_PATTERN  # noqa: E402

# Usual format per field, as in check_truth.py.
TIME = r"\d{3,4} ?[LZ]?"
FORMATS = [
    (re.compile(r"field_time_\d+|field_.*_time"), re.compile(TIME), "time: 3-4 digits, optional L/Z"),
    (re.compile(r"field(_first_responder)?_last_4"), re.compile(r"\d{4}"), "last 4: 4 digits"),
    (re.compile(r"field_avpu_\d+"),
     re.compile(r"[AVPU]( ?[+-])?|ALERT|VERBAL|VOICE|PAIN(FUL)?|UNRESP\w*|UNRESPONSIVE"), "AVPU: A/V/P/U or the word"),
    (re.compile(r"field_pain_\d+"), re.compile(r"\d{1,2}( ?/ ?10)?"), "pain: 0-10 or x/10"),
    (re.compile(r"field_battle_roster(_back)?"), re.compile(r"[A-Z]{2} ?\d{3,4}"), "battle roster: 2 letters + digits"),
]
TIME_FIELD = re.compile(r"field_time(_\d+)?")


def strict(text) -> str:
    return " ".join(str(text or "").split()).upper()


def loose(text) -> str:
    return re.sub(r"[^A-Z0-9]", "", strict(text))


# What the score sees in a field: filled or not, and for time fields whether it passes the time check.
def score_view(field: str, text) -> tuple:
    text = strict(text)
    return bool(text), (TIME_PATTERN.search(text) is not None) if TIME_FIELD.fullmatch(field) else None


def bad_format(field: str, text) -> bool:
    text = strict(text)
    rule = next((pattern for name, pattern, _ in FORMATS if name.fullmatch(field)), None)
    return bool(text) and rule is not None and not rule.fullmatch(text)


def load(path: str) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        return {(row["card"], row["key"]): row for row in csv.DictReader(f) if row["kind"] == "text"}


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "n/a"


def main(path_a: str, path_b: str) -> None:
    a_rows, b_rows = load(path_a), load(path_b)
    name_a, name_b = Path(path_a).stem, Path(path_b).stem
    keys = sorted(a_rows.keys() & b_rows.keys())
    cards = sorted({card for card, _ in keys})
    only = sorted({c for c, _ in a_rows} ^ {c for c, _ in b_rows})
    if not keys:
        sys.exit("No cards in common.")

    fields = []
    for key in keys:
        field = key[1]
        t, a, b = a_rows[key]["truth"], a_rows[key]["pipeline"], b_rows[key]["pipeline"]
        fields.append({
            "card": key[0], "field": field, "truth": t, name_a: a, name_b: b,
            "a_ok": strict(a) == strict(t), "b_ok": strict(b) == strict(t),
            "a_ok_loose": loose(a) == loose(t), "a_ok_score": score_view(field, a) == score_view(field, t),
            "differ": strict(a) != strict(b),
            "differ_loose": loose(a) != loose(b),
            "a_bad_format": bad_format(field, a),
            "differ_score": score_view(field, a) != score_view(field, b),
        })

    n = len(fields)
    a_right = sum(f["a_ok"] for f in fields)
    b_right = sum(f["b_ok"] for f in fields)
    agree = [f for f in fields if not f["differ"]]
    print(f"{len(cards)} cards in both ({n} text fields)")
    if only:
        print(f"  left out, read by only one: {', '.join(only)}")
    print(f"\n  {name_a:<24} exact {pct(a_right, n):>6}   {a_right} / {n}   (primary)")
    print(f"  {name_b:<24} exact {pct(b_right, n):>6}   {b_right} / {n}   (checker)")
    print(f"  Agree                    {pct(len(agree), n):>6}   right when they agree "
          f"{pct(sum(f['a_ok'] for f in agree), len(agree))}")

    policies = [
        ("all differences", "differ", "a_ok", "exact"),
        ("transcription", None, "a_ok_loose", "ignoring punctuation"),
        ("grading", "differ_score", "a_ok_score", "as the score sees it"),
    ]
    print(f"\nReview policies (keep {name_a}'s reading, a reviewer fixes flagged fields):")
    print(f"  {'policy':<16} {'flags/card':>10} {'catches':>16} {'after review':>14}   accuracy measured")
    for label, flag_key, ok_key, measured in policies:
        flagged = [(f["differ_loose"] or f["a_bad_format"]) if flag_key is None else f[flag_key] for f in fields]
        errors = [not f[ok_key] for f in fields]
        caught = sum(e and fl for e, fl in zip(errors, flagged))
        missed = sum(e and not fl for e, fl in zip(errors, flagged))
        n_err = sum(errors)
        print(f"  {label:<16} {sum(flagged) / len(cards):>10.1f} {f'{caught}/{n_err} ({pct(caught, n_err)})':>16} "
              f"{pct(n - missed, n):>14}   {measured}")
        for f, fl in zip(fields, flagged):
            f.setdefault("flagged_by", [])
            if fl:
                f["flagged_by"].append(label)

    fmt_flags = [f for f in fields if f["a_bad_format"] and not f["differ_loose"]]
    if fmt_flags:
        real = sum(not f["a_ok_loose"] for f in fmt_flags)
        print(f"\n  format check alone flagged {len(fmt_flags)} fields where the readers agree: "
              f"{real} were real errors, {len(fmt_flags) - real} were right but unusual")

    out = Path(path_a).parent / "disagreements.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["card", "field", "truth", name_a, name_b, "right", "flagged_by"])
        writer.writeheader()
        for row in fields:
            if row["flagged_by"]:
                right = name_a if row["a_ok"] else name_b if row["b_ok"] else "neither"
                writer.writerow({"card": row["card"], "field": row["field"], "truth": row["truth"],
                                 name_a: row[name_a], name_b: row[name_b], "right": right,
                                 "flagged_by": "; ".join(row["flagged_by"])})
    print(f"\nAll flagged fields: {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])