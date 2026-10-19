"""Runs whole cards (front + back) through the pipeline and compares them to the truth files.

    python tools/eval_cards.py front_source.json back_source.json path/to/photos [more/photos ...]

Only cards with both sides labeled are used. Prints the accuracy summary
for the whole card, then text accuracy per side.
"""
import asyncio
import contextlib
import csv
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scan-pipeline"))

import pipeline  # noqa: E402
from scoring import score_card  # noqa: E402

TRUTH_DIR = ROOT / "dataset" / "truth"
RESULTS_PATH = TRUTH_DIR / "eval_results.csv"
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}


# HEIC photos are converted to JPEG bytes, since OpenCV can't read them.
def read_image_bytes(path: Path) -> bytes:
    if path.suffix.lower() not in {".heic", ".heif"}:
        return path.read_bytes()
    import pillow_heif
    from PIL import Image

    pillow_heif.register_heif_opener()
    buffer = io.BytesIO()
    with Image.open(path) as image:
        image.convert("RGB").save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def normalize(text) -> str:
    return " ".join(str(text or "").split()).upper()


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def find_photo(photos: list[Path], name: str) -> Path | None:
    for path in photos:
        if path.name == name:
            return path
    for path in photos:
        if path.stem == Path(name).stem and path.suffix.lower() in IMAGE_TYPES:
            return path
    return None


def as_checkbox_results(checked: dict) -> dict:
    return {k: {"label": "checked" if v else "unchecked"} for k, v in checked.items()}


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "n/a"


def card_table(result: dict) -> str:
    lines = [f"  {result['card']}   score {result['score']} (truth {result['truth_score']})",
             f"    {'':<6} {'field':<24} {'card':<32} pipeline"]
    for f in result["text"]:
        if f["t"] or f["p"]:
            mark = "right" if f["right"] else "WRONG"
            lines.append(f"    {mark:<6} {f['key']:<24} {f['truth']!r:<32} {f['read']!r}")
    return "\n".join(lines)


def summarize(results: list[dict]) -> str:
    text = defaultdict(int)
    field_right, field_seen, field_example = defaultdict(int), defaultdict(int), {}
    for r in results:
        for f in r["text"]:
            t, p = f["t"], f["p"]
            text["fields"] += 1
            text["exact"] += f["right"]
            text["presence"] += bool(t) == bool(p)
            text["made_up"] += not t and bool(p)
            text["missed"] += bool(t) and not p
            text["not_read"] += f["not_read"]
            text["dist"] += edit_distance(t, p)
            text["chars"] += len(t)
            field_seen[f["key"]] += 1
            field_right[f["key"]] += f["right"]
            if not f["right"]:
                field_example[f["key"]] = (f["truth"], f["read"])

    all_boxes = [b for r in results for b in r["boxes"]]
    box_wrong = defaultdict(int)
    for b in all_boxes:
        if not b["right"]:
            box_wrong[(b["key"], "false check" if b["read"] else "missed")] += 1
    box_right = sum(b["right"] for b in all_boxes)
    false_checks = sum(b["read"] and not b["truth"] for b in all_boxes)
    missed_checks = sum(b["truth"] and not b["read"] for b in all_boxes)

    item_wrong = defaultdict(int)
    for r in results:
        for item, _, _ in r["items_wrong"]:
            item_wrong[item] += 1

    cards = len(results)
    diffs = [r["score"] - r["truth_score"] for r in results]
    exact_scores = sum(d == 0 for d in diffs)
    avg = sum(diffs) / cards
    direction = "lower" if avg < 0 else "higher"
    plural = "s" if cards != 1 else ""
    worst = sorted((k for k in field_seen if field_right[k] < field_seen[k]),
                   key=lambda k: field_right[k] / field_seen[k])[:15]

    out = [f"Evaluated {cards} card{plural} with OCR backend: {os.environ.get('OCR_BACKEND', 'doctr')}", ""]
    out.append(f"TEXT FIELDS ({text['fields'] // cards} per card)")
    out.append(f"  Exact match          {pct(text['exact'], text['fields']):>6}   {text['exact']} / {text['fields']}")
    out.append(f"  Filled vs blank      {pct(text['presence'], text['fields']):>6}   {text['presence']} / {text['fields']}")
    out.append(f"    made up text       {text['made_up']:>6}   blank on card, pipeline read something")
    out.append(f"    missed text        {text['missed']:>6}   written on card, pipeline read nothing")
    out.append(f"  Character errors     {pct(text['dist'], max(1, text['chars'])):>6}   wrong characters per character written on cards")
    if text["not_read"]:
        out.append(f"  WARNING: {text['not_read']} truth fields missing from pipeline output (different source.json?)")

    out += ["", f"CHECKBOXES ({len(all_boxes) // cards} per card)"]
    out.append(f"  Accuracy             {pct(box_right, len(all_boxes)):>6}   {box_right} / {len(all_boxes)}")
    out.append(f"    false checks       {false_checks:>6}   unchecked on card, pipeline said checked")
    out.append(f"    missed checks      {missed_checks:>6}   checked on card, pipeline said unchecked")
    for (key, kind), n in sorted(box_wrong.items(), key=lambda kv: -kv[1]):
        out.append(f"      {key:<28} {kind}" + (f" x{n}" if n > 1 else ""))

    out += ["", "SCORE (30-item)"]
    out.append(f"  Exact score          {exact_scores:>6}   of {cards} card{plural}")
    out.append(f"  Average difference   {abs(avg):>6.1f}   " + (f"points {direction} than truth" if avg else "matches truth"))
    for item, n in sorted(item_wrong.items(), key=lambda kv: -kv[1]):
        out.append(f"    {item:<28} wrong on {n} card{'s' if n != 1 else ''}")

    if worst:
        out += ["", "WRONG FIELDS (right / cards, example: card -> pipeline)"]
        for key in worst:
            truth_value, read = field_example[key]
            out.append(f"  {key:<24} {field_right[key]}/{field_seen[key]}   {truth_value!r} -> {read!r}")
    return "\n".join(out)


def write_csv(results: list[dict]) -> None:
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["card", "kind", "key", "truth", "pipeline", "match"])
        for r in results:
            for t in r["text"]:
                writer.writerow([r["card"], "text", t["key"], t["truth"], t["read"], t["right"]])
            for b in r["boxes"]:
                writer.writerow([r["card"], "checkbox", b["key"], b["truth"], b["read"], b["right"]])
            for item, expected, got in r["items_wrong"]:
                writer.writerow([r["card"], "score", item, expected, got, False])




def compare_card(key: str, truths: dict, photos: dict) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        card = asyncio.run(pipeline.process_card(read_image_bytes(photos["front"]), read_image_bytes(photos["back"])))
    fields, checkboxes = card["fields"], card["checkboxes"]
    truth_fields = {**truths["front"]["fields"], **truths["back"]["fields"]}
    truth_boxes = {**truths["front"]["checkboxes"], **truths["back"]["checkboxes"]}

    text = []
    for side in ("front", "back"):
        for k, truth_value in truths[side]["fields"].items():
            t, p = normalize(truth_value), normalize(fields.get(k))
            text.append({"key": k, "truth": truth_value, "read": fields.get(k, "(not read)"), "right": t == p,
                         "t": t, "p": p, "not_read": k not in fields, "side": side})
    boxes = []
    for k, truth_checked in truth_boxes.items():
        result = checkboxes.get(k)
        read = result is not None and result["label"] == "checked"
        boxes.append({"key": k, "truth": truth_checked, "read": read, "right": truth_checked == read})

    ours = card["score"]
    expected = score_card(truth_fields, as_checkbox_results(truth_boxes))
    items_wrong = [(item, value, ours["score_components"][item]) for item, value in expected["score_components"].items()
                   if value is not None and ours["score_components"][item] != value]
    return {"card": key, "text": text, "boxes": boxes, "score": int(ours["total_score"]),
            "truth_score": int(expected["total_score"]), "items_wrong": items_wrong,
            "not_scored": len(ours["not_scored"])}


def side_summary(results: list[dict]) -> str:
    out = ["", "", "TEXT FIELDS BY SIDE"]
    for side in ("front", "back"):
        rows = [f for r in results for f in r["text"] if f["side"] == side]
        written = [f for f in rows if f["t"]]
        dist = sum(edit_distance(f["t"], f["p"]) for f in written)
        chars = sum(len(f["t"]) for f in written)
        out.append(f"  {side:<6} exact {pct(sum(f['right'] for f in rows), len(rows)):>6}   "
                   f"filled vs blank {pct(sum(bool(f['t']) == bool(f['p']) for f in rows), len(rows)):>6}   "
                   f"character errors {pct(dist, max(1, chars)):>6}   ({len(rows) // max(1, len(results))} fields per card)")
    left = sum(r["not_scored"] for r in results) / len(results)
    out.append(f"\n  Checklist items still not scored: {left:.1f} of 30 per card")
    return "\n".join(out)


def main(front_path: str, back_path: str, *photo_dirs: str) -> None:
    pipeline._sources.update(front=json.loads(Path(front_path).read_text()),
                             back=json.loads(Path(back_path).read_text()))
    photos = [p for d in photo_dirs for p in Path(d).iterdir()]
    truths = {}
    for path in sorted(TRUTH_DIR.glob("*.truth.json")):
        truth = json.loads(path.read_text())
        for key, sides in pipeline.pair_photos([truth["image"]]).items():
            for side in sides:
                truths.setdefault(key, {})[side] = truth
    cards = {k: v for k, v in truths.items() if len(v) == 2}
    print(f"{len(cards)} cards with both sides labeled")

    results = []
    for key, sides in sorted(cards.items()):
        found = {side: find_photo(photos, sides[side]["image"]) for side in sides}
        missing = [sides[s]["image"] for s, p in found.items() if p is None]
        if missing:
            print(f"skip {key}: photo not found: {', '.join(missing)}")
            continue
        try:
            results.append(compare_card(key, sides, found))
        except Exception as err:
            print(f"skip {key}: pipeline failed: {err}")
            continue
        print("\n" + card_table(results[-1]))
    if not results:
        sys.exit("No cards evaluated.")
    write_csv(results)
    print("\n" + summarize(results) + side_summary(results))
    print(f"\nAll results: {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *sys.argv[3:])