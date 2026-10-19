"""Scoring demo: runs whole cards (front + back) through the pipeline, scores them,
and compares TrOCR with the LLM reader and with the truth files when there are any.

    $env:OCR_BACKEND="trocr"; $env:TROCR_MODEL="models/trocr-dd1380-v6"
    python tools/eval_app.py front_source.json back_source.json

Runs each card exactly like eval_cards.py (pipeline.process_card), once with
TrOCR and, if ticked, once more with the LLM. Upload photos named like
#38_Front_Case1.HEIC / #38_Back_Case1.HEIC. Cards without a truth file are
still read and scored, just not compared.
"""
import asyncio
import contextlib
import io
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "scan-pipeline"))

from eval_cli import IMAGE_TYPES, TRUTH_DIR, as_checkbox_results, normalize, pct, read_image_bytes  # noqa: E402
import pipeline  # noqa: E402
from pipeline import get_source, pair_photos  # noqa: E402
from scoring import score_card  # noqa: E402

DISPLAY_WIDTH = 1100
READERS = ["TrOCR", "LLM"]
READER_NAMES = {"trocr": "TrOCR", "llm": "LLM"}

# BGR
RIGHT = (80, 180, 80)
WRONG = (60, 60, 230)
DISAGREE = (0, 140, 255)
BLANK = (190, 190, 190)


class LLMFailed(Exception):
    pass


@contextlib.contextmanager
def backend(name: str):
    old = os.environ.get("OCR_BACKEND")
    os.environ["OCR_BACKEND"] = name
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("OCR_BACKEND")
        else:
            os.environ["OCR_BACKEND"] = old


# One full process_card run, the same call eval_cards.py makes.
def read_card(photos: dict, name: str, times: dict) -> dict:
    raw = {side: read_image_bytes(path) for side, path in photos.items()}
    start = time.perf_counter()
    with backend(name), contextlib.redirect_stdout(io.StringIO()):
        card = asyncio.run(pipeline.process_card(raw.get("front"), raw.get("back")))
    times[READER_NAMES[name]] = time.perf_counter() - start
    return card


def load_truth(photo: Path) -> dict | None:
    path = TRUTH_DIR / f"{photo.stem}.truth.json"
    return json.loads(path.read_text()) if path.exists() else None


def run_card(key: str, photos: dict, use_llm: bool) -> dict:
    times = {}
    runs = {"TrOCR": read_card(photos, "trocr", times)}
    if use_llm:
        try:
            runs["LLM"] = read_card(photos, "llm", times)
        except Exception as err:
            raise LLMFailed(err) from err
    main = runs["TrOCR"]
    sides = {side: {"aligned": s["aligned"], "checkboxes": s["checkboxes"],
                    "fields": {r: run["sides"][side]["fields"] for r, run in runs.items()}}
             for side, s in main["sides"].items()}
    readers = list(runs)
    fields = {r: run["fields"] for r, run in runs.items()}
    scores = {r: run["score"] for r, run in runs.items()}

    truths = {side: load_truth(path) for side, path in photos.items()}
    truth = None
    if all(truths.values()):
        truth = {"fields": {k: v for t in truths.values() for k, v in t["fields"].items()},
                 "checkboxes": {k: v for t in truths.values() for k, v in t["checkboxes"].items()}}
        scores["truth"] = score_card(truth["fields"], as_checkbox_results(truth["checkboxes"]))
    return {"card": key, "sides": sides, "fields": fields, "checkboxes": main["checkboxes"], "scores": scores,
            "truth": truth, "times": times, "readers": readers}


def items_wrong(card: dict, reader: str) -> list[str]:
    if "truth" not in card["scores"]:
        return []
    expected = card["scores"]["truth"]["score_components"]
    got = card["scores"][reader]["score_components"]
    return [item for item, value in expected.items() if value is not None and got[item] != value]


def total(card: dict, who: str) -> int:
    return int(card["scores"][who]["total_score"])


def summarize(cards: list[dict]) -> str:
    labeled = [c for c in cards if c["truth"]]
    readers = [r for r in READERS if all(r in c["readers"] for c in cards)]
    out = [f"**{len(cards)} card{'s' if len(cards) != 1 else ''}** · {len(labeled)} with truth files", ""]

    if labeled:
        out += ["| | Text exact | Filled vs blank | Exact score | Avg points off |", "|---|---|---|---|---|"]
        for r in readers:
            rows = [(normalize(v), normalize(c["fields"][r].get(k))) for c in labeled
                    for k, v in c["truth"]["fields"].items()]
            exact = sum(t == p for t, p in rows)
            presence = sum(bool(t) == bool(p) for t, p in rows)
            exact_scores = sum(total(c, r) == total(c, "truth") for c in labeled)
            off = sum(abs(total(c, r) - total(c, "truth")) for c in labeled) / len(labeled)
            out.append(f"| **{r}** | {pct(exact, len(rows))} | {pct(presence, len(rows))} | "
                       f"{exact_scores} / {len(labeled)} cards | {off:.2f} |")
        boxes = [(v, c["checkboxes"].get(k, {}).get("label") == "checked") for c in labeled
                 for k, v in c["truth"]["checkboxes"].items()]
        out += ["", f"Checkboxes: {pct(sum(t == p for t, p in boxes), len(boxes))} right (shared by both readers)"]
        for r in readers:
            wrong = defaultdict(int)
            for c in labeled:
                for item in items_wrong(c, r):
                    wrong[item] += 1
            if wrong:
                top = sorted(wrong.items(), key=lambda kv: -kv[1])[:5]
                out.append(f"{r} checklist items wrong: " + ", ".join(f"{k} ×{n}" for k, n in top))

    if "LLM" in readers:
        pairs = [(c, k) for c in cards for k in c["fields"]["TrOCR"]]
        agree = sum(normalize(c["fields"]["TrOCR"][k]) == normalize(c["fields"]["LLM"].get(k)) for c, k in pairs)
        line = f"TrOCR and LLM agree on {pct(agree, len(pairs))} of fields"
        split = defaultdict(int)
        for c in labeled:
            for k, v in c["truth"]["fields"].items():
                t, a, b = normalize(v), normalize(c["fields"]["TrOCR"].get(k)), normalize(c["fields"]["LLM"].get(k))
                if a != b:
                    split["TrOCR right" if a == t else "LLM right" if b == t else "both wrong"] += 1
        if split:
            line += " · where they disagree: " + ", ".join(f"{k} {n}" for k, n in split.items())
        out += ["", line]

    timed = cards[1:] if len(cards) > 1 else cards
    out += ["", "| Reader | Avg s per card | Fastest | Slowest |", "|---|---|---|---|"]
    for r in readers:
        t = [c["times"][r] for c in timed]
        out.append(f"| {r} | **{sum(t) / len(t):.1f}** | {min(t):.1f} | {max(t):.1f} |")
    if len(cards) > 1:
        out.append("\n_Times are the whole card (both sides: scan, align, checkboxes, text) and leave out "
                   "the first card, which includes loading the models._")
    return "\n".join(out)


def draw(card: dict, side: str):
    if side not in card["sides"]:
        return None
    result = card["sides"][side]
    image = result["aligned"]
    scale = DISPLAY_WIDTH / image.shape[1]
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = image.shape[:2]
    source = get_source(side)
    geometry = {wd["value"]: wd["geometry"] for b in source["blocks"] for l in b["lines"] for wd in l["words"]
                if wd["value"].startswith("field_")}
    truth = card["truth"]
    for key, ((x0, y0), (x1, y1)) in geometry.items():
        p0, p1 = (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h))
        read = None
        if key.startswith("field_image_"):
            got = result["checkboxes"].get(key, {}).get("label") == "checked"
            written, right = got or (truth and truth["checkboxes"].get(key)), \
                truth is None or truth["checkboxes"].get(key) == got
        else:
            got = normalize(result["fields"]["TrOCR"].get(key))
            want = normalize(truth["fields"].get(key)) if truth else None
            other = normalize(result["fields"]["LLM"].get(key)) if "LLM" in result["fields"] else got
            written = bool(got or want or other)
            right = got == want if truth else got == other
            read = result["fields"]["TrOCR"].get(key, "")
        color = BLANK if not written else RIGHT if right else WRONG if truth else DISAGREE
        cv2.rectangle(image, p0, p1, color, 2 if right else 3)
        if not right and read is not None:
            cv2.putText(image, read or "(blank)", (p0[0] + 4, p1[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def card_view(key: str, state: dict):
    if not state or key not in state["cards"]:
        return None, None, "", None
    card = state["cards"][key]
    readers, truth = card["readers"], card["truth"]
    who = (["truth"] if truth else []) + readers

    scores = " · ".join(f"{w} **{total(card, w)}**/30" for w in who)
    took = " · ".join(f"{r} {card['times'][r]:.1f} s" for r in readers)
    info = [f"### {key}", f"30-item score: {scores}", f"Time for the whole card: {took}"]
    for r in readers:
        wrong = items_wrong(card, r)
        if wrong:
            info.append(f"{r} scored wrong: {', '.join(wrong)}")
    not_scored = card["scores"][readers[0]]["not_scored"]
    if not_scored:
        info.append(f"Not scored ({len(not_scored)}): {', '.join(not_scored)}")

    rows = []
    for k in card["fields"]["TrOCR"]:
        values = [truth["fields"].get(k, "") if truth else None] + [card["fields"][r].get(k, "") for r in readers]
        shown = [v for v in values if v is not None]
        if len({normalize(v) for v in shown}) > 1:
            rows.append([k.removeprefix("field_")] + shown)
    if truth:
        for k, want in truth["checkboxes"].items():
            got = card["checkboxes"].get(k, {}).get("label") == "checked"
            if got != want:
                mark = lambda v: "checked" if v else "unchecked"
                rows.append([k.removeprefix("field_")] + [mark(want)] + [mark(got)] * len(readers))
    columns = ["field"] + (["card"] if truth else []) + readers
    table = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame([["no differences"] + [""] * (len(columns) - 1)],
                                                                         columns=columns)
    return draw(card, "front"), draw(card, "back"), "\n\n".join(info), table


def run(photo_paths, use_llm):
    if not photo_paths:
        raise gr.Error("Upload card photos (named _Front_ / _Back_).")
    photos = [p for p in photo_paths if Path(p).suffix.lower() in IMAGE_TYPES]
    pairs = pair_photos(photos)
    if not pairs:
        raise gr.Error("No photos with Front or Back in the name.")
    state, notes = {"cards": {}}, []
    for n, (key, sides) in enumerate(sorted(pairs.items()), 1):
        try:
            card = run_card(key, sides, use_llm)
        except LLMFailed as err:
            notes.append(f"LLM reader failed ({err}), continuing with TrOCR only.")
            use_llm = False
            try:
                card = run_card(key, sides, use_llm)
            except Exception as err:
                notes.append(f"skip {key}: {err}")
                continue
        except Exception as err:
            notes.append(f"skip {key}: {err}")
            continue
        state["cards"][key] = card
        cards = list(state["cards"].values())
        summary = f"Reading card {n} of {len(pairs)}...\n\n" if n < len(pairs) else ""
        summary += summarize(cards) + ("\n\n" + "\n\n".join(notes) if notes else "")
        yield (summary, gr.update(choices=list(state["cards"]), value=key), *card_view(key, state), state)
    if not state["cards"]:
        raise gr.Error("No cards could be read. " + " ".join(notes))


with gr.Blocks(title="DD-1380 Scoring Demo") as demo:
    with gr.Row():
        photo_paths = gr.Files(label="Card photos (front and back)", file_types=sorted(IMAGE_TYPES),
                               type="filepath", scale=4)
        with gr.Column(scale=1):
            use_llm = gr.Checkbox(label="Compare with LLM reader (uses the API)", value=False)
            run_btn = gr.Button("Run", variant="primary")
    summary = gr.Markdown()
    card_pick = gr.Dropdown(label="Card", choices=[])
    info = gr.Markdown()
    table = gr.Dataframe(label="Differences (card = truth file)", interactive=False, wrap=True)
    with gr.Row():
        front = gr.Image(label="Front · green right · red wrong (TrOCR reading shown) · orange TrOCR ≠ LLM",
                         type="numpy", interactive=False)
        back = gr.Image(label="Back", type="numpy", interactive=False)

    state = gr.State({})
    run_btn.click(run, [photo_paths, use_llm], [summary, card_pick, front, back, info, table, state])
    card_pick.input(card_view, [card_pick, state], [front, back, info, table])

if __name__ == "__main__":
    pipeline._sources.update(front=json.loads(Path(sys.argv[1]).read_text()),
                             back=json.loads(Path(sys.argv[2]).read_text()))
    demo.launch()
