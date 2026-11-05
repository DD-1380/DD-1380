#!/usr/bin/env python3
"""Import OCR once, annotate in labelme, export output.json once at exit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

CROP_ORIENTATION = {"value": 0, "confidence": None}
LABELME_VERSION = "6.3.1"


def seed_labelme(ocr: dict, image: Path, labelme_path: Path) -> None:
    page_h, page_w = ocr["dimensions"]
    page_w, page_h = int(page_w), int(page_h)
    shapes: list[dict] = []

    for block in ocr.get("blocks", []):
        for line_idx, line in enumerate(block.get("lines", [])):
            for word in line.get("words") or []:
                value = str(word.get("value", "")).strip()
                geometry = word.get("geometry")
                if not value or not geometry:
                    continue
                (x0, y0), (x1, y1) = geometry
                shapes.append(
                    {
                        "label": value,
                        "points": [
                            [min(x0, x1) * page_w, min(y0, y1) * page_h],
                            [max(x0, x1) * page_w, max(y0, y1) * page_h],
                        ],
                        "group_id": line_idx,
                        "description": json.dumps(
                            {
                                "confidence": word.get("confidence"),
                                "objectness_score": word.get("objectness_score"),
                                "crop_orientation": word.get("crop_orientation"),
                                "line_objectness_score": line.get("objectness_score"),
                            }
                        ),
                        "shape_type": "rectangle",
                        "flags": {},
                        "mask": None,
                    }
                )

    ann_dir = labelme_path.parent
    try:
        image_rel = image.resolve().relative_to(ann_dir.resolve()).as_posix()
    except ValueError:
        image_rel = str(image.resolve())

    labelme_path.write_text(
        json.dumps(
            {
                "version": LABELME_VERSION,
                "flags": {},
                "shapes": shapes,
                "imagePath": image_rel,
                "imageData": None,
                "imageHeight": page_h,
                "imageWidth": page_w,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Seeded {len(shapes)} word box(es) for labelme")


def lines_from_labelme(labelme: dict, page_w: int, page_h: int) -> list[dict]:
    groups: dict[int | str, list[dict]] = defaultdict(list)
    solo_idx = 0

    for shape in labelme.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        label = str(shape.get("label", "")).strip()
        points = shape.get("points") or []
        if not label or len(points) != 2:
            continue

        group_id = shape.get("group_id")
        if group_id is None:
            group_id = f"solo_{solo_idx}"
            solo_idx += 1
        groups[group_id].append(shape)

    def group_sort_key(item: tuple[int | str, list[dict]]) -> tuple[float, float]:
        _, shapes = item
        tops = [min(p[1] for p in shape["points"]) for shape in shapes]
        lefts = [min(p[0] for p in shape["points"]) for shape in shapes]
        return min(tops), min(lefts)

    lines: list[dict] = []
    for _, shapes in sorted(groups.items(), key=group_sort_key):
        shapes.sort(key=lambda shape: min(p[0] for p in shape["points"]))

        words: list[dict] = []
        line_scores: list[float] = []
        for shape in shapes:
            label = str(shape.get("label", "")).strip()
            (x0, y0), (x1, y1) = shape["points"]
            left, top = min(x0, x1), min(y0, y1)
            right, bottom = max(x0, x1), max(y0, y1)
            word_geometry = [
                [max(0.0, min(1.0, left / page_w)), max(0.0, min(1.0, top / page_h))],
                [max(0.0, min(1.0, right / page_w)), max(0.0, min(1.0, bottom / page_h))],
            ]

            meta: dict = {}
            description = shape.get("description") or ""
            if description:
                try:
                    meta = json.loads(description)
                except json.JSONDecodeError:
                    pass

            crop_orientation = meta.get("crop_orientation")
            if not isinstance(crop_orientation, dict):
                crop_orientation = deepcopy(CROP_ORIENTATION)

            word = {
                "value": label,
                "geometry": word_geometry,
                "confidence": meta.get("confidence", 1.0),
                "objectness_score": meta.get("objectness_score", 1.0),
                "crop_orientation": crop_orientation,
            }
            words.append(word)
            line_score = meta.get("line_objectness_score")
            if isinstance(line_score, (int, float)):
                line_scores.append(float(line_score))
            line_scores.append(float(word["objectness_score"]))

        if not words:
            continue

        xs = [coord for word in words for coord in (word["geometry"][0][0], word["geometry"][1][0])]
        ys = [coord for word in words for coord in (word["geometry"][0][1], word["geometry"][1][1])]
        lines.append(
            {
                "geometry": [[min(xs), min(ys)], [max(xs), max(ys)]],
                "objectness_score": max(line_scores) if line_scores else 1.0,
                "words": words,
            }
        )

    return lines


def annotate(ocr_path: Path, image: Path) -> Path:
    if not image.is_file():
        raise SystemExit(f"Image not found: {image}")

    ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
    if "blocks" not in ocr:
        raise SystemExit(f"Not a docTR OCR page JSON: {ocr_path}")

    with tempfile.TemporaryDirectory(prefix="labelme-") as ann_dir_name:
        ann_dir = Path(ann_dir_name)
        labelme_file = ann_dir / f"{image.stem}.json"
        seed_labelme(ocr, image, labelme_file)

        subprocess.run(["labelme", str(image), "--output", str(ann_dir)], check=True)

        labelme = json.loads(labelme_file.read_text(encoding="utf-8"))
        page_h, page_w = ocr["dimensions"]
        page_w = int(labelme.get("imageWidth") or page_w)
        page_h = int(labelme.get("imageHeight") or page_h)
        lines = lines_from_labelme(labelme, page_w, page_h)

        document = deepcopy(ocr)
        blocks = document.get("blocks") or []
        if not blocks:
            raise SystemExit("OCR JSON has no blocks")
        blocks[0]["lines"] = lines

        out = ocr_path.parent / "output.json"
        out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    raw_words = sum(len(line.get("words") or []) for line in ocr["blocks"][0]["lines"])
    out_words = sum(len(line.get("words") or []) for line in lines)
    print(f"Wrote {out} ({len(lines)} lines, {out_words} words from labelme; imported {raw_words} words)")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ocr", type=Path, help="OCR JSON to import once (e.g. RAW-OCR.json)")
    parser.add_argument("image", type=Path, help="Image for labelme")
    args = parser.parse_args()
    try:
        annotate(args.ocr, args.image)
    except subprocess.CalledProcessError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
