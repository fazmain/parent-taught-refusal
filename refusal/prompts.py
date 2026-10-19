"""Datasets and prompt templates.

D_text: Arditi et al. splits (harmful = AdvBench/HarmBench/JBB/MaliciousInstruct; harmless = Alpaca), cached under data/arditi/.
Robot template: NORA's deployed LIBERO inference passes the RAW instruction as the user text (nora/experiments/libero/nora_utils.py),
so that is the default. `wrap=True` gives the training-code wrapper "What action should the robot take to {x}?".
"""
from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path

from PIL import Image

from refusal.paths import DATA_DIR

ARDITI = "https://raw.githubusercontent.com/andyrdt/refusal_direction/main/dataset/splits/{name}.json"
DATA = DATA_DIR
ROBOT_WRAP = "What action should the robot take to {x}?"


def _arditi(name: str) -> list[dict]:
    p = DATA / "arditi" / f"{name}.json"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ARDITI.format(name=name), p)
    return json.loads(p.read_text())


def load_dtext(split: str, n: int, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Returns (harmful, harmless), each a list of {instruction, label, category} of length ≤ n, shuffled with `seed`."""
    rng = random.Random(seed)
    out = []
    for label in ("harmful", "harmless"):
        rows = _arditi(f"{label}_{split}")
        rng.shuffle(rows)
        out.append([{"instruction": r["instruction"], "label": label, "category": r.get("category", ""), "split": split}
                    for r in rows[:n]])
    return out[0], out[1]


def robot_text(instruction: str, wrap: bool = False) -> str:
    return ROBOT_WRAP.format(x=instruction) if wrap else instruction


def fixed_frame(path: str | Path = DATA / "fixed_frame.png") -> Image.Image:
    return Image.open(path).convert("RGB")


def make_placeholder_frame(path: str | Path = DATA / "fixed_frame.png") -> Path:
    """Synthetic 224x224 tabletop (grey table, three coloured objects). POC only; replace with a LIBERO frame later."""
    from PIL import ImageDraw
    img = Image.new("RGB", (224, 224), (60, 60, 70))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 90, 224, 224], fill=(170, 150, 120))          # table
    d.ellipse([40, 130, 100, 170], fill=(200, 200, 210))          # plate
    d.ellipse([55, 120, 85, 145], fill=(120, 140, 200))           # bowl
    d.rectangle([140, 125, 175, 165], fill=(200, 80, 60))         # mug
    d.rectangle([150, 60, 200, 95], fill=(90, 90, 100))           # drawer/box
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return Path(path)
