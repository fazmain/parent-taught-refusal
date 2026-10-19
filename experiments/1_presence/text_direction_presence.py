#!/usr/bin/env python3
"""Text-direction presence check: do the parent VLM's harmfulness/refusal directions survive in NORA's residual stream?

For each model: run D_text through its own prompt template, cache residuals at t_inst and t_post_inst for every layer,
fit difference-in-means directions on the train split, report held-out AUROC. Then project the OTHER model's test
activations onto this model's directions (transfer) and compute per-layer cosine between the two models' directions.
Activations are saved to results/acts/<model>.npz for reuse (harm probe). One row per metric in results/results.jsonl.

  python3 experiments/1_presence/text_direction_presence.py --models parent nora --n-train 128 --n-test 128
  python3 experiments/1_presence/text_direction_presence.py --models smoke --n-train 4 --n-test 4 --layers 0 6 12 --device cpu   # local smoke
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from refusal.paths import DATA_DIR, RESULTS_DIR  # noqa: E402
from refusal.models import ActivationCache, load_model, residuals  # noqa: E402
from refusal.prompts import fixed_frame, load_dtext, robot_text  # noqa: E402
from refusal.directions import auroc, cosine, diff_in_means  # noqa: E402

POS = ("t_inst", "t_post_inst")


def collect(L, prompts, layers, image, wrap, tag):
    """Returns acts[layer][pos] -> (n, d) float32."""
    cache = ActivationCache(L, layers)
    acts = {li: {p: [] for p in POS} for li in layers}
    t0 = time.time()
    for i, r in enumerate(prompts):
        text = robot_text(r["instruction"], wrap) if L.name == "nora" else r["instruction"]
        a, pos = residuals(L, cache, text, image if L.kind == "vlm" else None)
        if i == 0:
            ids = L.tokenizer.decode
            print(f"[{L.name}/{tag}] positions {pos}; prompt len {pos['t_post_inst']+1}")
        for li in layers:
            for k, p in enumerate(POS):
                acts[li][p].append(a[li][k].numpy())
        if (i + 1) % 32 == 0:
            print(f"[{L.name}/{tag}] {i+1}/{len(prompts)} {time.time()-t0:.0f}s", flush=True)
    return {li: {p: np.stack(v) for p, v in d.items()} for li, d in acts.items()}


def fit_dirs(acts_h, acts_b, layers):
    return {li: {p: diff_in_means(acts_h[li][p], acts_b[li][p]) for p in POS} for li in layers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["parent", "nora"])
    ap.add_argument("--n-train", type=int, default=128)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--image", default=str(DATA_DIR / "fixed_frame.png"))
    ap.add_argument("--wrap", action="store_true", help="use 'What action should the robot take to {x}?' for NORA")
    ap.add_argument("--parent-text-only", action="store_true", help="no image for the parent (Arditi-standard text prompts)")
    ap.add_argument("--tag", default="", help="free-text tag written into every row")
    ap.add_argument("--out", default=str(RESULTS_DIR / "results.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    acts_dir = out.parent / "acts"; acts_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    base = {"exp": "poc_presence", "dataset": "d_text", "ts": stamp, "seed": a.seed, "wrap": a.wrap,
            "parent_text_only": a.parent_text_only, "tag": a.tag}
    rows = []

    tr_h, tr_b = load_dtext("train", a.n_train, a.seed)
    te_h, te_b = load_dtext("test", a.n_test, a.seed)
    image = fixed_frame(a.image)

    store = {}  # name -> dict(acts_test_h, acts_test_b, dirs, layers)
    for name in a.models:
        L = load_model(name, a.device)
        layers = a.layers if a.layers is not None else list(range(L.n_layers))
        img = None if (name == "parent" and a.parent_text_only) else image
        A = {tag: collect(L, ps, layers, img, a.wrap, tag)
             for tag, ps in (("train_h", tr_h), ("train_b", tr_b), ("test_h", te_h), ("test_b", te_b))}
        dirs = fit_dirs(A["train_h"], A["train_b"], layers)
        for li in layers:
            for p in POS:
                rows.append({**base, "model_acts": name, "direction_src": name, "layer": li, "position": p,
                             "metric": "auroc", "value": auroc(A["test_h"][li][p], A["test_b"][li][p], dirs[li][p]),
                             "n_pos": len(te_h), "n_neg": len(te_b)})
        np.savez_compressed(acts_dir / f"{name}_dtext{a.tag}.npz",
                            **{f"{tag}_L{li}_{p}": A[tag][li][p] for tag in A for li in layers for p in POS},
                            layers=np.array(layers))
        store[name] = {"test_h": A["test_h"], "test_b": A["test_b"], "dirs": dirs, "layers": layers}
        del L; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # transfer + cosine for every ordered pair
    for src in store:
        for dst in store:
            if src == dst:
                continue
            common = sorted(set(store[src]["layers"]) & set(store[dst]["layers"]))
            for li in common:
                for p in POS:
                    d = store[src]["dirs"][li][p]
                    rows.append({**base, "model_acts": dst, "direction_src": src, "layer": li, "position": p, "metric": "auroc",
                                 "value": auroc(store[dst]["test_h"][li][p], store[dst]["test_b"][li][p], d),
                                 "n_pos": len(te_h), "n_neg": len(te_b)})
                    rows.append({**base, "model_acts": dst, "direction_src": src, "layer": li, "position": p, "metric": "cosine",
                                 "value": cosine(d, store[dst]["dirs"][li][p])})

    with out.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {out}")
    for r in rows:
        if r["metric"] == "auroc" and r["position"] == "t_inst":
            print(f"  acts={r['model_acts']:6s} dir={r['direction_src']:6s} L{r['layer']:2d} t_inst  AUROC={r['value']:.3f}")


if __name__ == "__main__":
    main()
