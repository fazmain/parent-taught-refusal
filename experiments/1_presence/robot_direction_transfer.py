#!/usr/bin/env python3
"""Direction transfer on D_robot: does a harm direction fitted on the PARENT with robot instructions survive into the VLA?

The parent's D_text direction only reached AUROC 0.64 on robot prompts inside nora_lg. Here the direction is fitted
on D_robot itself. Repeated stratified 50/50 splits (difference-in-means on the fit half, AUROC on the held-out
half) give four direction x activations combinations -- parent->vla, vla->vla, parent->parent, vla->parent -- plus
the ood_benign-vs-id_benign novelty confound for the two directions read inside the VLA, and cos(d_parent, d_vla).

  python3 experiments/1_presence/robot_direction_transfer.py
  python3 experiments/1_presence/robot_direction_transfer.py --layers 0 1 2 --n-splits 5 --results /tmp/r.jsonl --fig /tmp/f.png
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from refusal.paths import RESULTS_DIR  # noqa: E402
from refusal.directions import auroc, cosine, diff_in_means  # noqa: E402

POS = ("t_inst", "t_post_inst")
BENIGN = ("id_benign", "ood_benign", "lexical")
TRANSFERS = ("parent->vla", "vla->vla", "parent->parent", "vla->parent")
STYLE = {"parent->vla": "-", "vla->vla": "--", "parent->parent": ":", "vla->parent": "-."}


def mean_std(v):
    return (float(np.mean(v)), float(np.std(v))) if len(v) else (float("nan"), float("nan"))


def load_acts(path, layers, ids):
    """acts[layer][position] -> (n, d) float32, in prompt-file row order. Any mismatch is fatal."""
    z = np.load(path)
    got = [str(x) for x in z["ids"]]
    if got != ids:
        raise SystemExit(f"{path}: ids do not match the {len(ids)} prompt rows (got {len(got)})")
    missing = [f"L{li}_{p}" for li in layers for p in POS if f"L{li}_{p}" not in z]
    if missing:
        raise SystemExit(f"{path}: missing keys {missing}")
    return {li: {p: np.asarray(z[f"L{li}_{p}"], dtype=np.float32) for p in POS} for li in layers}


def make_fig(path, layers, auc, cos, stamp, n_splits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, p in zip(axes, POS):
        for t in TRANSFERS:
            m, s = np.array([mean_std(auc[(t, li, p)]) for li in layers]).T
            ln, = ax.plot(layers, m, STYLE[t], marker="o", ms=3, label=t)
            ax.fill_between(layers, m - s, m + s, color=ln.get_color(), alpha=0.15, lw=0)
        ax.axhline(0.5, color="grey", lw=0.5)
        ax.set_title(f"position = {p}"); ax.set_xlabel("layer"); ax.set_ylim(0.4, 1.02)
        ax2 = ax.twinx(); ax2.plot(layers, [cos[(li, p)] for li in layers], color="k", lw=1, alpha=0.5)
        ax2.set_ylim(-0.2, 1.05); ax2.set_ylabel("cos(d_parent, d_vla)", color="k")
    axes[0].set_ylabel(f"held-out AUROC (D_robot fit, {n_splits} splits)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"D_robot harm direction transfer, parent <-> nora_lg ({stamp})")
    fig.tight_layout(); fig.savefig(path, dpi=150)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(RESULTS_DIR / "raw/pod4/drobot_prompts.jsonl"))
    ap.add_argument("--acts-dir", default=str(RESULTS_DIR / "raw/pod4/acts"))
    ap.add_argument("--n-splits", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", nargs="*", type=int, default=list(range(36)))
    ap.add_argument("--results", default=str(RESULTS_DIR / "raw/pod4/results_analysis.jsonl"))
    ap.add_argument("--fig", default=str(ROOT / "figures/fig1b_drobot_transfer.png"))
    a = ap.parse_args()

    layers = sorted(a.layers)
    rows = [json.loads(l) for l in Path(a.prompts).read_text().splitlines() if l.strip()]
    ids = [str(r["id"]) for r in rows]
    cond = np.array([r["condition"] for r in rows])
    cat = np.array([r.get("category", "") for r in rows])
    harm = cond == "harm"
    acts = {"parent": load_acts(Path(a.acts_dir) / "drobot_parent.npz", layers, ids),
            "vla": load_acts(Path(a.acts_dir) / "drobot_nora_lg.npz", layers, ids)}
    cats = sorted(set(cat[harm]))
    print(f"[data] {len(rows)} rows; " + ", ".join(f"{c}={int((cond == c).sum())}" for c in ("harm",) + BENIGN)
          + f"; harm categories {cats}; layers {layers[0]}..{layers[-1]}; {a.n_splits} splits, seed {a.seed}")

    rng = np.random.default_rng(a.seed)
    groups = [np.flatnonzero(cond == c) for c in sorted(set(cond))]  # 50/50 split is stratified by condition
    auc = {(t, li, p): [] for t in TRANSFERS for li in layers for p in POS}
    nov = {(t, li, p): [] for t in ("parent->vla", "vla->vla") for li in layers for p in POS}
    percat = {(c, li): [] for c in cats for li in layers}
    for _ in range(a.n_splits):
        fit, held = [], []
        for g in groups:
            g = rng.permutation(g); k = len(g) // 2
            fit += list(g[:k]); held += list(g[k:])
        fit, held = np.array(sorted(fit)), np.array(sorted(held))
        hf, bf = fit[harm[fit]], fit[~harm[fit]]
        hh, bh = held[harm[held]], held[~harm[held]]
        oh, ih = held[cond[held] == "ood_benign"], held[cond[held] == "id_benign"]
        for li in layers:
            for p in POS:
                P, V = acts["parent"][li][p], acts["vla"][li][p]
                dp, dv = diff_in_means(P[hf], P[bf]), diff_in_means(V[hf], V[bf])
                auc[("parent->vla", li, p)].append(auroc(V[hh], V[bh], dp))
                auc[("vla->vla", li, p)].append(auroc(V[hh], V[bh], dv))
                auc[("parent->parent", li, p)].append(auroc(P[hh], P[bh], dp))
                auc[("vla->parent", li, p)].append(auroc(P[hh], P[bh], dv))
                if len(oh) and len(ih):
                    nov[("parent->vla", li, p)].append(auroc(V[oh], V[ih], dp))
                    nov[("vla->vla", li, p)].append(auroc(V[oh], V[ih], dv))
                if p == "t_post_inst":
                    for c in cats:
                        hc = hh[cat[hh] == c]
                        if len(hc):
                            percat[(c, li)].append(auroc(V[hc], V[bh], dp))

    cos = {(li, p): cosine(diff_in_means(acts["parent"][li][p][harm], acts["parent"][li][p][~harm]),
                           diff_in_means(acts["vla"][li][p][harm], acts["vla"][li][p][~harm]))
           for li in layers for p in POS}
    best = {(t, p): max(layers, key=lambda li: mean_std(auc[(t, li, p)])[0]) for t in TRANSFERS for p in POS}

    print(f"\nbest layer by mean held-out AUROC, harm vs benign_all ({a.n_splits} splits)")
    print(f"  {'transfer':15s} {'position':13s} {'layer':>5s} {'AUROC':>16s} {'cos(d_p,d_v)':>13s}")
    for t in TRANSFERS:
        for p in POS:
            li = best[(t, p)]; m, s = mean_std(auc[(t, li, p)])
            print(f"  {t:15s} {p:13s} {li:5d} {m:9.3f}+-{s:5.3f} {cos[(li, p)]:13.3f}")

    print("\nnovelty confound at those layers (ood_benign vs id_benign, same direction, nora_lg activations)")
    print(f"  {'transfer':15s} {'position':13s} {'layer':>5s} {'AUROC_novelty':>20s}")
    for t in ("parent->vla", "vla->vla"):
        for p in POS:
            li = best[(t, p)]; m, s = mean_std(nov[(t, li, p)])
            print(f"  {t:15s} {p:13s} {li:5d} {m:13.3f}+-{s:5.3f}")

    lb = best[("parent->vla", "t_post_inst")]
    print(f"\nper-category held-out AUROC, parent->vla, L{lb} t_post_inst (vs all held-out benign)")
    print(f"  {'category':20s} {'AUROC':>16s} {'n_harm':>7s}")
    for c in cats:
        m, s = mean_std(percat[(c, lb)])
        print(f"  {c:20s} {m:9.3f}+-{s:5.3f} {int((cat[harm] == c).sum()):7d}")

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    out = []

    def add(transfer, li, p, metric, vals, category=""):
        """vals is a list over splits (mean +- std) or a single number computed once, e.g. the cosine."""
        m, s = mean_std(vals) if isinstance(vals, list) else (float(vals), 0.0)
        out.append({"exp": "e1_drobot_transfer", "ts": stamp, "transfer": transfer, "layer": li, "position": p,
                    "metric": metric, "value": m, "std": s,
                    "n_splits": len(vals) if isinstance(vals, list) else 0, "category": category})

    for li in layers:
        for p in POS:
            for t in TRANSFERS:
                add(t, li, p, "auroc", auc[(t, li, p)])
            for t in ("parent->vla", "vla->vla"):
                add(t, li, p, "auroc_novelty", nov[(t, li, p)])
            add("parent_vs_vla", li, p, "cosine", cos[(li, p)])
    for c in cats:
        add("parent->vla", lb, "t_post_inst", "auroc", percat[(c, lb)], category=c)

    res = Path(a.results); res.parent.mkdir(parents=True, exist_ok=True)
    with res.open("a") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"\nappended {len(out)} rows to {res}")
    fig = Path(a.fig); fig.parent.mkdir(parents=True, exist_ok=True)
    make_fig(fig, layers, auc, cos, stamp, a.n_splits)


if __name__ == "__main__":
    main()
