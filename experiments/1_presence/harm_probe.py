#!/usr/bin/env python3
"""Latent harm probe on D_robot. Can a cheap read-out of NORA's residual stream flag harmful instructions?

Four detectors, higher score = more harmful: a trained logistic probe, the zero-shot parent D_text direction,
a novelty (Mahalanobis-to-id_benign) baseline that only knows "unfamiliar", and the parent's own refusal flag.
Scored by held-out-category rotation (train on four harm categories, test on the fifth plus held-out benign),
plus a novelty confound check (ood_benign vs id_benign) and a lexical check (harm vs lexical). Rows to results.jsonl.

  python3 experiments/1_presence/harm_probe.py
  python3 experiments/1_presence/harm_probe.py --layers 12 20 --fig figures/fig3_monitor.png
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from refusal.paths import RESULTS_DIR  # noqa: E402
POS = ("t_inst", "t_post_inst")
DETECTORS = ("probe", "parent_dir", "novelty", "parent_guard")
BENIGN = ("id_benign", "ood_benign", "lexical")


def safe_auroc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def probe_score(Xtr, ytr, Xte, seed):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.1, max_iter=2000, random_state=seed).fit(sc.transform(Xtr), ytr)
    return clf.decision_function(sc.transform(Xte))


def novelty_score(X_id, Xte):
    """Mahalanobis-style distance to the id_benign mean, diagonal covariance."""
    mu, var = X_id.mean(0), X_id.var(0) + 1e-3
    return np.sqrt((((Xte - mu) ** 2) / var).sum(1))


def make_fig(path, li, pos, full, y, cats, res, latest_ts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for det in DETECTORS:
        fpr, tpr, _ = roc_curve(y, full[det])
        ax1.plot(fpr, tpr, lw=1.4, label=f"{det} (AUROC {res[(det, li, pos, 'all')][0]:.3f})")
    ax1.plot([0, 1], [0, 1], color="grey", lw=0.5)
    ax1.set_xlabel("false positive rate"); ax1.set_ylabel("true positive rate")
    ax1.set_title(f"harm vs benign, L{li} {pos} (5-fold out-of-fold)")
    ax1.legend(loc="lower right", fontsize=8)

    x, w = np.arange(len(cats)), 0.8 / len(DETECTORS)
    for j, det in enumerate(DETECTORS):
        ax2.bar(x + j * w - 0.4 + w / 2, [res[(det, li, pos, k)][0] for k in cats], w, label=det)
    ax2.axhline(0.5, color="grey", lw=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(cats, rotation=20, ha="right", fontsize=8)
    ax2.set_ylim(0, 1.28); ax2.set_ylabel("held-out-category AUROC")
    ax2.set_title("held-out harm category"); ax2.legend(fontsize=7, ncol=4, loc="upper center")

    fig.suptitle(f"E4 latent harm monitor, D_robot ({latest_ts})")
    fig.tight_layout(); fig.savefig(path, dpi=150)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(RESULTS_DIR / "drobot_prompts.jsonl"))
    ap.add_argument("--acts", default=str(RESULTS_DIR / "acts/drobot_nora_lg.npz"))
    ap.add_argument("--dirs", default=str(RESULTS_DIR / "acts/parent_dtext_dirs.npz"))
    ap.add_argument("--layers", nargs="+", type=int, default=[12, 16, 20, 21, 24, 28])
    ap.add_argument("--out", default=str(RESULTS_DIR / "results.jsonl"))
    ap.add_argument("--fig", default=str(ROOT / "figures/fig3_monitor.png"), help='empty string disables plotting')
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    prompts = [json.loads(l) for l in Path(a.prompts).read_text().splitlines() if l.strip()]
    acts = np.load(a.acts)
    dirs = np.load(a.dirs)
    ids = [str(i) for i in acts["ids"]]
    if ids != [r["id"] for r in prompts]:
        raise SystemExit(f"activation ids do not match {a.prompts} row order (n={len(ids)} vs {len(prompts)})")

    cond = np.array([r["condition"] for r in prompts])
    cat = np.array([r["category"] for r in prompts])
    # parent_guard = the parent's 0-10 harm rating from parent_harm_judge.py when present (None -> 5), else the old refusal flag
    guard = np.array([float(r["parent_judge_score"]) if r.get("parent_judge_score") is not None
                      else (5.0 if "parent_judge_score" in r else float(bool(r["parent_refusal"]))) for r in prompts])
    y = (cond == "harm").astype(int)
    id_mask = cond == "id_benign"
    cats = sorted(set(cat[y == 1]))
    if set(cond) - {"harm", *BENIGN} or not cats:
        raise SystemExit(f"unexpected conditions {sorted(set(cond))} / harm categories {cats}")
    print(f"{len(prompts)} rows: {sum(y)} harm ({len(cats)} categories), {len(y)-sum(y)} benign; "
          f"parent refused {int(guard.sum())}")

    # fixed benign holdout, identical for every held-out harm category
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(np.where(y == 0)[0])
    n_hold = int(round(0.3 * len(perm)))
    ben_te, ben_tr = perm[:n_hold], perm[n_hold:]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=a.seed).split(np.zeros(len(y)), y))

    res, fulls = {}, {}  # (detector, layer, position, split) -> (value, n_pos, n_neg)
    for li in a.layers:
        for pos in POS:
            X = acts[f"L{li}_{pos}"].astype(np.float64)
            d = dirs[("r_L" if pos == "t_post_inst" else "h_L") + str(li)].astype(np.float64)
            proj = X @ (d / (np.linalg.norm(d) + 1e-8))

            # full-set scores: out-of-fold for the trained detectors, direct for the zero-shot ones
            oof_p, oof_n = np.zeros(len(y)), np.zeros(len(y))
            for tr, te in folds:
                oof_p[te] = probe_score(X[tr], y[tr], X[te], a.seed)
                oof_n[te] = novelty_score(X[tr][id_mask[tr]], X[te])
            full = {"probe": oof_p, "novelty": oof_n, "parent_dir": proj, "parent_guard": guard}
            fulls[(li, pos)] = full

            for name, m, yy in (("all", np.ones(len(y), bool), y),
                                ("novelty_id_vs_ood", np.isin(cond, ("id_benign", "ood_benign")), (cond == "ood_benign").astype(int)),
                                ("harm_vs_lexical", np.isin(cond, ("harm", "lexical")), y)):
                for det in DETECTORS:
                    res[(det, li, pos, name)] = (safe_auroc(yy[m], full[det][m]), int(yy[m].sum()), int((yy[m] == 0).sum()))

            for k in cats:
                te = np.concatenate([np.where((y == 1) & (cat == k))[0], ben_te])
                tr = np.concatenate([np.where((y == 1) & (cat != k))[0], ben_tr])
                s = {"probe": probe_score(X[tr], y[tr], X[te], a.seed),
                     "novelty": novelty_score(X[tr][id_mask[tr]], X[te]),
                     "parent_dir": proj[te], "parent_guard": guard[te]}
                for det in DETECTORS:
                    res[(det, li, pos, k)] = (safe_auroc(y[te], s[det]), int(y[te].sum()), int((y[te] == 0).sum()))
        print(f"  layer {li} done", flush=True)

    def mean_cat(det, li, pos):
        return float(np.nanmean([res[(det, li, pos, k)][0] for k in cats]))

    best_li, best_pos = max(((li, p) for li in a.layers for p in POS), key=lambda lp: mean_cat("probe", *lp))

    hdr = "".join(f"{k[:13]:>15s}" for k in cats)
    print(f"\nselected layer {best_li} / {best_pos} (best mean held-out-category probe AUROC)")
    print(f"{'detector':<14s}{hdr}{'mean':>9s}{'all':>9s}{'ood/id':>9s}{'harm/lex':>10s}")
    for det in DETECTORS:
        cells = "".join(f"{res[(det, best_li, best_pos, k)][0]:15.3f}" for k in cats)
        print(f"{det:<14s}{cells}{mean_cat(det, best_li, best_pos):9.3f}"
              f"{res[(det, best_li, best_pos, 'all')][0]:9.3f}"
              f"{res[(det, best_li, best_pos, 'novelty_id_vs_ood')][0]:9.3f}"
              f"{res[(det, best_li, best_pos, 'harm_vs_lexical')][0]:10.3f}")

    print("\nmean held-out-category AUROC, all layers/positions")
    print(f"{'layer/pos':<20s}" + "".join(f"{d:>14s}" for d in DETECTORS))
    for li in a.layers:
        for pos in POS:
            lab = f"L{li}/{pos}" + (" *" if (li, pos) == (best_li, best_pos) else "")
            print(f"{lab:<20s}" + "".join(f"{mean_cat(d, li, pos):14.3f}" for d in DETECTORS))

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for (det, li, pos, split), (v, npos, nneg) in res.items():
            f.write(json.dumps({"exp": "e4_probe", "ts": stamp, "detector": det, "layer": li, "position": pos,
                                "split": split, "metric": "auroc", "value": v, "n_pos": npos, "n_neg": nneg}) + "\n")
    print(f"\nwrote {len(res)} rows to {out}")

    if a.fig:
        Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
        make_fig(a.fig, best_li, best_pos, fulls[(best_li, best_pos)], y, cats, res, stamp)


if __name__ == "__main__":
    main()
