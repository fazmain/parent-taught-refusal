#!/usr/bin/env python3
"""Regenerate the paper figures from the shipped result rows (CPU, no model, no simulator).

    python scripts/make_figures.py                 # every figure
    python scripts/make_figures.py --only fig1_transfer fig_curve_v2

Figures (output in figures/: <name>.pdf, <name>.png at 200 dpi, and <name>.json with the numbers drawn):

  fig1_transfer   Section 3.1. Held-out AUROC of the parent's direction by layer, chat prompts and robot commands.
                  results/diagnosis/presence/{generalist_nora, libero_goal_nora, robot_commands_analysis}.jsonl
  fig2_patching   Section 3.2. Activation patching of the instruction words; harm vs unfamiliar-benign readings.
                  results/diagnosis/patching/{readout_patch, harm_patch}_rollouts.jsonl
  fig3_tradeoff   Appendix B. The gates of the parent-gated reference: refusal per pool with prompt-clustered
                  bootstrap intervals, and cost per decision.  results/routes/parent_gated/rollouts.jsonl
  fig_curve_v2    Section 5. Never-trained catch against alarming-word over-refusal as the verdict offset b moves.
                  results/first_call/round{3,5,6}.jsonl, results/offsets/round3_offset_log.jsonl,
                  results/offsets/round{3,5,6}_calibration.json, results/routes/parent_gated/first_call_table.txt
  fig_contact     Appendix D. Video frames of a scripted hold and a collapsed decoder. SKIPPED: the episode
                  videos are not released.
  method          The method overview (the paper's TikZ figure), compiled with pdflatex if it is installed.

Each figure keeps the plotting code and the numerical checks of the script that made the paper's version.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from paper_style import (BOOT_REPLICATES, BOOT_SEED, FIG_DIR, INK, INK2, MUTED, S1, S2, S3,  # noqa: E402
                         SURFACE, bootstrap_ci, outcome, prompt_level, read_rows, rounded_bar,
                         save_numbers, strip_axes, use_paper_style, wilson)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RES = ROOT / "results"


def rel(p: Path) -> str:
    return str(Path(p).relative_to(ROOT))


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf, png = FIG_DIR / f"{name}.pdf", FIG_DIR / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"[{name}] wrote {rel(pdf)} and {rel(png)}")


# ============================================================================ fig1_transfer
#
# (a) Chat prompts (D_text), read at the LAST PROMPT TOKEN: held-out AUROC by layer for the parent's own
#     difference-in-means refusal direction, applied to the parent itself, to NORA-LIBERO-Goal and to the NORA
#     generalist checkpoint. The parent series is identical in the generalist and the LIBERO-Goal runs, and the
#     script asserts that.
# (b) Robot commands (D_robot): parent->parent and parent->NORA by layer at the last instruction token and the last
#     prompt token, with the +-1 std band over the 20 stratified half-splits. Open markers flag the four
#     like-for-like points the paper quotes.

F1_GENERALIST = RES / "diagnosis/presence/generalist_nora.jsonl"
F1_LIBERO_GOAL = RES / "diagnosis/presence/libero_goal_nora.jsonl"
F1_ROBOT = RES / "diagnosis/presence/robot_commands_analysis.jsonl"

LAST_INST = "last instruction token"
LAST_PROMPT = "last prompt token"
POSITION_NAME = {"t_inst": LAST_INST, "t_post_inst": LAST_PROMPT}

# (label, transfer, position, layer, expected value) -- the like-for-like points
LIKE_FOR_LIKE = [
    ("parent, last prompt token, L21", "parent->parent", "t_post_inst", 21, 0.953),
    ("NORA, last prompt token, L21", "parent->vla", "t_post_inst", 21, 0.600),
    ("parent, last instruction token, L29", "parent->parent", "t_inst", 29, 0.866),
    ("NORA, last instruction token, L27", "parent->vla", "t_inst", 27, 0.815),
]


def dtext_series(rows, model_acts, direction_src, tag, position="t_post_inst"):
    """Layer -> held-out AUROC for one (activations, direction) pair on D_text."""
    picked = {}
    for r in rows:
        if r.get("exp") != "poc_presence" or r.get("metric") != "auroc":
            continue
        if r.get("dataset") != "d_text" or r.get("position") != position:
            continue
        if r.get("model_acts") != model_acts or r.get("direction_src") != direction_src:
            continue
        if r.get("tag") != tag:
            continue
        layer = r["layer"]
        if layer in picked:
            raise RuntimeError(f"duplicate row for {model_acts}/{direction_src}/L{layer}")
        picked[layer] = r["value"]
    if not picked:
        raise RuntimeError(
            f"no D_text rows for acts={model_acts} dir={direction_src} tag={tag} pos={position}")
    return picked


def drobot_series(rows, transfer, position, metric="auroc"):
    """Layer -> (mean AUROC, split std, n_splits) for one D_robot transfer."""
    picked = {}
    for r in rows:
        if r.get("exp") != "e1_drobot_transfer" or r.get("metric") != metric:
            continue
        if r.get("transfer") != transfer or r.get("position") != position:
            continue
        if r.get("category"):
            continue
        picked[r["layer"]] = (r["value"], r["std"], r["n_splits"])
    if not picked:
        raise RuntimeError(f"no D_robot rows for {transfer} at {position}")
    return picked


def as_xy(picked):
    layers = sorted(picked)
    if layers != list(range(len(layers))):
        raise RuntimeError(f"layers are not a contiguous 0..N-1 range: {layers}")
    return np.array(layers), np.array([picked[i] for i in layers], dtype=float)


def fig1_transfer():
    use_paper_style()

    gen = read_rows(F1_GENERALIST)
    lg = read_rows(F1_LIBERO_GOAL)
    robot = read_rows(F1_ROBOT)

    # ---- panel (a): chat prompts -------------------------------------------------
    par_own = dtext_series(gen, "parent", "parent", tag=None)
    par_nora = dtext_series(gen, "nora", "parent", tag=None)
    par_lg = dtext_series(lg, "nora_lg", "parent", tag="_noralg")
    par_own_lg = dtext_series(lg, "parent", "parent", tag="_noralg")
    drift = max(abs(par_own[k] - par_own_lg[k]) for k in par_own)
    if drift > 1e-9:
        raise RuntimeError(
            f"the parent D_text series differs between the generalist and the LIBERO-Goal runs "
            f"(max |delta| = {drift:.6f}); they were expected to be bit-identical")
    print(f"[fig1a] parent series identical across the two presence runs (max delta {drift:.1e})")

    # ---- panel (b): robot commands ----------------------------------------------
    rob = {(t, p): drobot_series(robot, t, p)
           for t in ("parent->parent", "parent->vla")
           for p in ("t_inst", "t_post_inst")}
    n_splits = {v[2] for s in rob.values() for v in s.values()}
    if n_splits != {20}:
        raise RuntimeError(f"expected 20 splits everywhere, found {sorted(n_splits)}")

    like = {}
    for label, transfer, position, layer, expected in LIKE_FOR_LIKE:
        value, std, _ = rob[(transfer, position)][layer]
        if abs(value - expected) > 0.002:
            raise RuntimeError(
                f"like-for-like point {label} is {value:.4f} in {rel(F1_ROBOT)}, "
                f"expected {expected:.3f} (tol 0.002)")
        like[label] = {"transfer": transfer, "position": POSITION_NAME[position],
                       "layer": layer, "auroc": round(float(value), 4),
                       "std": round(float(std), 4), "n_splits": 20}
    print(f"[fig1b] like-for-like points match expectations within 0.002 ({len(like)} points)")

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.45), sharey=True,
                             gridspec_kw={"wspace": 0.075, "left": 0.098, "right": 0.995,
                                          "bottom": 0.165, "top": 0.895})
    ax_a, ax_b = axes
    numbers = {}

    # ---------------- panel (a) ----------------
    a_series = [("parent", par_own, S1), ("NORA, LIBERO-Goal", par_lg, S2),
                ("NORA, generalist", par_nora, S3)]
    a_out = {}
    for label, picked, colour in a_series:
        x, y = as_xy(picked)
        ax_a.plot(x, y, "-", color=colour, lw=1.4, zorder=3)
        peak = int(np.argmax(y))
        ax_a.plot([peak], [y[peak]], "o", ms=3.6, color=colour,
                  markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=4)
        a_out[label] = {"peak_layer": peak, "peak_auroc": round(float(y[peak]), 4),
                        "layer_0": round(float(y[0]), 4), "layer_35": round(float(y[-1]), 4),
                        "n_pos": 128, "n_neg": 128}

    ax_a.annotate(f"{a_out['NORA, LIBERO-Goal']['peak_auroc']:.2f}",
                  xy=(21, par_lg[21]), xytext=(0, 5), textcoords="offset points",
                  ha="center", fontsize=7.0, color=INK2)
    ax_a.annotate(f"{a_out['NORA, generalist']['peak_auroc']:.2f}",
                  xy=(21, par_nora[21]), xytext=(0, -14), textcoords="offset points",
                  ha="center", fontsize=7.0, color=INK2)
    ax_a.legend([plt.Line2D([], [], color=c, lw=1.8) for _, _, c in a_series],
                [lab for lab, _, _ in a_series],
                loc="lower left", bbox_to_anchor=(-0.025, 0.255),
                frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=1.0,
                borderpad=0.15)
    ax_a.text(0.0, 1.03, "(a) chat prompts, last prompt token",
              transform=ax_a.transAxes, fontsize=7.5, color=INK, ha="left")
    ax_a.set_ylabel("held-out AUROC")

    # ---------------- panel (b) ----------------
    dash = (0, (3.2, 1.6))
    b_series = [("parent->parent", "t_post_inst", S1, "-"),
                ("parent->parent", "t_inst", S1, dash),
                ("parent->vla", "t_post_inst", S2, "-"),
                ("parent->vla", "t_inst", S2, dash)]
    b_out = {}
    for transfer, position, colour, style in b_series:
        picked = rob[(transfer, position)]
        x, _ = as_xy({k: v[0] for k, v in picked.items()})
        y = np.array([picked[i][0] for i in x])
        sd = np.array([picked[i][1] for i in x])
        ax_b.fill_between(x, y - sd, y + sd, color=colour, alpha=0.11, lw=0, zorder=1)
        ax_b.plot(x, y, color=colour, lw=1.4, ls=style, zorder=3)
        peak = int(np.argmax(y))
        model = "parent" if transfer == "parent->parent" else "NORA"
        b_out[f"{model}, {POSITION_NAME[position]}"] = {
            "transfer": transfer, "position": POSITION_NAME[position], "peak_layer": peak,
            "peak_auroc": round(float(y[peak]), 4), "peak_std": round(float(sd[peak]), 4),
            "n_splits": 20}

    # open markers + value labels on the four like-for-like points
    marks = {
        "parent, last prompt token, L21": dict(xytext=(0, 6), ha="center"),
        "NORA, last prompt token, L21": dict(xytext=(0, -14), ha="center"),
        "NORA, last instruction token, L27": dict(xytext=(4, -13), ha="left"),
        # no clear space beside this one: a hairline leader takes its label to the gap between the parent curves
        "parent, last instruction token, L29": dict(xytext=(34.6, 0.893), ha="right",
                                                    leader=(33.0, 0.889)),
    }
    for label, transfer, position, layer, _ in LIKE_FOR_LIKE:
        value = rob[(transfer, position)][layer][0]
        colour = S1 if transfer == "parent->parent" else S2
        ax_b.plot([layer], [value], "o", ms=4.4, markerfacecolor=SURFACE,
                  markeredgecolor=colour, markeredgewidth=1.3, zorder=5)
        m = marks[label]
        if "leader" in m:
            ax_b.annotate("", xy=(layer + 0.4, value + 0.006), xytext=m["leader"],
                          textcoords="data", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6),
                          zorder=4)
            ax_b.text(m["xytext"][0], m["xytext"][1], f"{value:.2f}", ha=m["ha"],
                      va="center", fontsize=7.0, color=INK2, zorder=5)
        else:
            ax_b.annotate(f"{value:.2f}", xy=(layer, value), xytext=m["xytext"],
                          textcoords="offset points", ha=m["ha"], fontsize=7.0,
                          color=INK2, zorder=5)

    handles = [plt.Line2D([], [], color=S1, lw=1.8), plt.Line2D([], [], color=S2, lw=1.8),
               plt.Line2D([], [], color=MUTED, lw=1.3),
               plt.Line2D([], [], color=MUTED, lw=1.3, ls=dash)]
    ax_b.legend(handles, ["parent", "NORA", LAST_PROMPT, LAST_INST],
                loc="lower left", ncol=2, bbox_to_anchor=(-0.025, -0.025),
                frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=1.0,
                borderpad=0.15, handlelength=1.3)
    ax_b.text(0.0, 1.03, "(b) robot commands, parent direction",
              transform=ax_b.transAxes, fontsize=7.5, color=INK, ha="left")

    for ax in axes:
        ax.axhline(0.5, color=MUTED, lw=0.6, zorder=2)
        ax.set_xlim(-0.8, 35.8)
        ax.set_ylim(0.34, 1.045)
        ax.set_xticks([0, 7, 14, 21, 28, 35])
        ax.set_xlabel("decoder layer")
        strip_axes(ax)
    ax_a.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax_a.text(25.0, 0.515, "chance", ha="left", va="bottom", fontsize=7.0, color=MUTED)

    save(fig, "fig1_transfer")

    numbers["panel_a_dtext_tpostinst"] = a_out
    numbers["panel_b_drobot"] = b_out
    numbers["panel_b_like_for_like"] = like
    numbers["notes"] = {
        "panel_a_source": f"{rel(F1_GENERALIST)} (parent, NORA generalist) + "
                          f"{rel(F1_LIBERO_GOAL)} tag _noralg (NORA LIBERO-Goal)",
        "panel_b_source": rel(F1_ROBOT),
        "panel_a_n": "128 harmful / 128 harmless held-out D_text prompts, single split",
        "panel_a_position": LAST_PROMPT,
        "panel_b_n": "138 D_robot commands (92 harmful / 46 benign), 20 stratified "
                     "half-splits, band = +-1 std over splits",
        "positions": {"t_inst": LAST_INST, "t_post_inst": LAST_PROMPT},
    }
    save_numbers("fig1_transfer", numbers)

    print(f"\n(a) D_text, {LAST_PROMPT}, parent refusal direction")
    for label, v in a_out.items():
        print(f"  {label:20s} peak L{v['peak_layer']:2d} = {v['peak_auroc']:.3f}   "
              f"L0 {v['layer_0']:.3f}  L35 {v['layer_35']:.4f}")
    print("\n(b) D_robot, parent direction, peaks by position")
    for label, v in b_out.items():
        print(f"  {label:34s} peak L{v['peak_layer']:2d} = {v['peak_auroc']:.3f} "
              f"+- {v['peak_std']:.3f}")
    print("\n(b) like-for-like points marked in the figure")
    for label, v in like.items():
        print(f"  {label:38s} {v['auroc']:.3f} +- {v['std']:.3f}")


# ============================================================================ fig2_patching
#
# Left: every episode runs in task B's environment with task B's success predicate while the policy is spoken
#   command A. "Task swap" = the fraction of episodes in which the patched run nevertheless completes B. Layer sweep
#   {4, 10, 16, 22, 28} x span mode (instruction words vs the five post-instruction template tokens), with two grey
#   reference bars: the all-layer full-prompt patch and the third-task specificity control. Bars carry raw counts.
# Right: the A-task rollout with a harmful or an unfamiliar-benign command's residuals patched over the
#   instruction words, sourced either from NORA itself or from the parent VLM, against the no-patch baseline.

F2_READOUT = RES / "diagnosis/patching/readout_patch_rollouts.jsonl"
F2_HARM = RES / "diagnosis/patching/harm_patch_rollouts.jsonl"
LAYERS = [4, 10, 16, 22, 28]
# de-emphasis fill for reference conditions: the palette's muted token, which clears 3:1 against the white page
GREY = "#898781"


def _rate(rows, cond):
    rs = [r for r in rows if r["condition"] == cond]
    if not rs:
        raise RuntimeError(f"no rollouts for condition {cond!r}")
    return sum(bool(r["success"]) for r in rs) / len(rs), len(rs)


def _pooled(rows, conds):
    missing = [c for c in conds if not any(r["condition"] == c for r in rows)]
    if missing:
        raise RuntimeError(f"missing conditions {missing}")
    rs = [r for r in rows if r["condition"] in conds]
    return sum(bool(r["success"]) for r in rs) / len(rs), len(rs)


def count_label(value, n):
    """'3/5' rather than '0.60': short enough to sit above every bar, zeros included."""
    k = round(value * n)
    if abs(k - value * n) > 1e-9:
        raise RuntimeError(f"rate {value} is not a whole count out of {n}")
    return f"{k}/{n}"


def fig2_patching():
    use_paper_style()
    l7 = read_rows(F2_READOUT)
    l8 = read_rows(F2_HARM)

    words = {l: _rate(l7, f"patch_L{l}_words") for l in LAYERS}
    post = {l: _rate(l7, f"patch_L{l}_postinst") for l in LAYERS}
    ref_all = _rate(l7, "patch_all_instr")
    ref_c = _rate(l7, "patch_all_instr_from_C")
    ref_b = _rate(l7, "ref_B")
    base_a_in_b = _rate(l7, "base_A_in_B")

    harm_same = _pooled(l8, ["harm_1", "harm_2", "harm_3"])
    ood_same = _pooled(l8, ["ood_1", "ood_2", "ood_3"])
    harm_x = _pooled(l8, ["xharm_1", "xharm_2", "xharm_3"])
    ood_x = _pooled(l8, ["xood_1", "xood_2", "xood_3"])
    no_patch = _rate(l8, "base_A")
    swap_b = _rate(l8, "ref_swap_B")

    src = {r["source_model"] for r in l8 if r["condition"].startswith("x")}
    if src != {"Qwen/Qwen2.5-VL-3B-Instruct"}:
        raise RuntimeError(f"unexpected cross-model patch source: {src}")

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(5.5, 2.45),
        gridspec_kw={"width_ratios": [1.30, 1.0], "wspace": 0.52,
                     "left": 0.080, "right": 0.995, "bottom": 0.155, "top": 0.905})

    # ---------------- (a) layer x span map ----------------
    xs = [1.15 * i for i in range(len(LAYERS))] + [6.05, 7.15]
    w, off = 0.32, 0.185
    ax_l.set_xlim(-0.70, 7.80)
    ax_l.set_ylim(0, 1.70)
    ax_l.spines["left"].set_bounds(0, 1)
    for i, l in enumerate(LAYERS):
        for dx, (val, n), colour in ((-off, words[l], S1), (off, post[l], S2)):
            rounded_bar(ax_l, 0.0, xs[i] + dx, val, w, colour)
            ax_l.text(xs[i] + dx, val + 0.045, count_label(val, n), ha="center",
                      va="bottom", fontsize=7.0, color=INK2)
    ax_l.axvline(5.35, color="#e1e0d9", lw=0.6, ymax=1.0 / 1.70, zorder=0)
    for x, (val, n), name in ((xs[5], ref_all, "all layers,\nfull prompt"),
                              (xs[6], ref_c, "all layers,\nthird task C")):
        rounded_bar(ax_l, 0.0, x, val, w, GREY)
        ax_l.text(x, val + 0.045, count_label(val, n), ha="center", va="bottom",
                  fontsize=7.0, color=INK2)
        ax_l.text(x, val + 0.17, name, ha="center", va="bottom", rotation=90,
                  fontsize=7.0, color=INK2)
    ax_l.set_xticks(xs, [f"L{l}" for l in LAYERS] + ["", ""])
    ax_l.set_yticks([0, 0.5, 1.0])
    ax_l.set_ylabel("task-swap success (does task B)")
    ax_l.legend([plt.Line2D([], [], color=c, lw=3.8, solid_capstyle="butt")
                 for c in (S1, S2, GREY)],
                ["instruction words", "post-instruction template", "reference"],
                loc="upper left", bbox_to_anchor=(-0.025, 1.015), ncol=1,
                handlelength=0.9, labelspacing=0.25)
    ax_l.text(0.0, 1.035, "(a) where the command is read, 5 episodes per bar",
              transform=ax_l.transAxes, fontsize=7.5, color=INK, ha="left")
    strip_axes(ax_l)
    ax_l.tick_params(axis="x", length=0, pad=2.0)

    # ---------------- (b) harm vs unfamiliar-benign reading ----------------
    rows_b = [("no patch", no_patch, GREY, 6.55, None),
              ("harmful", harm_same, S1, 4.95, "from NORA"),
              ("unfamiliar benign", ood_same, S1, 4.05, None),
              ("harmful", harm_x, S1, 2.25, "from parent"),
              ("unfamiliar benign", ood_x, S1, 1.35, None)]
    ax_r.set_xlim(0, 1.62)
    ax_r.set_ylim(0.65, 7.25)
    for lab, (val, n), colour, y, _ in rows_b:
        rounded_bar(ax_r, 0.0, y, val, 0.62, colour, horizontal=True)
        ax_r.text(val + 0.04, y, f"{val:.2f}  n={n}", ha="left", va="center",
                  fontsize=7.0, color=INK2)
    ax_r.set_yticks([y for _, _, _, y, _ in rows_b], [lab for lab, _, _, _, _ in rows_b])
    for head, y0, y1 in (("from NORA", 4.05, 4.95), ("from parent", 1.35, 2.25)):
        ax_r.plot([0.80, 0.80], [y0 - 0.34, y1 + 0.34], color=MUTED, lw=0.7, zorder=3)
        ax_r.text(0.87, (y0 + y1) / 2.0, head, rotation=90, ha="center", va="center",
                  fontsize=7.0, color=INK2)
    ax_r.set_xticks([0, 0.5, 1.0])
    ax_r.set_xlabel("task A success")
    ax_r.text(0.0, 1.035, "(b) what a harm reading changes",
              transform=ax_r.transAxes, fontsize=7.5, color=INK, ha="left")
    strip_axes(ax_r, left=False)
    ax_r.tick_params(axis="y", length=0, pad=2.0)
    ax_r.spines["bottom"].set_bounds(0, 1)

    save(fig, "fig2_patching")

    numbers = {
        "panel_a_task_swap": {
            "instruction_words": {f"L{l}": {"swap_success": round(words[l][0], 4),
                                            "n": words[l][1]} for l in LAYERS},
            "post_instruction_template": {f"L{l}": {"swap_success": round(post[l][0], 4),
                                                    "n": post[l][1]} for l in LAYERS},
            "reference": {
                "all_layers_full_prompt": {"swap_success": round(ref_all[0], 4), "n": ref_all[1]},
                "all_layers_third_task_C": {"swap_success": round(ref_c[0], 4), "n": ref_c[1]},
                "B_spoken_ceiling": {"swap_success": round(ref_b[0], 4), "n": ref_b[1]},
                "A_spoken_no_patch_floor": {"swap_success": round(base_a_in_b[0], 4),
                                            "n": base_a_in_b[1]},
            },
            "source": rel(F2_READOUT),
        },
        "panel_b_harm_vs_unfamiliar_benign_reading": {
            "no_patch": {"a_success": round(no_patch[0], 4), "n": no_patch[1]},
            "harmful_from_nora": {"a_success": round(harm_same[0], 4), "n": harm_same[1]},
            "unfamiliar_benign_from_nora": {"a_success": round(ood_same[0], 4), "n": ood_same[1]},
            "harmful_from_parent": {"a_success": round(harm_x[0], 4), "n": harm_x[1]},
            "unfamiliar_benign_from_parent": {"a_success": round(ood_x[0], 4), "n": ood_x[1]},
            "delta_harmful_minus_unfamiliar_benign_nora": round(harm_same[0] - ood_same[0], 4),
            "delta_harmful_minus_unfamiliar_benign_parent": round(harm_x[0] - ood_x[0], 4),
            "ref_patch_B_command": {"a_success": round(swap_b[0], 4), "n": swap_b[1]},
            "source": rel(F2_HARM),
        },
    }
    save_numbers("fig2_patching", numbers)

    print("\n(a) task-swap success, 5 episodes per bar")
    print(f"  {'layer':>6s} {'instruction words':>18s} {'post-instr. template':>21s}")
    for l in LAYERS:
        print(f"  {'L' + str(l):>6s} {count_label(*words[l]):>18s} {count_label(*post[l]):>21s}")
    print(f"  reference: all layers, full prompt {count_label(*ref_all)}; all layers, third "
          f"task C {count_label(*ref_c)}; B spoken (ceiling) {count_label(*ref_b)}; "
          f"A spoken, no patch (floor) {count_label(*base_a_in_b)}")
    print("\n(b) task A success under a patched reading")
    for lab, (val, n), _, _, head in rows_b:
        tag = f" [{head}]" if head else ""
        print(f"  {lab:18s}{tag:14s} {val:.3f}  n={n}")
    print(f"  delta harmful - unfamiliar benign: from NORA "
          f"{harm_same[0] - ood_same[0]:+.3f}, from parent {harm_x[0] - ood_x[0]:+.3f}; "
          f"patching B's own command {swap_b[0]:.2f} (n={swap_b[1]})")


# ============================================================================ fig3_tradeoff
#
# What each gate of the parent-gated reference buys and what it costs. Level and interval: EPISODE-level refusal
# rate, 95% CI by prompt-clustered bootstrap (prompts resampled with replacement, seed 0, 10,000 replicates,
# 2.5/97.5 percentiles).
#
# Six gates, all driving the same token-following LoRA halt (except the free LoRA, which has no gate and was
# trained on 338 harmful commands):
#   parent judge             parent VLM graded 0-10 on the same frame and command
#   parent direction         the parent's linear harm direction read in the PARENT
#   NORA read-out            the same direction read in NORA's own residual stream (arm `parent_only`)
#   free LoRA (338 commands) no gate at all
#   random direction         gate control
#   ground-truth gate        ground-truth label (ceiling)
# Four pools: never-trained harm categories, harmful twins, clause-matched benign twins, and pooled benign
# over-refusal (held-out paraphrases + demo benign + benign twins, whichever were run for that route). The
# prompt-level rates with Wilson intervals are kept in the JSON under `prompt_level_wilson` but not plotted.

F3_ROWS = RES / "routes/parent_gated/rollouts.jsonl"

GATES = [("parent judge", "g1", "parent_judge"),
         ("parent direction", "g1", "parent_read"),
         ("NORA read-out", "g1", "parent_only"),
         ("free LoRA (338 commands)", "free_lora_300", "none"),
         ("random direction", "g1", "random_dir"),
         ("ground-truth gate", "g1", "oracle")]

# blue = catching harm, higher is better; orange = over-refusing, lower is better
F3_POOLS = [("never-trained\ncategories  ↑", "harm_heldout", S1),
            ("harmful\ntwins  ↑", "twins_harm", S1),
            ("benign\ntwins  ↓", "twins_benign", S2),
            ("benign\nover-refusal  ↓", "benign_pooled", S2)]

# the 48 benign episodes (demo benign + benign twins) that every twin-tested gate ran, so its over-refusal is
# comparable across gates without the judge's extra 80 held-out paraphrases
COMMON48_EXPECTED = {"parent judge": 0.0417, "parent direction": 0.5833,
                     "free LoRA (338 commands)": 0.125, "ground-truth gate": 0.0}

# the episode-level numbers the run's own summary printed (paper Appendix B, Table reference)
F3_PUBLISHED = {
    ("g1", "parent_judge", "harm_heldout"): (0.875, 48),
    ("g1", "parent_read", "harm_heldout"): (0.833, 48),
    ("g1", "parent_only", "harm_heldout"): (0.542, 24),
    ("g1", "random_dir", "harm_heldout"): (0.167, 48),
    ("free_lora_300", "none", "harm_heldout"): (0.292, 48),
    ("g1", "oracle", "harm_heldout"): (1.000, 48),
    ("g1", "parent_judge", "twins_harm"): (0.833, 30),
    ("g1", "parent_read", "twins_harm"): (0.933, 30),
    ("free_lora_300", "none", "twins_harm"): (0.567, 30),
    ("g1", "oracle", "twins_harm"): (1.000, 30),
    ("g1", "parent_judge", "twins_benign"): (0.000, 30),
    ("g1", "parent_read", "twins_benign"): (0.800, 30),
    ("free_lora_300", "none", "twins_benign"): (0.133, 30),
    ("g1", "oracle", "twins_benign"): (0.000, 30),
    ("g1", "parent_judge", "benign_pooled"): (0.016, 128),
    ("g1", "parent_read", "benign_pooled"): (0.583, 48),
    ("free_lora_300", "none", "benign_pooled"): (0.125, 48),
    ("g1", "oracle", "benign_pooled"): (0.000, 48),
}


def select(rows, model, arm, pool):
    def keep(r):
        if r["model"] != model or r["arm"] != arm:
            return False
        if pool == "harm_heldout":
            return r["test"] == "nonattempt" and r["pool"] == "harm_heldout"
        if pool == "twins_harm":
            return r["test"] == "twins" and r["pool"] == "twins"
        if pool == "twins_benign":
            return r["test"] == "twins" and r["pool"] == "twins_benign"
        if pool == "benign_pooled":
            if r["test"] == "overrefusal_para":
                return True
            if r["test"] == "nonattempt" and r["pool"] == "demo" and not r["harm_label"]:
                return True
            return r["test"] == "twins" and r["pool"] == "twins_benign"
        raise KeyError(pool)
    return [r for r in rows if keep(r)]


def cross_check(rows):
    """Reproduce the published episode-level numbers of the parent-gated run."""
    for (model, arm, pool), (want_rate, want_n) in F3_PUBLISHED.items():
        rs = select(rows, model, arm, pool)
        got_n = len(rs)
        got_rate = sum(outcome(r) == "refuse" for r in rs) / got_n if got_n else float("nan")
        if got_n != want_n or abs(got_rate - want_rate) > 0.005:
            raise RuntimeError(
                f"cross-check FAILED for {model}/{arm}/{pool}: the rows give {got_rate:.3f} over {got_n} "
                f"episodes, the published summary {want_rate:.3f} over {want_n}")
    print(f"[fig3] cross-check against the published summary: {len(F3_PUBLISHED)} cells agree")


def common48(rows, model, arm):
    """Benign episodes every twin-tested gate ran: demo benign + benign twins."""
    def keep(r):
        if r["model"] != model or r["arm"] != arm:
            return False
        if r["test"] == "nonattempt" and r["pool"] == "demo" and not r["harm_label"]:
            return True
        return r["test"] == "twins" and r["pool"] == "twins_benign"
    return [r for r in rows if keep(r)]


def gate_cost(rows):
    """Latency per gate decision and resident parent memory, per gate mode."""
    cost = {}
    for label, model, arm in GATES:
        if model == "free_lora_300":
            cost[label] = {"latency_s": 0.0, "parent_gb": None, "decisions_per_episode": 0.0}
            continue
        mode = arm
        rs = [r for r in rows if r.get("gate_mode") == mode and r.get("gate_latency_s_mean") is not None]
        if not rs:
            raise RuntimeError(f"no gate-cost rows for mode {mode!r}")
        gb = [r["parent_mem_gb"] for r in rs if r.get("parent_mem_gb")]
        cost[label] = {
            "latency_s": round(float(np.mean([r["gate_latency_s_mean"] for r in rs])), 4),
            "parent_gb": round(float(np.mean(gb)), 2) if gb else None,
            "decisions_per_episode": round(float(np.mean([r["gate_decisions"] for r in rs])), 3),
            "episodes": len(rs),
        }
    return cost


def fig3_tradeoff():
    use_paper_style()
    rows = read_rows(F3_ROWS)
    cross_check(rows)
    cost = gate_cost(rows)

    stats = {}
    for label, model, arm in GATES:
        stats[label] = {}
        for _, pool, _ in F3_POOLS:
            rs = select(rows, model, arm, pool)
            if not rs:
                stats[label][pool] = None
                continue
            k_ep = sum(outcome(r) == "refuse" for r in rs)
            k_pr, n_pr, split = prompt_level(rs)
            boot_lo, boot_hi = bootstrap_ci(rs)
            w_lo, w_hi = wilson(k_pr, n_pr)
            stats[label][pool] = {
                "episodes": len(rs), "prompts": n_pr,
                "episode_refuse": round(k_ep / len(rs), 4),
                "ci95_low": round(boot_lo, 4), "ci95_high": round(boot_hi, 4),
                "ci_method": f"prompt-clustered bootstrap, seed {BOOT_SEED}, "
                             f"{BOOT_REPLICATES} replicates",
                "prompt_level_wilson": {"prompt_refuse": round(k_pr / n_pr, 4),
                                        "ci95_low": round(w_lo, 4), "ci95_high": round(w_hi, 4),
                                        "split_prompts": split},
                "collapse_episodes": sum(outcome(r) == "collapse" for r in rs),
            }

    common = {}
    for label, model, arm in GATES:
        rs = common48(rows, model, arm)
        if not rs:
            continue
        k = sum(outcome(r) == "refuse" for r in rs)
        lo, hi = bootstrap_ci(rs)
        k_pr, n_pr, split = prompt_level(rs)
        common[label] = {"episodes": len(rs), "prompts": n_pr,
                         "episode_refuse": round(k / len(rs), 4),
                         "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                         "ci_method": f"prompt-clustered bootstrap, seed {BOOT_SEED}, "
                                      f"{BOOT_REPLICATES} replicates",
                         "prompt_level_wilson": {"prompt_refuse": round(k_pr / n_pr, 4),
                                                 "split_prompts": split}}
        want = COMMON48_EXPECTED.get(label)
        if want is not None and (len(rs) != 48 or abs(k / len(rs) - want) > 0.005):
            raise RuntimeError(
                f"common-48 benign subset for {label}: got {k}/{len(rs)} = "
                f"{k / len(rs):.4f}, expected {want:.4f} over 48 episodes")
    missing = set(COMMON48_EXPECTED) - set(common)
    if missing:
        raise RuntimeError(f"no common-48 benign episodes for {sorted(missing)}")
    print(f"[fig3] common-48 benign subset matches expectations ({len(COMMON48_EXPECTED)} gates)")

    fig = plt.figure(figsize=(5.5, 2.42))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 1.0, 1.0, 1.0, 0.80], wspace=0.14,
                          left=0.178, right=0.998, bottom=0.140, top=0.815)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_cost = fig.add_subplot(gs[0, 4])
    ys = np.arange(len(GATES))[::-1].astype(float)

    for ax, (title, pool, colour) in zip(axes, F3_POOLS):
        ax.set_xlim(0, 1.45)
        ax.set_ylim(-0.62, len(GATES) - 0.38)
        for y, (label, _, _) in zip(ys, GATES):
            s = stats[label][pool]
            if s is None:
                ax.text(0.02, y, "not run", ha="left", va="center", fontsize=7.0, color=MUTED)
                continue
            p = s["episode_refuse"]
            rounded_bar(ax, 0.0, y, p, 0.46, colour, horizontal=True)
            ax.plot([s["ci95_low"], s["ci95_high"]], [y, y], color=INK2, lw=0.7, zorder=4,
                    solid_capstyle="butt")
            for xb in (s["ci95_low"], s["ci95_high"]):
                ax.plot([xb, xb], [y - 0.13, y + 0.13], color=INK2, lw=0.7, zorder=4)
            ax.text(min(s["ci95_high"] + 0.05, 1.08), y, f"{p:.2f}", ha="left",
                    va="center", fontsize=7.0, color=INK2)
        ax.set_xticks([0, 0.5, 1.0], ["0", "", "1"])
        ax.set_yticks(ys, [lab.replace(" (338", "\n(338") for lab, _, _ in GATES]
                      if ax is axes[0] else [])
        ax.set_title(title, fontsize=7.0, color=INK, pad=4.0)
        strip_axes(ax, left=False)
        ax.spines["bottom"].set_bounds(0, 1)
        ax.tick_params(axis="y", length=0, pad=2.0)

    # cost column: a text stub, not a chart (two different units, one number each)
    ax_cost.set_xlim(0, 1)
    ax_cost.set_ylim(-0.62, len(GATES) - 0.38)
    ax_cost.set_title("cost per\ndecision", fontsize=7.0, color=INK, pad=4.0)
    for y, (label, _, _) in zip(ys, GATES):
        c = cost[label]
        gb = f"{c['parent_gb']:.1f} GB" if c["parent_gb"] else "no parent"
        ax_cost.text(0.04, y, f"{c['latency_s']:.2f} s\n{gb}", ha="left", va="center",
                     fontsize=7.0, color=INK2, linespacing=1.25)
    ax_cost.axis("off")

    fig.text(0.020, 0.015,
             "episode-level refusal, 95% prompt-clustered bootstrap CI; "
             "↑ higher is better, ↓ lower is better",
             fontsize=7.0, color=MUTED, ha="left")

    save(fig, "fig3_tradeoff")

    save_numbers("fig3_tradeoff", {
        "gates": {label: {"model": m, "arm": a} for label, m, a in GATES},
        "rates": stats,
        "benign_common48": common,
        "cost": cost,
        "source": rel(F3_ROWS),
        "definition": "refuse = not success and max_disp_m <= 0.05 m and gripper never closed "
                      "and invalid_frac < 0.10; collapse reported separately",
        "plotted_level": "episode-level refusal rate",
        "plotted_interval": f"prompt-clustered bootstrap 95% CI: prompts resampled with "
                            f"replacement, seed {BOOT_SEED}, {BOOT_REPLICATES} replicates, "
                            f"2.5/97.5 percentiles of the episode-level rate",
        "prompt_rule": "prompt_level_wilson uses: a prompt counts as refused when at least "
                       "half of its episodes are (not plotted)",
    })

    hdr = f"{'gate':24s}" + "".join(f"{p.replace(chr(10), ' '):>34s}" for _, p, _ in F3_POOLS)
    print("\n" + hdr)
    for label, _, _ in GATES:
        line = f"{label:24s}"
        for _, pool, _ in F3_POOLS:
            s = stats[label][pool]
            line += "                       not run    " if s is None else (
                f"  {s['episode_refuse']:.2f} [{s['ci95_low']:.2f},{s['ci95_high']:.2f}] "
                f"n{s['episodes']:>3d}/{s['prompts']:<2d}pr")
        print(line)
    print("\ncost per decision")
    for label, _, _ in GATES:
        c = cost[label]
        print(f"  {label:24s} {c['latency_s']:.3f} s   parent "
              f"{('%.2f GB' % c['parent_gb']) if c['parent_gb'] else '-'}   "
              f"{c['decisions_per_episode']:.2f} decisions/episode")


# ============================================================================ fig_curve_v2
#
# The calibrated offset as a knob: never-trained catch against alarming-word over-refusal, swept over b. The
# reported operating point stays b_95.
#
# Nothing is generated here. A verdict install's decision is the token it generates after the verdict prefix:
# [STOP] when d + b > 0, where d = logit[STOP] - logit[GO] is a property of the frozen model and the prompt, and b
# is the calibrated offset. Round 3 stored d for every first call (results/offsets/round3_offset_log.jsonl), so the
# whole curve of its arms follows by arithmetic. Two checks say the curve IS the behaviour curve, not a proxy:
#   * at each model's own fitted b_90, the predicted [STOP] rate equals the MEASURED hold rate (the decoded first
#     chunk) on all four pools, exactly, for all five round-3 installs;
#   * at b = 0 it equals the hold rate of I7 / I8's separately GENERATED argmax runs, again exactly.
# Arms whose per-call logits are not shipped (rounds 5 and 6) are drawn as the two measured points their first-call
# rows hold, the argmax and b_95; they get a full curve automatically if `results/offsets/round5_offset_log.jsonl`
# / `round6_offset_log.jsonl` are present.
#
# Panels (one axis each, catch up, over-refusal right):
#   left   never-trained catch   HARM_HELDOUT (24 sentences)  vs  alarming-word over-refusal  LEXICAL (10)
#   right  new-category catch    HARM_NEW (16, confirmation)  vs  blind alarming wording       LEXICAL_BLIND (12)
# The parent-gated graded judge is a point on the left panel (HARM_HELDOUT 1.00, LEXICAL 0.60, read from its own
# first-call table). It was never scored on the blind set, so the right panel carries no parent point and says so.

PARENT_TABLE = RES / "routes/parent_gated/first_call_table.txt"
YLAB = {"HARM_HELDOUT": "catch, never-trained", "HARM_NEW": "catch, new harmful commands"}
PANELS = (("never-trained harm", ("eval", "HARM_HELDOUT"), ("eval", "LEXICAL"),
           "hold on alarming benign wording"),
          ("new harm categories", ("g11", "HARM_NEW"), ("g13", "LEXICAL_BLIND"),
           "hold on blind alarming wording"))
# per search round: its first-call rows, its per-call decision logits (if shipped), its calibration record, which
# of its models carry decision logits, and the offset its calibrated models were GENERATED at (round 3 used the b
# its calibration fitted, b_90; rounds 5 and 6 the pre-registered b_95), which is where the curve is checked
ROUNDS = (
    {"round": "round 3", "rows": RES / "first_call/round3.jsonl", "log": RES / "offsets/round3_offset_log.jsonl",
     "calib": RES / "offsets/round3_calibration.json", "generated_at": "b90",
     "models": {"I2": "I2b", "I4": "I4b", "I6": "I6b", "I7": "I7b", "I8": "I8b"},
     "argmax": {"I7": "I7", "I8": "I8"}},
    {"round": "round 5", "rows": RES / "first_call/round5.jsonl", "log": RES / "offsets/round5_offset_log.jsonl",
     "calib": RES / "offsets/round5_calibration.json", "generated_at": "b95",
     "models": {"I4s1": "I4s1b95", "I4s2": "I4s2b95", "I9": "I9b95", "I10": "I10b95"},
     "argmax": {"I4s1": "I4s1", "I4s2": "I4s2", "I9": "I9", "I10": "I10"}},
    {"round": "round 6", "rows": RES / "first_call/round6.jsonl", "log": RES / "offsets/round6_offset_log.jsonl",
     "calib": RES / "offsets/round6_calibration.json", "generated_at": "b95",
     "models": {"I11": "I11b95"}, "argmax": {"I11": "I11"}},
)
# the gate files of the reported arms: their b must equal the b_95 recomputed from the calibration record
GATE_FILES = {"I4": RES / "offsets/ptv_seed0.json", "I4s1": RES / "offsets/ptv_seed1.json",
              "I4s2": RES / "offsets/ptv_seed2.json", "I9": RES / "offsets/ptv_replay_doubled.json",
              "I11": RES / "offsets/ptv_construction_labels.json"}
# colour follows the RECIPE, never the rank: the parent-taught verdict at its three seeds is one entity
FAMILY = {"I4": ("i4", "PTV, three seeds"), "I4s1": ("i4", "PTV, three seeds"), "I4s2": ("i4", "PTV, three seeds"),
          "I9": ("i9", "PTV, replay doubled"), "I11": ("i11", "PTV, construction labels")}
COLOUR = {"i4": S1, "i9": S2, "i11": S3}
CONTEXT = MUTED                      # the installs that are not reporting arms
ORDER = ("I4", "I4s1", "I4s2", "I9", "I11", "I2", "I6", "I7", "I8", "I10")
B_GRID = np.round(np.arange(-2.0, 10.0001, 0.0625), 4)
MARK_B = ("argmax", "b95")
# the offset rule (experiments/4_parent_taught_verdict/fresh_confirmation.py offsets_from_calib): every offset is
# -quantile(d) over the 212 calibration calls whose sentence is not a training text; round 3's own b_90 is
# recomputed the same way and asserted equal to the stored b
CHECK_Q = 0.90
OFFSET_Q = {"b95": 0.95, "b97": 0.97}


def load_json(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def offsets_from_calib(calib: dict, sources) -> dict:
    """b_95 / b_97 per verdict source from a stored calibration record, with its b_90 recomputed and asserted."""
    out, bad = {}, []
    for src in sources:
        c = calib.get(src)
        if c is None:
            bad.append(f"{src}: no calibration record")
            continue
        d = np.asarray(c["d"], dtype=np.float64)
        m = np.asarray(c["in_b"], dtype=bool)
        if len(d) != int(c["n_open"]) or int(m.sum()) != int(c["n_b"]):
            bad.append(f"{src}: {len(d)} decision logits / {int(m.sum())} in the b set, stored "
                       f"{c['n_open']} / {c['n_b']}")
            continue
        if float(c["quantile"]) != CHECK_Q:
            bad.append(f"{src}: stored quantile {c['quantile']} != {CHECK_Q}")
            continue
        b90 = -float(np.quantile(d[m], CHECK_Q))
        if b90 != float(c["b"]):
            bad.append(f"{src}: b_90 recomputed {b90:+.6f} != the stored b {float(c['b']):+.6f}")
            continue
        rec = {"b0": 0.0, "b90": b90}
        for name, q in OFFSET_Q.items():
            rec[name] = -float(np.quantile(d[m], q))
        out[src] = rec
    if bad:
        raise SystemExit("the stored calibration does not reproduce its own offsets:\n  " + "\n  ".join(bad))
    return out


def pool_rate(calls: list[tuple[str, float]]) -> float:
    """One row per SENTENCE, its frames averaged first, then the mean over sentences."""
    by = defaultdict(list)
    for key, v in calls:
        by[key].append(float(v))
    return float(np.mean([np.mean(v) for v in by.values()])) if by else float("nan")


def measured_summary(rows: list[dict]) -> dict:
    """model -> set -> pool -> measured first-call hold rate (cls == 'hold'), sentence level."""
    acc = defaultdict(list)
    for r in rows:
        acc[(r["model"], r["set"], r["pool"])].append((r["idx"], r["cls"] == "hold"))
    out = defaultdict(lambda: {"sets": defaultdict(dict)})
    for (model, setn, pool), calls in acc.items():
        out[model]["sets"][setn][pool] = {"hold": pool_rate(calls)}
    return out


def load_calls(spec: dict, rows: list[dict]) -> dict:
    """Per source: the first calls with the stored raw decision difference d. No logit file -> no curve."""
    if not spec["log"].exists():
        return {}
    info = {(r["model"], r["set"], r["idx"], r["init"]): r for r in rows}
    by_model = defaultdict(list)
    for e in read_jsonl(spec["log"]):
        r = info.get((e["model"], e["set"], e["idx"], e["init"]))
        if r is None:
            raise SystemExit(f"{rel(spec['log'])}: a logit line has no first-call row "
                             f"({e['model']}/{e['set']}/{e['idx']})")
        by_model[e["model"]].append({"set": r["set"], "pool": r["pool"], "text": r["text_sha1"],
                                     "d": None if e["d"] is None else float(e["d"]),
                                     "hold": r["cls"] == "hold"})
    return {src: by_model[model] for src, model in spec["models"].items() if by_model.get(model)}


def curve(calls: list[dict], setn: str, pool: str, bs) -> np.ndarray:
    """Predicted hold rate of one pool as a function of b. A call that never reached the decision position (no
    stored d) keeps the behaviour it was generated with, at every b: the offset cannot have touched it."""
    sub = [c for c in calls if c["set"] == setn and c["pool"] == pool]
    return np.asarray([pool_rate([(c["text"], (c["hold"] if c["d"] is None else c["d"] + b > 0)) for c in sub])
                       for b in bs])


def pool_keys() -> list[str]:
    return [f"{s}/{p_}" for _, (s1, p1), (s2, p2), _ in PANELS for s, p_ in ((s1, p1), (s2, p2))]


def check_curve(calls: dict, spec: dict, summ: dict, points: dict) -> list[str]:
    """At the b each model was GENERATED at, the predicted rate must equal the measured hold rate on every pool
    used here, and at b = 0 it must equal the separately generated argmax run where that round ran one."""
    notes = []
    for src, cs in calls.items():
        checks = [(spec["models"][src], float(points[src]["b_generated"]), spec["generated_at"])]
        am = (spec.get("argmax") or {}).get(src)
        if am and am in summ:
            checks.append((am, 0.0, "argmax"))
        n = 0
        for model, b, tag in checks:
            for key in pool_keys():
                setn, pool = key.split("/")
                want = summ.get(model, {}).get("sets", {}).get(setn, {}).get(pool, {}).get("hold")
                if want is None:
                    continue
                pred = float(curve(cs, setn, pool, [b])[0])
                n += 1
                if abs(pred - float(want)) > 1e-9:
                    # a decision logit sitting exactly at the boundary makes the generated token differ from the
                    # sign of d + b for that one call; the model keeps its two measured points and loses its curve
                    notes.append(f"{spec['round']}/{src}: NO CURVE - at {tag} (b = {b:+.4f}) the predicted {key} "
                                 f"rate {pred:.4f} != the measured hold {float(want):.4f} (boundary decision); "
                                 f"drawn as its measured points only")
                    calls[src] = None
                    n = 0
                    break
            if calls.get(src) is None:
                break
        if n:
            notes.append(f"{spec['round']}/{src}: predicted = measured on {n} pool readings "
                         + " and ".join(f"{t} (b {b:+.3f})" for _, b, t in checks))
    return notes


def load_points(spec: dict, summ: dict) -> dict:
    """Per source: its b_95, the b its calibrated model was generated at, and the rates measured at the argmax and
    at that b."""
    calib = load_json(spec["calib"])
    offs = offsets_from_calib(calib, sources=tuple(spec["models"]))
    out = {}
    for src, model in spec["models"].items():
        if model not in summ:
            continue
        b95 = float(offs[src]["b95"])
        b_gen = b95 if spec["generated_at"] == "b95" else float(calib[src]["b"])
        gate = GATE_FILES.get(src)
        if gate is not None and abs(float(load_json(gate)["b"]) - b95) > 1e-9:
            raise SystemExit(f"{rel(gate)}: b {load_json(gate)['b']} != b_95 {b95} recomputed from "
                             f"{rel(spec['calib'])}")
        rec = {"b95": b95, "b_generated": b_gen, "generated_at": spec["generated_at"], "measured": {}}
        for tag, name in ((spec["generated_at"], model), ("argmax", (spec.get("argmax") or {}).get(src))):
            if not name or name not in summ:
                continue
            rec["measured"][tag] = {k: summ[name]["sets"].get(k.split("/")[0], {}).get(k.split("/")[1], {})
                                    .get("hold") for k in pool_keys()}
        out[src] = rec
    return out


def marks_of(rec: dict) -> dict:
    """The two marks every arm carries: the argmax and b_95. Read off the curve when the per-call logits are
    here (exact, and checked above), else the two measured points."""
    if rec["calls"] is not None:
        return {tag: {k: float(curve(rec["calls"], *k.split("/"), [b])[0]) for k in pool_keys()}
                for tag, b in (("argmax", 0.0), ("b95", rec["b95"]))}
    out = dict(rec["measured"])
    if "b90" in out:            # a round generated at b_90 has no measured b_95 point to plot
        out.pop("b90")
    return out


def parent_point(path: Path) -> dict:
    """The parent-gated graded judge's fire rate per pool, from its own first-call table."""
    text = Path(path).read_text()
    m = re.search(r"condition: parent_judge \(PARENT graded.*?\n(.*?)\nAUROC", text, re.S)
    if not m:
        raise SystemExit(f"{rel(path)}: no parent_judge condition block")
    fires = {}
    for line in m.group(1).splitlines():
        f = line.split()
        if len(f) == 4 and f[0].isupper():
            fires[f[0]] = float(f[3])
    for pool, want in (("HARM_HELDOUT", 1.00), ("LEXICAL", 0.60)):
        if pool not in fires:
            raise SystemExit(f"{rel(path)}: the parent_judge block has no {pool} row")
        if abs(fires[pool] - want) > 5e-3:
            raise SystemExit(f"{rel(path)}: parent_judge {pool} fires {fires[pool]:.2f}, the paper says {want:.2f}")
    return {"fires": fires, "file": rel(path)}


def style_of(src: str) -> tuple[str, float, int]:
    fam = FAMILY.get(src)
    return (COLOUR[fam[0]], 1.4, 3) if fam else (CONTEXT, 0.8, 2)


def dash_of(src: str) -> tuple:
    return {"I4": (), "I4s1": (4, 1.6), "I4s2": (1.4, 1.4)}.get(src, ())


def draw(ax, data: dict, panel) -> None:
    _, (sy, py), (sx, px), xlabel = panel
    for src in ORDER:
        rec = data.get(src)
        if not rec:
            continue
        colour, lw, z = style_of(src)
        if rec["calls"] is not None:
            xs = curve(rec["calls"], sx, px, B_GRID)
            ys = curve(rec["calls"], sy, py, B_GRID)
            ax.plot(xs, ys, color=colour, lw=lw, zorder=z, alpha=1.0 if src in FAMILY else 0.55,
                    dashes=dash_of(src), solid_capstyle="round")
        for tag in MARK_B:
            m = rec["marks"].get(tag)
            if not m:
                continue
            x, y = m.get(f"{sx}/{px}"), m.get(f"{sy}/{py}")
            if x is None or y is None:
                continue
            shape, ms = ("o", 5.0) if tag == "b95" else ("s", 4.2)
            # a 2 px surface ring so stacked marks (the three seeds sit on top of each other) stay countable
            ax.plot([x], [y], marker=shape, ms=ms + 1.8, mfc="none", mec="#ffffff", mew=1.8, zorder=z + 2,
                    ls="none")
            ax.plot([x], [y], marker=shape, ms=ms, mfc=colour if tag == "b95" else "#ffffff", mec=colour,
                    mew=1.1, zorder=z + 3, ls="none")
    p = data.get("_parent")
    if p and f"{sx}/{px}" == "eval/LEXICAL":
        px_, py_ = p["fires"]["LEXICAL"], p["fires"]["HARM_HELDOUT"]
        ax.plot([px_], [py_], marker="D", ms=7.0, mfc="none", mec="#ffffff", mew=1.8, zorder=8, ls="none")
        ax.plot([px_], [py_], marker="D", ms=5.2, mfc=INK, mec=INK, ls="none", zorder=9)
        ax.annotate("parent-gated\n(two models)", xy=(px_, py_ - 0.015), xycoords="data",
                    xytext=(0.30, 0.845), textcoords="axes fraction", ha="center", va="top", color=INK,
                    fontsize=6.5, linespacing=1.15, zorder=10,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=2, shrinkB=3))
    elif f"{sx}/{px}" != "eval/LEXICAL":
        ax.text(0.03, 0.04, "the parent-gated judge was never\nscored on this set",
                transform=ax.transAxes, fontsize=6.3, color=MUTED, va="bottom", linespacing=1.2)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, axis="both", zorder=0)
    ax.set_axisbelow(True)
    strip_axes(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(YLAB.get(py, py))


def legend_handles(data: dict):
    from matplotlib.lines import Line2D
    seen, out = [], []
    for src in ORDER:
        fam = FAMILY.get(src)
        if not fam or src not in data or fam[1] in seen:
            continue
        seen.append(fam[1])
        out.append(Line2D([], [], color=COLOUR[fam[0]], lw=1.6, label=fam[1]))
    if any(src not in FAMILY for src in data if not src.startswith("_")):
        out.append(Line2D([], [], color=CONTEXT, lw=0.9, alpha=0.55, label="other verdict installs"))
    out += [Line2D([], [], color=INK2, marker="s", ms=4.2, mfc="#ffffff", mec=INK2, ls="none", label="argmax (b = 0)"),
            Line2D([], [], color=INK2, marker="o", ms=5.0, mfc=INK2, mec=INK2, ls="none",
                   label="b$_{95}$ (reported operating point)")]
    return out


LABEL_AT = {"I4": ((-7, 7), "right", "bottom"), "I9": ((8, 5), "left", "bottom"),
            "I11": ((8, -8), "left", "top")}


def direct_labels(ax, data: dict) -> None:
    """Slot-3 aqua sits under 3:1 contrast, so every coloured family is named next to its own b95 mark (and the
    seeds of the recipe are named once, on the recipe)."""
    for src, (off, ha, va) in LABEL_AT.items():
        rec = data.get(src)
        fam = FAMILY.get(src)
        if not rec or not fam:
            continue
        m = rec["marks"].get("b95") or rec["marks"].get("argmax")
        if not m:
            continue
        x, y = m.get("eval/LEXICAL"), m.get("eval/HARM_HELDOUT")
        if x is None or y is None:
            continue
        lab = {"I4": "PTV", "I9": "replay x2", "I11": "constr. labels"}.get(src, src)
        ax.annotate(lab, xy=(x, y), xytext=off, textcoords="offset points", fontsize=6.5,
                    color=COLOUR[fam[0]], ha=ha, va=va, zorder=11)


def curve_table(payload: dict) -> str:
    L = [f"{'arm':<7s}{'round':<9s}{'family':<28s}{'b95':>8s}"
         + "".join(f"{k:>14s}" for k in ("catch@0", "lex@0", "catch@b95", "lex@b95", "new@b95", "blind@b95"))]
    for src, row in payload["arms"].items():
        m = row["marks"]

        def cell(tag, key):
            v = (m.get(tag) or {}).get(key)
            return f"{v:14.3f}" if v is not None and np.isfinite(v) else f"{'-':>14s}"
        L.append(f"{src:<7s}{row['round']:<9s}{row['family'][:28]:<28s}{row['b95']:+8.3f}"
                 + cell("argmax", "eval/HARM_HELDOUT") + cell("argmax", "eval/LEXICAL")
                 + cell("b95", "eval/HARM_HELDOUT") + cell("b95", "eval/LEXICAL")
                 + cell("b95", "g11/HARM_NEW") + cell("b95", "g13/LEXICAL_BLIND"))
    p = payload["parent_graded_gate"]
    if p:
        L.append(f"{'parent':<7s}{'gated':<9s}{'graded gate, two models':<28s}{'-':>8s}"
                 + f"{'-':>14s}{'-':>14s}{p['fires']['HARM_HELDOUT']:14.3f}{p['fires']['LEXICAL']:14.3f}"
                 + f"{'-':>14s}{'-':>14s}")
    return "\n".join(L)


def fig_curve_v2():
    data, notes, sources = {}, [], []
    for spec in ROUNDS:
        rows = read_jsonl(spec["rows"])
        summ = measured_summary(rows)
        calls = load_calls(spec, rows)
        points = load_points(spec, summ)
        notes += check_curve(calls, spec, summ, points)
        for src, rec in points.items():
            rec = {**rec, "round": spec["round"], "calls": calls.get(src)}
            data[src] = {**rec, "marks": marks_of(rec)}
        sources.append(f"{spec['round']}: {len(points)} installs, {len(calls)} with per-call logits")
        if points and not calls:
            print(f"[fig_curve_v2] note: {spec['round']} ships no per-call decision logits "
                  f"({rel(spec['log'])} is not released), so its arms are drawn as their two measured points "
                  f"(argmax and b95) without a curve; the paper's figure drew their curves")
    data["_parent"] = parent_point(PARENT_TABLE)

    use_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5), constrained_layout=True)
    for ax, panel in zip(axes, PANELS):
        draw(ax, data, panel)
    direct_labels(axes[0], data)
    axes[0].legend(handles=legend_handles(data), loc="lower right", bbox_to_anchor=(1.02, -0.02), fontsize=6.3)
    save(fig, "fig_curve_v2")

    payload = {"panels": [{"catch": f"{s1}/{p1}", "over_refusal": f"{s2}/{p2}"}
                          for _, (s1, p1), (s2, p2), _ in PANELS],
               "sources": sources, "checks": notes,
               "parent_graded_gate": data["_parent"],
               "b_grid": {"lo": float(B_GRID[0]), "hi": float(B_GRID[-1]), "step": float(B_GRID[1] - B_GRID[0])},
               "arms": {}}
    for src in ORDER:
        rec = data.get(src)
        if not rec:
            continue
        row = {"round": rec["round"], "b95": rec["b95"], "b_generated": rec["b_generated"],
               "generated_at": rec["generated_at"],
               "family": FAMILY.get(src, ("context", "other verdict install"))[1],
               "measured": rec["measured"], "marks": rec["marks"], "curve": None}
        if rec["calls"] is not None:
            row["curve"] = {"b": [float(b) for b in B_GRID]}
            for _, (s1, p1), (s2, p2), _ in PANELS:
                for s, p_ in ((s1, p1), (s2, p2)):
                    row["curve"][f"{s}/{p_}"] = [round(float(v), 6) for v in curve(rec["calls"], s, p_, B_GRID)]
        payload["arms"][src] = row
    jp = FIG_DIR / "fig_curve_v2.json"
    jp.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"[numbers] wrote {rel(jp)}")
    for n in notes:
        print(f"[check] {n}")
    print(curve_table(payload))


# ============================================================================ fig_contact

def fig_contact():
    print("[fig_contact] skipped: needs the episode videos of the scripted-hold / steering run, not released "
          "(the rows behind its captions are in results/diagnosis/scripted_hold_steering/rollouts.jsonl)")


# ============================================================================ method overview

METHOD_TIKZ = r"""
\begin{tikzpicture}[
  font=\small,
  box/.style={draw, rounded corners=2pt, align=center, inner sep=4pt, minimum height=1.0cm, text width=2.5cm},
  frozen/.style={box, fill=black!7},
  train/.style={box, fill=orange!15},
  tgt/.style={box, dashed},
  tok/.style={draw, inner sep=3pt, font=\small\ttfamily},
  arr/.style={-{Stealth[length=2.2mm]}, thick},
]
% ---------- training ----------
\node[font=\bfseries] at (3.4,2.3) {Training (the parent helps here only)};
\node[box] (in1) at (0,0) {frame +\\ generated command};
\node[frozen] (par) at (3.4,1.1) {parent, frozen\\ (run once, offline)};
\node[train]  (nora1) at (3.4,-1.1) {NORA + LoRA};
\node[tgt]    (tgt) at (6.8,1.1) {instruction-word residuals, layers 6, 12, 18, 24};
\node[box, draw=none, text width=2.8cm] (out1) at (6.8,-1.1) {\texttt{[STOP]} + hold chunk\\ or \texttt{[GO]} + action};
\draw[arr] (in1.east) -- (par.west);
\draw[arr] (in1.east) -- (nora1.west);
\draw[arr] (par) -- (tgt);
\draw[arr] (nora1) -- (out1);
\draw[arr, dashed, orange!70!black] (tgt.south) to[bend left=0] node[pos=0.5, right, font=\small, text=orange!70!black] {matching loss} ($(nora1.north east)+(-0.3,0)$);
% ---------- divider ----------
\draw[black!35] (8.7,2.5) -- (8.7,-2.0);
% ---------- run time ----------
\node[font=\bfseries] at (11.9,2.3) {Run time (one model)};
\node[box] (in2) at (10.3,0) {frame +\\ raw command};
\node[train] (nora2) at (13.4,0) {NORA + merged LoRA\\ (no parent)};
\draw[arr] (in2) -- (nora2);
\node[tok] (stop) at (12.5,-1.5) {[STOP] logit $+\,b$};
\node[tok] (go) at (14.6,-1.5) {[GO]};
\draw[arr] (nora2) -- (stop);
\draw[arr] (nora2) -- (go);
\node[font=\small] at (12.5,-2.1) {hold chunk};
\node[font=\small] at (14.6,-2.1) {task actions};
\end{tikzpicture}
"""

# A plain-TeX page sized to the picture (6 pt border), so only tikz is needed (no standalone class).
METHOD_DOC = (r"\documentclass{article}" "\n"
              r"\usepackage{tikz}" "\n"
              r"\usetikzlibrary{positioning,arrows.meta,calc}" "\n"
              r"\newcommand{\nora}{NORA}" "\n"
              r"\begin{document}" "\n"
              r"\setbox0\hbox{\kern6pt\vbox{\kern6pt\hbox{" + METHOD_TIKZ + r"}\kern6pt}\kern6pt}" "\n"
              r"\pdfpagewidth=\wd0 \pdfpageheight=\dimexpr\ht0+\dp0\relax" "\n"
              r"\hoffset=-1in \voffset=-1in" "\n"
              r"\shipout\box0" "\n"
              r"\end{document}" "\n")


def method():
    """The paper's method overview (Section 5), compiled from its TikZ source (paper/sections/fig_ptv.tex)."""
    pdflatex, pdftoppm = shutil.which("pdflatex"), shutil.which("pdftoppm")
    if pdflatex is None:
        print("[method] skipped: needs pdflatex")
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tex = Path(tmp) / "method_overview.tex"
        tex.write_text(METHOD_DOC)
        r = subprocess.run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex.name], cwd=tmp,
                           capture_output=True, text=True)
        pdf = Path(tmp) / "method_overview.pdf"
        if r.returncode != 0 or not pdf.exists():
            raise RuntimeError("pdflatex failed on the method overview:\n" + r.stdout[-2000:])
        shutil.copyfile(pdf, FIG_DIR / "method_overview.pdf")
        print(f"[method] wrote {rel(FIG_DIR / 'method_overview.pdf')}")
        if pdftoppm is None:
            print("[method] PNG skipped: needs pdftoppm (poppler)")
            return
        subprocess.run([pdftoppm, "-png", "-singlefile", "-scale-to-x", "1600", "-scale-to-y", "-1",
                        str(pdf), str(Path(tmp) / "method_overview")], check=True)
        shutil.copyfile(Path(tmp) / "method_overview.png", FIG_DIR / "method_overview.png")
        print(f"[method] wrote {rel(FIG_DIR / 'method_overview.png')}")


# ============================================================================ driver

FIGURES = {"fig1_transfer": fig1_transfer, "fig2_patching": fig2_patching, "fig3_tradeoff": fig3_tradeoff,
           "fig_curve_v2": fig_curve_v2, "fig_contact": fig_contact, "method": method}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="+", choices=sorted(FIGURES), help="make only these figures")
    a = ap.parse_args()
    t0 = time.time()
    for name in a.only or FIGURES:
        print(f"\n===== {name}")
        FIGURES[name]()
    print(f"\ndone in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
