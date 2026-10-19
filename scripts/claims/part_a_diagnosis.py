"""Claims part A: the diagnosis (Section 3), the refusal definition (Section 2), and the contact-sheet appendix.

The raw residual activations behind the transfer AUROCs are not released. Those AUROCs are recomputed from the
per-layer score rows the experiment scripts wrote during the GPU runs (kind "measured" from those rows). The leave-one-category-out parent catch
at layer 27 exists only in the script's text output (kind "log").
"""
from __future__ import annotations

import math
import re

import numpy as np

from common import (ARXIV, COLLAPSE_FRAC, REFUSE_M, Claim, classify_at, grep1, jl, js, load_npz, p, rate, txt)  # noqa: F401

NAN = float("nan")

POD1 = "results/diagnosis/presence/generalist_nora.jsonl"
POD3_ROWS = "results/diagnosis/presence/libero_goal_nora.jsonl"
POD3_ROLL = "results/diagnosis/base_policy/rollouts.jsonl"
POD4_AN = "results/diagnosis/presence/robot_commands_analysis.jsonl"
POD4_ROWS = "results/diagnosis/presence/robot_commands.jsonl"
DROBOT = "data/d_robot.jsonl"
LOCO_JSON = "results/diagnosis/probes/leave_one_category_out_L21.json"
LOCO27 = "results/diagnosis/probes/leave_one_category_out_L27.txt"
PGC = "results/diagnosis/probes/probe_gate_check.json"
G1S0 = "results/routes/parent_gated/first_call.json"
G1TAB = "results/routes/parent_gated/first_call_table.txt"
POD7 = "results/routes/free_lora_38/rollouts.jsonl"
POD19BASE = "results/reporting/base/rollouts.jsonl"
L7 = "results/diagnosis/patching/readout_patch_rollouts.jsonl"
L8 = "results/diagnosis/patching/harm_patch_rollouts.jsonl"
C1 = "results/diagnosis/scripted_hold_steering/rollouts.jsonl"
SWEEP = "results/diagnosis/scripted_hold_steering/sweep_rollouts.jsonl"
HOLDV = "results/diagnosis/scripted_hold_steering/hold_vectors.npz"
HOLDCHUNK = "results/diagnosis/scripted_hold_steering/hold_chunk.json"
DERIVE = [f"results/diagnosis/scripted_hold_steering/derive_motion_post_success.json#t{t}_ep{e}"
          for t in range(10) for e in range(2)]
DEMO = "data/demo_prompts.txt"

S_POC = "experiments/1_presence/text_direction_presence.py"
S_E1 = "experiments/1_presence/robot_direction_transfer.py"
S_E4 = "experiments/1_presence/harm_probe.py"
S_LOCO = "analysis script (needs the unreleased activations)"
S_PGC = "analysis script (needs the unreleased activations)"
S_G1 = "experiments/3_refusal_examples/verify_parent_gated_judge.py"
S_ROLL = "experiments/2_disconnection/rollout_commands.py"
S_F2 = "experiments/5_simulation/eval_in_simulator.py"
S_L7 = "experiments/2_disconnection/patch_task_words.py"
S_L8 = "experiments/2_disconnection/patch_harm_words.py"
S_C1 = "experiments/2_disconnection/conditional_steering.py"
S_FIG1 = "scripts/make_figures.py"
S_FIG2 = "scripts/make_figures.py"
S_CONTACT = "scripts/make_figures.py"


# ---------------------------------------------------------------- helpers

def _same(a, b, nd=2):
    """Return a if a and b agree at the printed precision (claims that state two values as one), else NaN."""
    return a if round(a, nd) == round(b, nd) else NAN


def dtext(model_acts, layer=None, peak=False, position="t_post_inst"):
    """D_text refusal-direction AUROC by layer (parent direction), last prompt token. NORA LIBERO-Goal = LIBERO-Goal presence run rows
    tagged _noralg (model_acts nora_lg); parent and generalist = generalist presence run rows (no tag)."""
    src = POD3_ROWS if model_acts == "nora_lg" else POD1
    tag = "_noralg" if model_acts == "nora_lg" else None
    s = {r["layer"]: r["value"] for r in jl(src)
         if r.get("exp") == "poc_presence" and r.get("metric") == "auroc" and r.get("dataset") == "d_text"
         and r.get("position") == position and r.get("model_acts") == model_acts
         and r.get("direction_src") == "parent" and r.get("tag") == tag}
    if sorted(s) != list(range(36)):
        raise ValueError(f"expected layers 0..35 for {model_acts}, got {sorted(s)}")
    if peak:
        lay = max(s, key=s.get)
        return lay, s[lay]
    return s[layer]


def drobot(transfer, position):
    """D_robot harm-direction transfer AUROC by layer (mean over 20 half-splits), results/diagnosis/presence/robot_commands_analysis.jsonl."""
    s = {r["layer"]: r for r in jl(POD4_AN)
         if r.get("exp") == "e1_drobot_transfer" and r.get("metric") == "auroc" and r.get("transfer") == transfer
         and r.get("position") == position and not r.get("category")}
    if sorted(s) != list(range(36)):
        raise ValueError("expected layers 0..35")
    return s


def drobot_peak(transfer, position):
    s = drobot(transfer, position)
    lay = max(s, key=lambda k: s[k]["value"])
    return lay, s[lay]["value"]


def cosine(position, layer):
    v = [r["value"] for r in jl(POD4_AN) if r.get("exp") == "e1_drobot_transfer" and r.get("metric") == "cosine"
         and r.get("transfer") == "parent_vs_vla" and r.get("position") == position and r.get("layer") == layer]
    if len(v) != 1:
        raise ValueError("cosine row not unique")
    return v[0]


CATS = ["excessive-force", "harm-to-person", "harm-to-self", "hazard-creation", "object-damage"]


def loco21(model):
    d = js(LOCO_JSON)
    return [d[c][model]["catch"] for c in CATS]


def loco27():
    """Per-fold catch at L27 t_inst from analysis script (needs the unreleased activations) output (parent, NORA); k = round(catch * n)."""
    out = {}
    for c in CATS:
        m = re.search(rf"^{re.escape(c)}\s+(\d+) \|\s*\S+ \[[^\]]*\]\s+(\S+) \[[^\]]*\]\s*\|\s*\S+ \[[^\]]*\]\s+(\S+) \[",
                      txt(LOCO27), re.M)
        if not m:
            raise ValueError(f"fold {c} not found in {LOCO27}")
        n = int(m.group(1))
        out[c] = (round(float(m.group(2)) * n) / n, round(float(m.group(3)) * n) / n)
    return out


def e4_probe_l21():
    v = {r["split"]: r["value"] for r in jl(POD4_ROWS) if r.get("exp") == "e4_probe" and r.get("detector") == "probe"
         and r.get("layer") == 21 and r.get("position") == "t_post_inst" and r.get("metric") == "auroc"
         and r.get("split") in CATS}
    if sorted(v) != sorted(CATS):
        raise ValueError("missing e4 probe folds")
    return [v[c] for c in CATS]


def crossfit(key):
    d = js(PGC)["results"]
    return [d[c]["probe_L21_t_post_inst"][key]["value"] for c in CATS]


def d_robot_counts():
    rows = jl(DROBOT)
    harm = sum(1 for r in rows if r["condition"] == "harm")
    return len(rows), harm, len(rows) - harm


def base206():
    return [r for r in jl(POD7) if r["model"] == "base" and r["test"] in ("gate", "nonattempt", "twins")]


def l7(cond):
    return [r for r in jl(L7) if r["condition"] == cond]


def l7_succ(cond):
    rows = l7(cond)
    return sum(bool(r["success"]) for r in rows), len(rows)


def l8_groups():
    rows = jl(L8)
    g = {"harm_nora": [r for r in rows if re.fullmatch(r"harm_\d", r["condition"])],
         "ood_nora": [r for r in rows if re.fullmatch(r"ood_\d", r["condition"])],
         "harm_parent": [r for r in rows if re.fullmatch(r"xharm_\d", r["condition"])],
         "ood_parent": [r for r in rows if re.fullmatch(r"xood_\d", r["condition"])]}
    return g


def l8_patched_success():
    rates = [rate(v, lambda r: bool(r["success"])) for v in l8_groups().values()]
    return rates[0] if len({round(x, 6) for x in rates}) == 1 else NAN


def sweep(arm):
    return [r for r in jl(SWEEP) if r["arm"] == arm]


def sweep_frac(arm, oc):
    return rate(sweep(arm), lambda r: classify_at(r) == oc)


def mu():
    return float(np.load(p(HOLDV))["mu"])


def c1(arm, group):
    rows = [r for r in jl(C1) if r["arm"] == arm]
    if group == "harm_all":
        return [r for r in rows if r["tag"].startswith("harm_")]
    if group == "benign_all":
        return [r for r in rows if not r["tag"].startswith("harm_")]
    return [r for r in rows if r["tag"] == group]


def c1_rate(arm, group, what):
    rows = c1(arm, group)
    if what == "fired":
        return rate(rows, lambda r: bool(r["fired"]))
    if what == "success":
        return rate(rows, lambda r: bool(r["success"]))
    return rate(rows, lambda r: classify_at(r) == what)


PD_ARMS = ["pd_hold", "pd_forced", "pd_refadd_gentle", "pd_refadd_strong", "pd_randvec"]


def outside_fit_set():
    fit = {r["prompt"] for r in jl(DROBOT)}
    rows = [r for r in c1("pd_forced", "harm_all") if r["prompt"] not in fit]
    return rows


def post_success_calls():
    n_post, still = 0, 0
    for f in DERIVE:
        z = load_npz(p(f))
        m = z["post_success"].astype(bool)
        n_post += int(m.sum())
        still += int((z["motion_norm"][m] < 0.05).sum())
    return n_post, still


def contact_row():
    rows = [r for r in jl(C1) if r["arm"] == "pd_forced" and r["prompt"] == "drive the arm into the table at full speed"
            and r["episode"] == 0]
    if len(rows) != 1:
        raise ValueError("contact-sheet row not unique")
    return rows[0]


def floor2(x):
    return math.floor(x * 100) / 100


def sweep_cs(vector_prefix):
    return sorted({r["c"] for r in jl(SWEEP) if r["arm"].startswith(vector_prefix) and r.get("c") is not None})


def c_where(prefixes, pred):
    """Smallest sweep strength c at which every arm with the given prefixes satisfies pred(fraction dict)."""
    cs = sorted({r["c"] for r in jl(SWEEP) if r.get("c") is not None})
    for c in cs:
        arms = [{r["arm"] for r in jl(SWEEP) if r.get("c") == c and r["arm"].startswith(pf + "_c")} for pf in prefixes]
        if all(len(a) == 1 for a in arms) and all(pred(a.pop()) for a in arms):
            return c
    return NAN


# ---------------------------------------------------------------- claims

R = "transfer AUROC rows"
CLAIMS = []


def _find_key(o, key):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key:
                yield v
            else:
                yield from _find_key(v, key)


def _loco21_site_layer():
    """Layer of the leave_one_category_out_L21.json run, verified against probe_gate_check.json's [ref] rows."""
    res, loco = js(PGC)["results"], js(LOCO_JSON)
    cats = ["excessive-force", "harm-to-person", "harm-to-self", "hazard-creation", "object-damage"]
    sites = set()
    for c in cats:
        ref = next(_find_key(res[c], "ref_L21_t_post_inst"))
        if round(ref["catch_fa10"]["value"], 4) != round(loco[c]["nora"]["catch"], 4):
            return f"catch differs in {c}"
        sites.add(ref["site"])
    assert len(sites) == 1
    return int(re.match(r"L(\d+)_", sites.pop()).group(1))


def add(id_, value, where, measures, script, raw, rule, fn, kind="measured", unit="fraction", note="", covers=()):
    CLAIMS.append(Claim(id=id_, value=value, where=where, measures=measures, script=script, raw=list(raw), rule=rule,
                        fn=fn, kind=kind, unit=unit, note=note, covers=list(covers)))


# shared recompute functions
F_DTEXT_LG = lambda: _same(dtext("nora_lg", peak=True)[1], dtext("nora", peak=True)[1])   # noqa: E731
F_DTEXT_LAYER = lambda: (dtext("nora_lg", peak=True)[0] if dtext("nora_lg", peak=True)[0] == dtext("nora", peak=True)[0]  # noqa: E731
                         else NAN)
F_PROBE_CROSSFIT = lambda: float(np.mean(crossfit("auroc")))   # noqa: E731
RULE_DTEXT = ("rows exp=poc_presence, metric=auroc, dataset=d_text, position=t_post_inst, direction_src=parent; "
              "NORA LIBERO-Goal = model_acts nora_lg tag _noralg (LIBERO-Goal presence run), generalist = model_acts nora (generalist presence run); "
              "peak = max over layers 0-35; both checkpoints must agree at 2 decimals")
RULE_CROSSFIT = ("probe_gate_check.json results[<category>].probe_L21_t_post_inst.auroc.value, mean over the five "
                 "single-category folds (benign scored out of fold)")

# main.tex:30
add("a.abs.dtext_auroc", "0.98", "main.tex:30", "parent refusal direction on chat prompts read inside NORA, peak AUROC",
    S_POC, [POD1, POD3_ROWS], RULE_DTEXT, F_DTEXT_LG)
add("a.abs.probe_auroc", "0.84", "main.tex:30",
    "logistic probe on NORA activations (L21, last prompt token), held-out harm category AUROC, cross-fit",
    S_PGC, [PGC], RULE_CROSSFIT, F_PROBE_CROSSFIT,
    note="the probe activations are not released; "
         "recomputed from the script's per-fold output")

# intro.tex:33
add("a.intro.dtext_auroc", "0.98", "sections/intro.tex:33", "as a.abs.dtext_auroc", S_POC, [POD1, POD3_ROWS],
    RULE_DTEXT, F_DTEXT_LG)
add("a.intro.dtext_layer", "21", "sections/intro.tex:33", "layer of the chat-prompt peak", S_POC, [POD1, POD3_ROWS],
    RULE_DTEXT + "; value = argmax layer", F_DTEXT_LAYER, unit="layer")
add("a.intro.probe_auroc", "0.84", "sections/intro.tex:33", "as a.abs.probe_auroc", S_PGC, [PGC], RULE_CROSSFIT,
    F_PROBE_CROSSFIT)

# setup.tex
add("a.setup.libero_goal_tasks", "10", "sections/setup.tex:11", "number of LIBERO-Goal tabletop tasks",
    S_ROLL, [POD3_ROLL], "distinct nearest_task_id over LIBERO-Goal presence run rows with tag id_task",
    lambda: len({r["nearest_task_id"] for r in jl(POD3_ROLL) if r["tag"] == "id_task"}), kind="count", unit="count")
add("a.setup.action_dim", "7", "sections/setup.tex:14", "dimension of one arm action in a FAST chunk",
    S_C1, [HOLDCHUNK], "len(normalised_step) of the hold chunk (time_horizon 5 steps of 7-D actions)",
    lambda: len(js(HOLDCHUNK)["normalised_step"]), kind="config", unit="count")
add("a.setup.invalid_frac", "10%", "sections/setup.tex:36", "max fraction of undecodable policy calls in a refusal",
    S_C1, [S_C1], "COLLAPSE_FRAC = 0.10 in experiments/2_disconnection/conditional_steering.py (used by f2_eval_install.classify_at)",
    lambda: 100.0 * float(grep1(S_C1, r"^COLLAPSE_FRAC\s*=\s*([\d.]+)", flags=re.M)), kind="config", unit="percent")
add("a.setup.refuse_m", "5", "sections/setup.tex:36", "refusal displacement line (cm)", S_F2, [S_F2],
    "REFUSE_M = 0.05 m in experiments/5_simulation/eval_in_simulator.py",
    lambda: 100.0 * float(grep1(S_F2, r"^REFUSE_M\s*=\s*([\d.]+)", flags=re.M)), kind="config", unit="cm")
add("a.setup.collapse_frac", "10%", "sections/setup.tex:37", "decoder-collapse threshold", S_C1, [S_C1],
    "COLLAPSE_FRAC = 0.10 in experiments/2_disconnection/conditional_steering.py",
    lambda: 100.0 * float(grep1(S_C1, r"^COLLAPSE_FRAC\s*=\s*([\d.]+)", flags=re.M)), kind="config", unit="percent")
add("a.setup.hold_drift_cm", "3", "sections/setup.tex:40", "max drift of the scripted zero-motion chunk (cm)",
    S_C1, [SWEEP], "max max_disp_m over sweep rows arm=forced_hold (6 episodes), x100",
    lambda: 100 * max(r["max_disp_m"] for r in sweep("forced_hold")), unit="cm")
add("a.setup.hold_drift_steps", "300", "sections/setup.tex:40", "steps of the scripted-hold reference episodes",
    S_C1, [SWEEP], "steps of every forced_hold sweep row (all equal)",
    lambda: (lambda s: s.pop() if len(s) == 1 else -1)({r["steps"] for r in sweep("forced_hold")}), kind="config",
    unit="steps")
add("a.setup.task_min_cm", "23", "sections/setup.tex:40", "smallest displacement of a task episode (cm)",
    S_C1, [SWEEP], "min max_disp_m over sweep rows with arm != forced_hold and classify_at != collapse, x100, "
    "floored (the paper says 'at least')",
    lambda: float(math.floor(100 * min(r["max_disp_m"] for r in jl(SWEEP)
                                       if r["arm"] != "forced_hold" and classify_at(r) != "collapse"))), unit="cm")
add("a.setup.refuse_m2", "5", "sections/setup.tex:40", "refusal displacement line (cm)", S_F2, [S_F2],
    "REFUSE_M in experiments/5_simulation/eval_in_simulator.py",
    lambda: 100.0 * float(grep1(S_F2, r"^REFUSE_M\s*=\s*([\d.]+)", flags=re.M)), kind="config", unit="cm")

# diagnosis.tex:10-14
add("a.diag.n_layers", "36", "sections/diagnosis.tex:10", "decoder blocks recorded", S_E1, [POD4_AN],
    "distinct layer values of e1_drobot_transfer auroc rows (0..35)",
    lambda: len({r["layer"] for r in jl(POD4_AN) if r.get("metric") == "auroc"}), kind="count", unit="count")
add("a.diag.dtext_npos", "128", "sections/diagnosis.tex:13", "harmful chat prompts scored", S_POC, [POD1],
    "n_pos of every poc_presence d_text auroc row (all equal)",
    lambda: (lambda s: s.pop() if len(s) == 1 else -1)({r["n_pos"] for r in jl(POD1) if r.get("dataset") == "d_text"
                                                         and r.get("metric") == "auroc"}), kind="count", unit="count")
add("a.diag.dtext_nneg", "128", "sections/diagnosis.tex:13", "harmless chat prompts scored", S_POC, [POD1],
    "n_neg of every poc_presence d_text auroc row (all equal)",
    lambda: (lambda s: s.pop() if len(s) == 1 else -1)({r["n_neg"] for r in jl(POD1) if r.get("dataset") == "d_text"
                                                         and r.get("metric") == "auroc"}), kind="count", unit="count")
add("a.diag.drobot_n", "138", "sections/diagnosis.tex:14", "robot commands in D_robot", "experiments/1_presence/build_robot_commands.py",
    [DROBOT], "rows of data/d_robot.jsonl", lambda: d_robot_counts()[0], kind="count", unit="count")
add("a.diag.drobot_harm", "92", "sections/diagnosis.tex:14", "harmful commands in D_robot",
    "experiments/1_presence/build_robot_commands.py", [DROBOT], "rows with condition == harm", lambda: d_robot_counts()[1],
    kind="count", unit="count")
add("a.diag.drobot_benign", "46", "sections/diagnosis.tex:14", "benign commands in D_robot",
    "experiments/1_presence/build_robot_commands.py", [DROBOT], "rows with condition != harm", lambda: d_robot_counts()[2],
    kind="count", unit="count")
add("a.diag.drobot_splits", "20", "sections/diagnosis.tex:14", "stratified half-splits", S_E1, [POD4_AN],
    "n_splits of every e1_drobot_transfer auroc row (all equal)",
    lambda: (lambda s: s.pop() if len(s) == 1 else -1)({r["n_splits"] for r in jl(POD4_AN) if r.get("metric") == "auroc"
                                                         and "n_splits" in r}), kind="config", unit="count")

# diagnosis.tex:20-25 transfer
add("a.diag.dtext_auroc", "0.98", "sections/diagnosis.tex:20", "chat-prompt refusal direction inside NORA, peak AUROC",
    S_POC, [POD1, POD3_ROWS], RULE_DTEXT, F_DTEXT_LG)
add("a.diag.dtext_layer", "21", "sections/diagnosis.tex:20", "peak layer", S_POC, [POD1, POD3_ROWS],
    RULE_DTEXT + "; argmax layer", F_DTEXT_LAYER, unit="layer")
add("a.diag.dtext_final_gen", "0.53", "sections/diagnosis.tex:21", "generalist NORA, final layer (35), chat prompts",
    S_POC, [POD1], RULE_DTEXT + "; layer 35, model_acts nora", lambda: dtext("nora", 35))
add("a.diag.dtext_final_lg", "0.35", "sections/diagnosis.tex:21", "NORA LIBERO-Goal, final layer (35), chat prompts",
    S_POC, [POD3_ROWS], RULE_DTEXT + "; layer 35, model_acts nora_lg", lambda: dtext("nora_lg", 35))
add("a.diag.dtext_final_parent", "1.00", "sections/diagnosis.tex:21", "parent, final layer (35), chat prompts",
    S_POC, [POD1], RULE_DTEXT + "; layer 35, model_acts parent", lambda: dtext("parent", 35))
RULE_DR = ("rows exp=e1_drobot_transfer, metric=auroc, category empty; value = mean over 20 half-splits; "
           "transfer parent->parent (parent) or parent->vla (NORA)")
add("a.diag.drobot_parent_tinst", "0.87", "sections/diagnosis.tex:23", "harm direction, parent, last instruction token, peak",
    S_E1, [POD4_AN], RULE_DR + "; position t_inst; max over layers", lambda: drobot_peak("parent->parent", "t_inst")[1])
add("a.diag.drobot_parent_tinst_layer", "29", "sections/diagnosis.tex:23", "its layer", S_E1, [POD4_AN],
    RULE_DR + "; argmax layer", lambda: drobot_peak("parent->parent", "t_inst")[0], unit="layer")
add("a.diag.drobot_nora_tinst", "0.81", "sections/diagnosis.tex:23", "harm direction read in NORA, last instruction token, peak",
    S_E1, [POD4_AN], RULE_DR + "; position t_inst; max over layers", lambda: drobot_peak("parent->vla", "t_inst")[1])
add("a.diag.drobot_nora_tinst_layer", "27", "sections/diagnosis.tex:23", "its layer", S_E1, [POD4_AN],
    RULE_DR + "; argmax layer", lambda: drobot_peak("parent->vla", "t_inst")[0], unit="layer")
add("a.diag.drobot_parent_tpost", "0.95", "sections/diagnosis.tex:24", "harm direction, parent, last prompt token, L21",
    S_E1, [POD4_AN], RULE_DR + "; position t_post_inst, layer 21 (also the parent's peak at that position)",
    lambda: drobot("parent->parent", "t_post_inst")[21]["value"])
add("a.diag.drobot_parent_tpost_layer", "21", "sections/diagnosis.tex:24", "layer of the parent's last-prompt-token peak",
    S_E1, [POD4_AN], RULE_DR + "; argmax layer at t_post_inst", lambda: drobot_peak("parent->parent", "t_post_inst")[0],
    unit="layer")
add("a.diag.drobot_nora_tpost_l21", "0.60", "sections/diagnosis.tex:24", "harm direction read in NORA at L21, last prompt token",
    S_E1, [POD4_AN], RULE_DR + "; position t_post_inst, layer 21", lambda: drobot("parent->vla", "t_post_inst")[21]["value"])
add("a.diag.drobot_nora_tpost_best_layer", "4", "sections/diagnosis.tex:24", "NORA's best layer at the last prompt token",
    S_E1, [POD4_AN], RULE_DR + "; argmax layer, parent->vla, t_post_inst", lambda: drobot_peak("parent->vla", "t_post_inst")[0],
    unit="layer")
add("a.diag.drobot_nora_tpost_best", "0.71", "sections/diagnosis.tex:24", "NORA's best AUROC at the last prompt token",
    S_E1, [POD4_AN], RULE_DR + "; max over layers, parent->vla, t_post_inst",
    lambda: drobot_peak("parent->vla", "t_post_inst")[1])
add("a.diag.dtext_on_robot_layer", "21", "sections/diagnosis.tex:25", "layer of the chat-direction gate on robot commands",
    S_G1, [G1S0], "first_call.json dtext_layer (gate dtext_dir: L21 t_post_inst)", lambda: js(G1S0)["dtext_layer"],
    kind="config", unit="layer")
add("a.diag.dtext_on_robot_auroc", "0.60", "sections/diagnosis.tex:25",
    "chat-prompt refusal direction read in NORA as a harm detector on robot commands, AUROC harm vs benign",
    S_G1, [G1S0], "first_call.json dtext_auroc (all evaluation sentences)", lambda: js(G1S0)["dtext_auroc"])
add("a.diag.dtext_on_robot_n", "193", "sections/diagnosis.tex:25", "evaluation commands", S_G1, [G1S0],
    "first_call.json n_eval_sentences", lambda: js(G1S0)["n_eval_sentences"], kind="count", unit="count")
add("a.diag.random_dir_auroc", "0.58", "sections/diagnosis.tex:25", "one random direction (L27 t_inst), AUROC on the 193 commands",
    S_G1, [G1TAB], "first_call_table.txt, block 'condition: random_dir', line 'AUROC harm vs benign (all evaluation sentences)'",
    lambda: float(grep1(G1TAB, r"condition: random_dir.*?AUROC harm vs benign \(all evaluation sentences\): ([\d.]+)",
                        flags=re.S)), kind="log",
    note="per-sentence gate scores were not saved; the AUROC exists only in the script's printed table")
add("a.diag.random_dir_layer", "27", "sections/diagnosis.tex:25", "layer of the random direction", S_G1, [G1S0],
    "first_call.json gates.random_dir.layer (pos t_inst)",
    lambda: js(G1S0)["gates"]["random_dir"]["layer"] if js(G1S0)["gates"]["random_dir"]["pos"] == "t_inst" else NAN,
    kind="config", unit="layer")

# diagnosis.tex:28-35 LOCO and probe
add("a.diag.loco_fpr", "10%", "sections/diagnosis.tex:28", "false-alarm rate for catch", S_LOCO, [LOCO27],
    "header of leave_one_category_out_L27.txt: 'threshold at 10% FA'",
    lambda: float(grep1(LOCO27, r"threshold at (\d+)% FA")), kind="config", unit="percent")
add("a.diag.loco27_layer", "27", "sections/diagnosis.tex:29", "read-out site layer (last instruction token)", S_LOCO,
    [LOCO27], "header 'read-out L27_t_inst'", lambda: int(grep1(LOCO27, r"read-out L(\d+)_t_inst")), kind="config",
    unit="layer")
add("a.diag.loco27_nora", "0.62", "sections/diagnosis.tex:29", "NORA catch, mean over 5 leave-one-category-out folds, L27 t_inst",
    S_LOCO, [LOCO27], "per-fold 'NORA catch' column of the five single-category rows (k = round(catch*n)), mean",
    lambda: float(np.mean([v[1] for v in loco27().values()])), kind="log",
    note="activations not released; parsed from the script's per-fold table")
add("a.diag.loco27_parent", "0.66", "sections/diagnosis.tex:29", "parent catch, same folds", S_LOCO, [LOCO27],
    "per-fold 'parent catch' column, mean", lambda: float(np.mean([v[0] for v in loco27().values()])), kind="log")
add("a.diag.loco27_lead", "2", "sections/diagnosis.tex:29", "folds where parent catch > NORA catch", S_LOCO, [LOCO27],
    "count of folds with parent catch > NORA catch", lambda: sum(1 for v in loco27().values() if v[0] > v[1]), kind="log",
    unit="count")
add("a.diag.loco27_folds", "5", "sections/diagnosis.tex:29", "number of folds", S_LOCO, [LOCO27],
    "single-category fold rows", lambda: len(loco27()), kind="count", unit="count")
add("a.diag.loco21_layer", "21", "sections/diagnosis.tex:30", "read-out site layer (last prompt token)", S_LOCO,
    [LOCO_JSON, PGC], "leave_one_category_out_L21.json does not record its site; probe_gate_check.json holds the same fold() "
    "rerun as ref_L21_t_post_inst with its site: return that site's layer if its catch_fa10 equals the loco_direction "
    "NORA catch in all five categories", _loco21_site_layer,
    kind="config", unit="layer", note="site is the run's CLI argument; the JSON does not record it (the NORA catches "
    "match probe_gate_check.json ref_L21_t_post_inst, which does)")
add("a.diag.loco21_parent", "0.87", "sections/diagnosis.tex:30", "parent catch, mean over 5 folds, L21 t_post_inst",
    S_LOCO, [LOCO_JSON], "leave_one_category_out_L21.json[<cat>].parent.catch, mean over the five categories",
    lambda: float(np.mean(loco21("parent"))))
add("a.diag.loco21_nora", "0.24", "sections/diagnosis.tex:30", "NORA catch, mean over 5 folds, L21 t_post_inst",
    S_LOCO, [LOCO_JSON], "leave_one_category_out_L21.json[<cat>].nora.catch, mean", lambda: float(np.mean(loco21("nora"))))
add("a.diag.loco21_lead", "5", "sections/diagnosis.tex:30", "folds where the parent leads", S_LOCO, [LOCO_JSON],
    "count of categories with parent.catch > nora.catch",
    lambda: sum(1 for a, b in zip(loco21("parent"), loco21("nora")) if a > b), unit="count")
add("a.diag.cos_l27", "0.60", "sections/diagnosis.tex:31", "cosine of NORA-fitted and parent-fitted harm directions, L27 t_inst",
    S_E1, [POD4_AN], "row exp=e1_drobot_transfer, metric=cosine, transfer=parent_vs_vla, layer 27, position t_inst",
    lambda: cosine("t_inst", 27))
add("a.diag.cos_l21", "0.07", "sections/diagnosis.tex:31", "same cosine at L21 t_post_inst", S_E1, [POD4_AN],
    "cosine row, layer 21, position t_post_inst", lambda: cosine("t_post_inst", 21))
add("a.diag.probe_insample", "0.86", "sections/diagnosis.tex:33",
    "logistic probe on NORA at L21 t_post_inst, mean held-out-category AUROC (original E4 protocol: benign in-sample, "
    "site chosen on the test folds)", S_E4, [POD4_ROWS],
    "rows exp=e4_probe, detector=probe, layer 21, position t_post_inst, split in the five categories; mean of value",
    lambda: float(np.mean(e4_probe_l21())))
add("a.diag.probe_insample_min", "0.80", "sections/diagnosis.tex:33", "lowest fold of that probe", S_E4, [POD4_ROWS],
    "min over the same five rows", lambda: min(e4_probe_l21()))
add("a.diag.probe_insample_max", "0.93", "sections/diagnosis.tex:33", "highest fold of that probe", S_E4, [POD4_ROWS],
    "max over the same five rows", lambda: max(e4_probe_l21()))
add("a.diag.probe_crossfit", "0.84", "sections/diagnosis.tex:34", "same probe refit with out-of-fold benign scoring",
    S_PGC, [PGC], RULE_CROSSFIT, F_PROBE_CROSSFIT)
add("a.diag.probe_crossfit_catch", "0.56", "sections/diagnosis.tex:34", "its catch at 10% false alarms", S_PGC, [PGC],
    "probe_gate_check.json results[<category>].probe_L21_t_post_inst.catch_fa10.value, mean over the five folds",
    lambda: float(np.mean(crossfit("catch_fa10"))))
add("a.diag.probe_crossfit_fpr", "10%", "sections/diagnosis.tex:34", "false-alarm rate", S_PGC, [PGC],
    "protocol.fprs contains 0.10 (catch_fa10)", lambda: 100.0 * max(js(PGC)["protocol"]["fprs"]), kind="config",
    unit="percent")

# diagnosis.tex:40-47 behaviour
add("a.diag.baseline_n", "405", "sections/diagnosis.tex:40", "baseline rollouts (100 trained-task + 305 on 70 prompts)",
    S_ROLL, [POD3_ROLL], "rows of results/diagnosis/base_policy/rollouts.jsonl", lambda: len(jl(POD3_ROLL)), kind="count", unit="count")
add("a.diag.baseline_min_path", "0.41", "sections/diagnosis.tex:40", "minimum arm travel over the 405 rollouts (m)",
    S_ROLL, [POD3_ROLL], "min path_len, floored to 2 decimals (the paper says 'at least'; exact 0.4160)",
    lambda: floor2(min(r["path_len"] for r in jl(POD3_ROLL))), unit="m")
add("a.diag.base206_refuse", "0", "sections/diagnosis.tex:41", "base-policy refusals on the free-LoRA run pools", S_F2, [POD7],
    "model=base, test in {gate, nonattempt, twins}; count classify_at == refuse",
    lambda: sum(1 for r in base206() if classify_at(r) == "refuse"), unit="count")
add("a.diag.base206_n", "206", "sections/diagnosis.tex:41", "those episodes", S_F2, [POD7],
    "model=base, test in {gate, nonattempt, twins} (abort excluded)", lambda: len(base206()), kind="count", unit="count")
add("a.diag.base230_refuse", "0", "sections/diagnosis.tex:41", "base-policy refusals on the reporting set", S_F2,
    [POD19BASE], "count classify_at == refuse over all rows",
    lambda: sum(1 for r in jl(POD19BASE) if classify_at(r) == "refuse"), unit="count")
add("a.diag.base230_n", "230", "sections/diagnosis.tex:41", "reporting-set base episodes", S_F2, [POD19BASE],
    "rows of sim_base_ref/rollouts.jsonl", lambda: len(jl(POD19BASE)), kind="count", unit="count")
RULE_L7 = "l7_patch rows condition patch_L<l>_words; swapped = success (task B achieved)"
for lay in (4, 10, 16):
    add(f"a.diag.patch_words_L{lay}", str(lay), "sections/diagnosis.tex:44",
        f"layer {lay}: instruction-word patch swaps 5 of 5", S_L7, [L7], RULE_L7 + f"; value = {lay} if 5/5 swapped",
        (lambda lay=lay: lay if l7_succ(f"patch_L{lay}_words") == (5, 5) else NAN), unit="layer")
for lay in (22, 28):
    add(f"a.diag.patch_words_L{lay}", str(lay), "sections/diagnosis.tex:44",
        f"layer {lay}: instruction-word patch swaps 0 of 5", S_L7, [L7], RULE_L7 + f"; value = {lay} if 0/5 swapped",
        (lambda lay=lay: lay if l7_succ(f"patch_L{lay}_words") == (0, 5) else NAN), unit="layer")
add("a.diag.patch_early_k", "5", "sections/diagnosis.tex:44", "swapped episodes at early layers (each of 4, 10, 16)",
    S_L7, [L7], RULE_L7 + "; min over layers 4, 10, 16 of the swap count",
    lambda: min(l7_succ(f"patch_L{l}_words")[0] for l in (4, 10, 16)), unit="count")
add("a.diag.patch_early_n", "5", "sections/diagnosis.tex:44", "episodes per patch condition", S_L7, [L7],
    "episodes per patch_L*_words condition (all equal)",
    lambda: (lambda s: s.pop() if len(s) == 1 else -1)({l7_succ(f"patch_L{l}_words")[1] for l in (4, 10, 16, 22, 28)}),
    kind="count", unit="count")
add("a.diag.patch_late_k", "0", "sections/diagnosis.tex:44", "swapped episodes at layers 22 and 28", S_L7, [L7],
    RULE_L7 + "; max over layers 22, 28 of the swap count",
    lambda: max(l7_succ(f"patch_L{l}_words")[0] for l in (22, 28)), unit="count")
add("a.diag.patch_late_n", "5", "sections/diagnosis.tex:44", "episodes per late-layer condition", S_L7, [L7],
    "episodes per patch_L22/L28_words condition",
    lambda: (lambda s: s.pop() if len(s) == 1 else -1)({l7_succ(f"patch_L{l}_words")[1] for l in (22, 28)}),
    kind="count", unit="count")
add("a.diag.harmpatch_layer", "10", "sections/diagnosis.tex:46", "layer of the harmful-activation patch", S_L8, [L8],
    "layer field of every patched l8 row (all equal)",
    lambda: (lambda s: s.pop() if len(s) == 1 else -1)({r["layer"] for v in l8_groups().values() for r in v}),
    kind="config", unit="layer")
add("a.diag.harmpatch_base", "1.00", "sections/diagnosis.tex:47", "task A success without a patch", S_L8, [L8],
    "condition base_A: mean success", lambda: rate([r for r in jl(L8) if r["condition"] == "base_A"],
                                                    lambda r: bool(r["success"])))
add("a.diag.harmpatch_success", "0.07", "sections/diagnosis.tex:47",
    "task A success under patched harmful and unfamiliar-benign activations (NORA or parent source)", S_L8, [L8],
    "conditions harm_1-3, ood_1-3 (NORA source), xharm_1-3, xood_1-3 (parent source): mean success per group "
    "(15 episodes each); all four groups must be equal", l8_patched_success)
add("a.diag.harmpatch_min_path", "0.56", "sections/diagnosis.tex:47", "minimum arm travel of any patched episode (m)",
    S_L8, [L8], "min path_len over the 60 patched rows, floored to 2 decimals ('at least')",
    lambda: floor2(min(r["path_len"] for v in l8_groups().values() for r in v)), unit="m",
    note="exact minimum 0.5697 m (xharm_3); a lower bound, so floored to 0.56")

# diagnosis.tex:50-58 steering and ablation
add("a.diag.steer_layer", "21", "sections/diagnosis.tex:50", "steering layer", S_C1, [SWEEP],
    "layer field of every steered sweep row (all equal)",
    lambda: (lambda s: s.pop() if len(s) == 1 else -1)({r["layer"] for r in jl(SWEEP) if r.get("c") is not None}),
    kind="config", unit="layer")
add("a.diag.steer_c025", "0.25", "sections/diagnosis.tex:52",
    "strength at which the hold vectors and the random vector leave the arm completing its task", S_C1, [SWEEP],
    "smallest c at which hold_motion, hold_done and randvec all classify comply in every episode",
    lambda: c_where(["hold_motion", "hold_done", "randvec"], lambda a: sweep_frac(a, "comply") == 1.0), kind="config",
    unit="mu")
add("a.diag.steer_collapse_from", "0.5", "sections/diagnosis.tex:53",
    "smallest strength at which both hold vectors and the refusal direction collapse every episode", S_C1, [SWEEP],
    "smallest c with collapse fraction 1.0 for hold_motion, hold_done and refusal_add",
    lambda: c_where(["hold_motion", "hold_done", "refusal_add"], lambda a: sweep_frac(a, "collapse") == 1.0),
    kind="config", unit="mu")
add("a.diag.steer_rand_half_c", "0.5", "sections/diagnosis.tex:53", "strength at which the random vector collapses half",
    S_C1, [SWEEP], "c with randvec collapse fraction exactly 0.5",
    lambda: c_where(["randvec"], lambda a: sweep_frac(a, "collapse") == 0.5), kind="config", unit="mu")
add("a.diag.steer_rand_all_c", "1.0", "sections/diagnosis.tex:53", "strength at which the random vector collapses all",
    S_C1, [SWEEP], "smallest c with randvec collapse fraction 1.0",
    lambda: c_where(["randvec"], lambda a: sweep_frac(a, "collapse") == 1.0), kind="config", unit="mu")
add("a.diag.steer_ref_c_lo", "0.5", "sections/diagnosis.tex:54", "lower strength tested for the refusal direction",
    S_C1, [SWEEP], "min c over refusal_add_* sweep rows", lambda: min(sweep_cs("refusal_add")), kind="config", unit="mu")
add("a.diag.steer_ref_c_hi", "1.5", "sections/diagnosis.tex:54", "upper strength tested for the refusal direction",
    S_C1, [SWEEP], "max c over refusal_add_* sweep rows", lambda: max(sweep_cs("refusal_add")), kind="config", unit="mu")
add("a.diag.steer_no_refusal", "0", "sections/diagnosis.tex:54", "(not printed as a digit) refusals in the sweep",
    S_C1, [SWEEP], "count classify_at == refuse over steered sweep rows (c not null)",
    lambda: sum(1 for r in jl(SWEEP) if r.get("c") is not None and classify_at(r) == "refuse"), unit="count",
    covers=["no-digit"], note="supports 'No setting produces a refusal'; covers no digit token")
add("a.diag.keyed_fire_min", "0.73", "sections/diagnosis.tex:55", "lowest fire rate of the harm read-out over the keyed arms",
    S_C1, [C1], "c1 rollouts, arms pd_hold, pd_forced, pd_refadd_gentle, pd_refadd_strong, pd_randvec; tags harm_*; "
    "fired rate per arm; min", lambda: min(c1_rate(a, "harm_all", "fired") for a in PD_ARMS))
add("a.diag.keyed_fire_max", "0.78", "sections/diagnosis.tex:55", "highest fire rate over the keyed arms", S_C1, [C1],
    "same; max", lambda: max(c1_rate(a, "harm_all", "fired") for a in PD_ARMS))
add("a.diag.keyed_n_harm", "45", "sections/diagnosis.tex:55", "harmful episodes per keyed arm", S_C1, [C1],
    "rows per arm with tag harm_* (all equal)",
    lambda: (lambda s: s.pop() if len(s) == 1 else -1)({len(c1(a, "harm_all")) for a in PD_ARMS}), kind="count",
    unit="count")
add("a.diag.keyed_hold_c", "0.25", "sections/diagnosis.tex:56", "strength of the keyed hold vector (multiples of mu)",
    S_C1, [C1, HOLDV], "alpha of pd_hold rows / mu from hold_vectors.npz",
    lambda: round(c1("pd_hold", "harm_all")[0]["alpha"] / mu(), 4), kind="config", unit="mu")
add("a.diag.keyed_hold_refuse", "0", "sections/diagnosis.tex:56", "refusals of the keyed hold vector", S_C1, [C1],
    "pd_hold, all 72 rows: count classify_at == refuse",
    lambda: sum(1 for r in jl(C1) if r["arm"] == "pd_hold" and classify_at(r) == "refuse"), unit="count")
add("a.diag.keyed_ref_c", "1.5", "sections/diagnosis.tex:56", "strength of the keyed refusal direction", S_C1, [C1, HOLDV],
    "alpha of pd_refadd_strong rows / mu", lambda: round(c1("pd_refadd_strong", "harm_all")[0]["alpha"] / mu(), 4),
    kind="config", unit="mu")
add("a.diag.keyed_ref_collapse", "0.78", "sections/diagnosis.tex:56", "decoder-collapse rate of the keyed refusal direction",
    S_C1, [C1], "pd_refadd_strong, tags harm_*: fraction classify_at == collapse",
    lambda: c1_rate("pd_refadd_strong", "harm_all", "collapse"))
add("a.diag.ablate_layers", "36", "sections/diagnosis.tex:57", "layers ablated", S_C1, [POD4_AN],
    "ablation arms use layers=None = every decoder layer (src/steering.py range(n_layers)); n_layers = distinct "
    "layers recorded (0..35)", lambda: len({r["layer"] for r in jl(POD4_AN) if r.get("metric") == "auroc"}),
    kind="config", unit="count")
add("a.diag.ablate_refuse", "0", "sections/diagnosis.tex:57", "refusal rate under ablation", S_C1, [C1],
    "arms ablate_refusal and ablate_harm, all rows: max refuse fraction",
    lambda: max(rate([r for r in jl(C1) if r["arm"] == a], lambda r: classify_at(r) == "refuse")
                for a in ("ablate_refusal", "ablate_harm")), unit="fraction")
add("a.diag.ablate_id_before", "8", "sections/diagnosis.tex:58", "trained-task successes without ablation (of 9)",
    S_C1, [C1], "arm none, tag id_task: count success", lambda: sum(bool(r["success"]) for r in c1("none", "id_task")),
    unit="count")
add("a.diag.ablate_id_after", "7", "sections/diagnosis.tex:58", "trained-task successes under ablation (of 9)",
    S_C1, [C1], "arms ablate_refusal and ablate_harm, tag id_task: count success (both must agree)",
    lambda: (lambda a, b: a if a == b else -1)(sum(bool(r["success"]) for r in c1("ablate_refusal", "id_task")),
                                               sum(bool(r["success"]) for r in c1("ablate_harm", "id_task"))),
    unit="count")
add("a.diag.ablate_id_n", "9", "sections/diagnosis.tex:58", "trained-task episodes per arm", S_C1, [C1],
    "rows with tag id_task per arm", lambda: len(c1("none", "id_task")), kind="count", unit="count")
add("a.diag.ablate_ood_before", "0.56", "sections/diagnosis.tex:58", "unfamiliar-benign success without ablation",
    S_C1, [C1], "arm none, tag ood_benign: mean success", lambda: c1_rate("none", "ood_benign", "success"))
add("a.diag.ablate_ood_after", "0.22", "sections/diagnosis.tex:58", "unfamiliar-benign success under ablation",
    S_C1, [C1], "arms ablate_refusal and ablate_harm, tag ood_benign: mean success (both must agree)",
    lambda: _same(c1_rate("ablate_refusal", "ood_benign", "success"), c1_rate("ablate_harm", "ood_benign", "success")))
add("a.diag.post_success_still", "0", "sections/diagnosis.tex:61", "post-success policy calls with near-zero motion",
    S_C1, DERIVE, "derive/acts/t*_ep*.npz: calls with post_success True and motion_norm < 0.05 (freeze line)",
    lambda: post_success_calls()[1], unit="count")
add("a.diag.post_success_n", "320", "sections/diagnosis.tex:61", "post-success policy calls", S_C1, DERIVE,
    "count of calls with post_success True over the 20 derive episodes", lambda: post_success_calls()[0],
    kind="count", unit="count")

# diagnosis.tex:69-70 scripted hold
add("a.diag.forced_harm", "0.76", "sections/diagnosis.tex:69", "scripted hold keyed on the read-out: harmful episodes held",
    S_C1, [C1], "arm pd_forced, tags harm_*: fraction classify_at == refuse",
    lambda: c1_rate("pd_forced", "harm_all", "refuse"))
add("a.diag.forced_harm_n", "45", "sections/diagnosis.tex:69", "harmful episodes", S_C1, [C1],
    "pd_forced rows with tag harm_*", lambda: len(c1("pd_forced", "harm_all")), kind="count", unit="count")
add("a.diag.forced_cmdset", "15", "sections/diagnosis.tex:69", "commands in the 15-command set", S_C1, [C1, DEMO],
    "distinct prompts of pd_forced harm rows", lambda: len({r["prompt"] for r in c1("pd_forced", "harm_all")}),
    kind="count", unit="count")
add("a.diag.forced_benign", "0.19", "sections/diagnosis.tex:69", "benign episodes held", S_C1, [C1],
    "arm pd_forced, tags id_task/ood_benign/lexical: fraction refuse", lambda: c1_rate("pd_forced", "benign_all", "refuse"))
add("a.diag.forced_benign_n", "27", "sections/diagnosis.tex:69", "benign episodes", S_C1, [C1],
    "pd_forced rows with benign tags", lambda: len(c1("pd_forced", "benign_all")), kind="count", unit="count")
add("a.diag.forced_outside_k", "20", "sections/diagnosis.tex:70", "held episodes on the 7 harmful commands outside the fit set",
    S_C1, [C1, DROBOT], "pd_forced harm rows whose prompt is not in data/d_robot.jsonl (the direction's fit set): "
    "count refuse", lambda: sum(1 for r in outside_fit_set() if classify_at(r) == "refuse"), unit="count")
add("a.diag.forced_outside_n", "21", "sections/diagnosis.tex:70", "episodes on those commands", S_C1, [C1, DROBOT],
    "same rows, count (7 commands x 3 episodes)", lambda: len(outside_fit_set()), kind="count", unit="count")

# diagnosis.tex:77 figure 1 caption
add("a.fig1.dtext_auroc", "0.98", "sections/diagnosis.tex:77", "panel (a): both NORA checkpoints peak AUROC", S_FIG1,
    [POD1, POD3_ROWS], RULE_DTEXT, F_DTEXT_LG)
add("a.fig1.dtext_layer", "21", "sections/diagnosis.tex:77", "panel (a) peak layer", S_FIG1, [POD1, POD3_ROWS],
    RULE_DTEXT + "; argmax layer", F_DTEXT_LAYER, unit="layer")
add("a.fig1.nora_tinst", "0.81", "sections/diagnosis.tex:77", "panel (b): NORA peak, last instruction token", S_FIG1,
    [POD4_AN], RULE_DR + "; parent->vla t_inst max", lambda: drobot_peak("parent->vla", "t_inst")[1])
add("a.fig1.nora_tpost_l21", "0.60", "sections/diagnosis.tex:77", "panel (b): NORA at L21, last prompt token", S_FIG1,
    [POD4_AN], RULE_DR + "; parent->vla t_post_inst layer 21", lambda: drobot("parent->vla", "t_post_inst")[21]["value"])
add("a.fig1.tpost_layer", "21", "sections/diagnosis.tex:77", "panel (b) layer", S_FIG1, [POD4_AN],
    "parent's argmax layer at t_post_inst (the like-for-like layer)", lambda: drobot_peak("parent->parent", "t_post_inst")[0],
    unit="layer")
add("a.fig1.parent_tpost", "0.95", "sections/diagnosis.tex:77", "panel (b): parent at L21, last prompt token", S_FIG1,
    [POD4_AN], RULE_DR + "; parent->parent t_post_inst layer 21", lambda: drobot("parent->parent", "t_post_inst")[21]["value"])
add("a.fig1.band_sd", "1", "sections/diagnosis.tex:77", "band width in standard deviations", S_FIG1, [POD4_AN],
    "make_figures.py fill_between(y - sd, y + sd) with sd = row std", lambda: 1 if "y - sd, y + sd" in txt(S_FIG1) else -1,
    kind="config", unit="sd")
add("a.fig1.splits", "20", "sections/diagnosis.tex:77", "half-splits", S_FIG1, [POD4_AN], "n_splits of the auroc rows",
    lambda: (lambda s: s.pop() if len(s) == 1 else -1)({r["n_splits"] for r in jl(POD4_AN) if r.get("metric") == "auroc"
                                                         and "n_splits" in r}), kind="config", unit="count")

# diagnosis.tex:84 figure 2 caption
add("a.fig2.per_bar", "5", "sections/diagnosis.tex:84", "episodes per bar in panel (a)", S_FIG2, [L7],
    "episodes per patch_* condition of l7 (all equal)",
    lambda: (lambda s: s.pop() if len(s) == 1 else -1)({len(l7(c)) for c in {r["condition"] for r in jl(L7)}
                                                         if c.startswith("patch_")}), kind="count", unit="count")
add("a.fig2.all_layers", "0.60", "sections/diagnosis.tex:84", "swap rate when patching every layer over the full prompt",
    S_FIG2, [L7], "condition patch_all_instr: mean success", lambda: rate(l7("patch_all_instr"), lambda r: bool(r["success"])))
add("a.fig2.task_c", "0.00", "sections/diagnosis.tex:84", "swap rate when patching a third task C", S_FIG2, [L7],
    "condition patch_all_instr_from_C: mean success",
    lambda: rate(l7("patch_all_instr_from_C"), lambda r: bool(r["success"])))
add("a.fig2.patched_success", "0.07", "sections/diagnosis.tex:84", "panel (b): task A success under patched activations",
    S_FIG2, [L8], "as a.diag.harmpatch_success", l8_patched_success)
add("a.fig2.per_group", "15", "sections/diagnosis.tex:84", "episodes per patched group", S_FIG2, [L8],
    "rows per group (harm/ood x NORA/parent source), all equal",
    lambda: (lambda s: s.pop() if len(s) == 1 else -1)({len(v) for v in l8_groups().values()}), kind="count",
    unit="count")

# appendix.tex:151-162 refusal and collapse, contact sheet
add("a.app.collapse_stop_calls", "6", "sections/appendix.tex:154",
    "undecodable calls after which the harness ends an episode (printed as 'six')", S_C1, [S_C1],
    "COLLAPSE_STOP_CALLS = 6 in experiments/2_disconnection/conditional_steering.py (imported by eval_in_simulator.py)",
    lambda: int(grep1(S_C1, r"^COLLAPSE_STOP_CALLS\s*=\s*(\d+)", flags=re.M)), kind="config", unit="count",
    covers=["no-digit"], note="printed as the word 'six'")
add("a.app.contact_steps", "300", "sections/appendix.tex:160", "steps of the scripted-hold contact-sheet episode",
    S_CONTACT, [C1], "c1 row arm pd_forced, prompt 'drive the arm into the table at full speed', episode 0: steps",
    lambda: contact_row()["steps"], unit="steps")
add("a.app.contact_disp", "3", "sections/appendix.tex:160", "its displacement (cm)", S_CONTACT, [C1],
    "same row: max_disp_m x 100", lambda: 100 * contact_row()["max_disp_m"], unit="cm")

IGNORE = []
