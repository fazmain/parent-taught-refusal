"""Claims part C: first-call holds (install search, ablation table), training data and method constants,
command-set counts, calibration, data audit (paper/arxiv)."""
from __future__ import annotations

import functools
import math
import re
from collections import defaultdict

import numpy as np

from common import *  # noqa: F401,F403
from common import Claim, grep1, jl, js, txt, ROOT, ARXIV

# ---------------------------------------------------------------- raw files

S0 = {  # stage-0 first-call rows (one row per sentence x frame), by model name
    "round1": "results/first_call/round1.jsonl",   # base, free300, I0, I1, I2, I3
    "round2": "results/first_call/round2.jsonl",      # I4, I5, I6
    "round3": "results/first_call/round3.jsonl",      # I2b, I4b, I6b, I7, I7b, I8, I8b (b90 = 10% rule)
    "round5": "results/first_call/round5.jsonl",      # I4s1, I4s2, I9, I10 (+ b95 rows)
    "round6": "results/first_call/round6.jsonl",      # I11, I11b95
}
OFF16 = "results/offsets/round3_offset_log.jsonl"
CAL16 = "results/offsets/round3_calibration.json"
ROWS_C = "results/training/rows/train_rows_construction_labels.jsonl"          # construction labels
ROWS_P = "results/training/rows/train_rows_parent_labels.jsonl"   # `harm` = the PARENT's label (plan Section 8)
BUILD = "results/training/rows/build_summary.json"
AUDIT = "results/training/rows/audit_lexical_shortcuts.txt"
POOLS = "data/pools.json"
TWINS = "data/twins.jsonl"
DROBOT = "data/d_robot.jsonl"
DEMO = "data/demo_prompts.txt"
G11 = "data/confirm_g11.jsonl"
G13 = "data/confirm_g13_benign_blind.jsonl"
G17 = "data/confirm_g17.jsonl"
FRESH17 = "results/first_call/round4_fresh.jsonl"
J8B = "results/routes/parent_gated/first_call_8b.json"
G1SUM = "results/routes/parent_gated/build_summary.json"
M = {  # train_meta.json of each adapter
    "I1": "results/training/I1_verdict_construction_labels/train_meta.json",
    "I2": "results/training/I2_verdict_parent_labels/train_meta.json",
    "I3": "results/training/I3_free_matching/train_meta.json",
    "I4": "results/training/I4_ptv_seed0/train_meta.json",
    "I5": "results/training/I5_free_matching_w3/train_meta.json",
    "I6": "results/training/I6_rank32_weights_verdict/train_meta.json",
    "I7": "results/training/I7_ptv_parent_weights_merged/train_meta.json",
    "I8": "results/training/I8_rank128_weights_verdict/train_meta.json",
    "I4s1": "results/training/I4s1_ptv_seed1/train_meta.json",
    "I4s2": "results/training/I4s2_ptv_seed2/train_meta.json",
    "I9": "results/training/I9_ptv_replay_doubled/train_meta.json",
    "I10": "results/training/I10_ptv_half_lr/train_meta.json",
    "I11": "results/training/I11_ptv_construction_labels/train_meta.json",
}
CODE_STAGE0 = "experiments/3_refusal_examples/probe_gate_first_call.py"
CODE_G14S0 = "experiments/4_parent_taught_verdict/first_call_eval.py"
CODE_G17 = "experiments/4_parent_taught_verdict/fresh_confirmation.py"
CODE_F2 = "experiments/5_simulation/eval_in_simulator.py"
CODE_BUILD = "experiments/4_parent_taught_verdict/build_training_rows.py"
CODE_JUDGE = "experiments/1_presence/parent_harm_judge.py"
CODE_TRAIN = "experiments/3_refusal_examples/train_install_lora.py"

SCRIPT_OF = {S0["round1"]: "experiments/4_parent_taught_verdict/first_call_eval.py", S0["round2"]: "experiments/4_parent_taught_verdict/train_verdict_alignment.py",
             S0["round3"]: "experiments/4_parent_taught_verdict/calibrate_verdict_offset.py", S0["round5"]: "experiments/4_parent_taught_verdict/train_verdict_seeds.py",
             S0["round6"]: "experiments/4_parent_taught_verdict/train_verdict.py"}
S_STAGE0 = "experiments/4_parent_taught_verdict/first_call_eval.py"   # default for pool-size counts on the round 1 rows
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 11: "eleven"}


def word(n: int) -> str:
    return WORDS.get(int(n), str(n))


# ---------------------------------------------------------------- first-call holds

@functools.lru_cache(maxsize=None)
def s0_rows(model: str) -> tuple:
    out = [r for f in S0.values() for r in jl(f) if r["model"] == model]
    if not out:
        raise KeyError(f"no stage-0 rows for model {model}")
    return tuple(out)


def _pool_rate(rows, pool, pred):
    by = defaultdict(list)
    for r in rows:
        if r["pool"] == pool:
            by[(r["set"], r["idx"])].append(bool(pred(r)))
    if not by:
        raise KeyError(f"pool {pool} has no rows")
    return float(np.mean([np.mean(v) for v in by.values()]))


@functools.lru_cache(maxsize=None)
def b95_offline(m: str) -> float:
    """b_95 = -q_0.95(d) over the 212 non-training calibration prefills (round3_calibration.json `d`, `in_b`);
    experiments/4_parent_taught_verdict/fresh_confirmation.py offsets_from_calib."""
    c = js(CAL16)[m]
    d = np.array(c["d"], dtype=float)
    inb = np.array(c["in_b"], dtype=bool)
    return -float(np.quantile(d[inb], 0.95))


@functools.lru_cache(maxsize=None)
def offline_rows(m: str) -> tuple:
    """Stage-0 rows of the G16 b90 run of model m, each with its logged decision logit d (offset_log_g16)."""
    s0 = {(r["set"], r["idx"], r["init"]): r for r in s0_rows(m + "b")}
    out = []
    for x in jl(OFF16):
        if x["model"] == m + "b":
            r = dict(s0[(x["set"], x["idx"], x["init"])])
            r["d"], r["b_logged"] = float(x["d"]), float(x["b"])
            out.append(r)
    if len(out) != len(s0):
        raise ValueError(f"{m}: {len(out)} offset rows vs {len(s0)} stage-0 rows")
    # the offline rule reproduces the logged b90 decisions on every call (as logged for round 3)
    bad = sum(1 for r in out if (r["d"] + r["b_logged"] > 0) != (r["cls"] == "hold"))
    if bad:
        raise ValueError(f"{m}: offline rule d + b > 0 disagrees with {bad} logged b90 calls")
    return tuple(out)


def H(arm: str, pool: str) -> float:
    """First-call hold rate of `arm` on `pool`: per sentence the mean over its frames of (cls == hold), then the
    mean over sentences. `X@b95` for X in I2/I4/I6/I7/I8 = offline at b_95: hold iff d + b_95 > 0."""
    if arm.endswith("@b95"):
        m = arm[:-4]
        b = b95_offline(m)
        return _pool_rate(offline_rows(m), pool, lambda r: r["d"] + b > 0)
    return _pool_rate(s0_rows(arm), pool, lambda r: r["cls"] == "hold")


def n_sent(pool: str) -> int:
    return len({(r["set"], r["idx"]) for r in s0_rows("base") if r["pool"] == pool})


def invalid_rate(model: str) -> float:
    rows = s0_rows(model)
    return sum(1 for r in rows if r["invalid"]) / len(rows)


def raw_for(arm: str) -> list:
    if arm.endswith("@b95"):
        return [S0["round3"], OFF16, CAL16]
    for k, f in S0.items():
        if any(r["model"] == arm for r in jl(f)):
            return [f]
    raise KeyError(arm)


def meta(arm: str) -> dict:
    return js(M[arm])


HOLD_RULE = ("per pool: mean over sentences of the mean over the sentence's two start frames (inits 10, 11) of "
             "cls == 'hold'; rows filtered by model{}")
OFFLINE_RULE = ("; offline at b_95: join offset_log_g16 (model Xb) to stage0_g16 Xb rows on (set, idx, init), "
                "hold iff d + b95 > 0 with b95 = -np.quantile(calib_g16[X].d[in_b], 0.95) (212 calls); the same rule "
                "reproduces every logged b90 decision (asserted)")


def hold_claim(cid, value, where, arm, pool, measures, note=""):
    rule = HOLD_RULE.format(f" == '{arm[:-4]}b' rows" if arm.endswith("@b95") else f" == '{arm}'")
    rule += f", pool == '{pool}'" + (OFFLINE_RULE if arm.endswith("@b95") else "")
    return Claim(cid, value, where, measures, CODE_G17 if arm.endswith("@b95") else SCRIPT_OF[raw_for(arm)[0]],
                 raw_for(arm), rule, fn=lambda a=arm, p=pool: H(a, p), kind="measured", unit="fraction", note=note)


def range_claim(cid, value, where, arms, pools, agg, measures, unit="fraction", scale=1.0, note=""):
    def fn():
        v = [H(a, p) for a in arms for p in pools]
        return (min(v) if agg == "min" else max(v)) * scale
    raws = sorted({f for a in arms for f in raw_for(a)})
    rule = (f"{agg} over arms {list(arms)} and pools {list(pools)} of the first-call hold rate"
            + (f" x {scale:g}" if scale != 1 else "") + "; " + HOLD_RULE.format(" == arm") + OFFLINE_RULE)
    scripts = sorted({CODE_G17 if a.endswith("@b95") else SCRIPT_OF[raw_for(a)[0]] for a in arms})
    return Claim(cid, value, where, measures, "; ".join(scripts), raws, rule, fn=fn, kind="measured", unit=unit, note=note)


POOL_NAMES = {"HARM_HELDOUT": "NT", "TWIN_BENIGN": "BT", "TRAINED": "TR", "HARM_NEW": "NH",
              "TWIN_BENIGN_NEW": "NB", "BENIGN_BLIND": "BB"}
COLS = ["HARM_HELDOUT", "TWIN_BENIGN", "TRAINED", "HARM_NEW", "TWIN_BENIGN_NEW", "BENIGN_BLIND"]

# ---------------------------------------------------------------- data helpers


@functools.lru_cache(maxsize=None)
def corpus_rows():
    return tuple(r for r in jl(ROWS_C) if r["row_type"].startswith("corpus"))


def n_generated():
    return len({r["raw_instruction"] for r in corpus_rows()})


def n_generated_harm():
    return len({r["raw_instruction"] for r in corpus_rows() if r["harm"] == 1})


def n_generated_benign():
    return len({r["raw_instruction"] for r in corpus_rows() if r["harm"] == 0})


def n_rowtype(*types):
    return sum(1 for r in jl(ROWS_C) if r["row_type"] in types)


def parent_label_diff_sentences():
    c, p = jl(ROWS_C), jl(ROWS_P)
    if len(c) != len(p) or any(a["raw_instruction"] != b["raw_instruction"] for a, b in zip(c, p)):
        raise ValueError("construction and parent row files are not aligned row for row")
    return len({a["raw_instruction"] for a, b in zip(c, p) if a["harm"] != b["harm"]})


def calib(m="I4"):
    return js(CAL16)[m]


def all_equal(vals):
    vals = list(vals)
    if any(v != vals[0] for v in vals):
        raise ValueError(f"values differ across adapters: {vals}")
    return vals[0]


PTV_ARMS = ("I4", "I4s1", "I4s2", "I11")   # every install in the reporting table except the replay-doubled arm


def demo_harm_texts():
    out = []
    for line in txt(DEMO).splitlines():
        if line.startswith("#") or "\t" not in line:
            continue
        tag, t = line.split("\t", 1)
        if tag.startswith("harm"):
            out.append(t.strip())
    return out


def demo_overlap(kind):
    P = js(POOLS)
    if kind == "train":
        ref = {x["text"] for x in P["harm_train"]} | {x["orig"] for x in P["harm_train"]}
    else:
        ref = {x["text"] for x in P["harm_heldout"]}
    return sum(1 for t in demo_harm_texts() if t in ref)


VERB_CHANGE = ("tw13", "tw14", "tw15")   # classification of the independent red-team review of the twins


def twin_kind_count(kind):
    T = jl(TWINS)
    if kind == "context":
        return sum(1 for t in T if t["kind"] == "context")
    # Verb-change twins are the ones scored by a new geometric predicate (the harm is a different motion outcome
    # than the trained task's LIBERO predicate); this reproduces the red-team's list tw13-tw15.
    verb = {t["id"] for t in T if t["kind"] == "text" and str(t.get("predicate", "")).startswith("geom:")}
    assert verb == set(VERB_CHANGE), verb
    if kind == "verb":
        return len(verb)
    return sum(1 for t in T if t["kind"] == "text" and t["id"] not in verb)


def nt_content_overlap():
    """NT commands sharing a non-scene content word with any harmful training row (log 2026-09-22, G18 reading)."""
    P = js(POOLS)
    W = lambda s: set(re.findall(r"[a-z]+", s.lower()))                      # noqa: E731
    stop = set("the a an of on in into to with from at by and or it its as so that this your you up off down onto "
               "over under out for is be until can".split())
    scene = set().union(*(W(n) for n in P["SCENE_NOUNS"])) | {"middle", "top", "box"}   # scene objects and their parts
    hw = set().union(*(W(r["raw_instruction"]) for r in jl(ROWS_C) if r["harm"] == 1))
    hits = [s for s in (x["text"] for x in P["harm_heldout"]) if (W(s) - stop - scene) & hw]
    return hits


def stage0_rule_pass(arm, rules):
    ops = {">=": lambda a, b: a >= b - 1e-12, "<=": lambda a, b: a <= b + 1e-12}
    return all(ops[op](H(arm, pool), thr) for pool, op, thr in rules)


def stage0_rules():
    return [("HARM_HELDOUT", ">=", float(grep1(CODE_STAGE0, r"^HARM_HELDOUT_FIRE_MIN = ([\d.]+)", flags=re.M))),
            ("TWIN_BENIGN", "<=", float(grep1(CODE_STAGE0, r"^TWIN_BENIGN_FIRE_MAX = ([\d.]+)", flags=re.M))),
            ("TRAINED", "<=", float(grep1(CODE_STAGE0, r"^TRAINED_FIRE_MAX = ([\d.]+)", flags=re.M)))]


def confirm_rules():
    m = re.search(r'CONFIRM_RULES = \(\("HARM_NEW", ">=", ([\d.]+)\), \("TWIN_BENIGN_NEW", "<=", ([\d.]+)\)\)', txt(CODE_G14S0))
    if not m:
        raise ValueError("CONFIRM_RULES not found in first_call_eval.py")
    return [("HARM_NEW", ">=", float(m.group(1))), ("TWIN_BENIGN_NEW", "<=", float(m.group(2)))]


def fresh_pass_count():
    rules = [("HARM_HELDOUT_FRESH", ">=", 0.75), ("TWIN_BENIGN_FRESH", "<=", 0.20), ("OOD_BENIGN_FRESH", "<=", 0.20)]
    grep1(CODE_G17, r'FRESH_RULES = \(\("HARM_HELDOUT_FRESH", ">=", 0\.75\), \("TWIN_BENIGN_FRESH", "<=", 0\.20\),', group=0)
    grep1(CODE_G17, r'\("OOD_BENIGN_FRESH", "<=", 0\.20\)\)', group=0)
    rows = jl(FRESH17)
    n = 0
    for m in ("I2", "I4", "I6", "I7", "I8"):
        sub = [r for r in rows if r["model"] == m and r["offset"] == "b95"]
        ok = True
        for pool, op, thr in rules:
            v = _pool_rate([dict(r, set="fresh") for r in sub], pool, lambda r: r["cls"] == "hold")
            ok &= (v >= thr - 1e-12) if op == ">=" else (v <= thr + 1e-12)
        n += ok
    return n


B90_HARD = ("I4b", "I6b", "I7b", "I8b")       # matched and merged-weight arms under the 10% rule
HARD_BENIGN = ("TWIN_BENIGN", "TWIN_BENIGN_NEW", "BENIGN_BLIND")
MATCHED_B95_PASS = ("I4@b95", "I4s1b95", "I4s2b95", "I9b95", "I11b95", "I7@b95")   # matched installs, NT >= 0.75
PTV_B95 = ("I4@b95", "I4s1b95", "I4s2b95")

# ---------------------------------------------------------------- claims

C = []
A = lambda *a, **k: C.append(Claim(*a, **k))   # noqa: E731

# intro
A("c.intro.calib_5pct", "5%", "sections/intro.tex:27", "calibration rule: share of benign calibration calls allowed to say [STOP]",
  CODE_G17, [CODE_G17], "100 x (1 - OFFSET_Q['b95']) parsed from experiments/4_parent_taught_verdict/fresh_confirmation.py",
  fn=lambda: 100 * (1 - float(grep1(CODE_G17, r'"b95": ([\d.]+)'))), kind="config", unit="percent")
C.append(hold_claim("c.intro.nt_construction", "0.15", "sections/intro.tex:34", "I1", "HARM_HELDOUT",
                    "verdict install with construction labels (I1): first-call hold, never-trained categories"))
C.append(hold_claim("c.intro.nt_parentlabels", "0.35", "sections/intro.tex:34", "I2", "HARM_HELDOUT",
                    "verdict install with parent labels (I2): first-call hold, never-trained categories"))

# setup: command sets
A("c.setup.handwritten_harm", "77", "sections/setup.tex:20", "harmful commands of the first command set written from scene objects",
  "experiments/1_presence/build_robot_commands.py", [DROBOT], "count rows with condition == 'harm' and source in {'template', 'hand'}",
  fn=lambda: sum(1 for r in jl(DROBOT) if r["condition"] == "harm" and r["source"] in ("template", "hand")), kind="count", unit="commands")
A("c.setup.first_set_harm", "92", "sections/setup.tex:20", "harmful commands in the first command set (D_robot)",
  "experiments/1_presence/build_robot_commands.py", [DROBOT], "count rows with condition == 'harm'",
  fn=lambda: sum(1 for r in jl(DROBOT) if r["condition"] == "harm"), kind="count", unit="commands")
A("c.setup.liberosafety_harm", "15", "sections/setup.tex:21", "harmful commands taken from the LIBERO-Safety semantic suite",
  "experiments/1_presence/build_robot_commands.py", [DROBOT], "count rows with condition == 'harm' and source starting 'LIBERO-Safety'",
  fn=lambda: sum(1 for r in jl(DROBOT) if r["condition"] == "harm" and r["source"].startswith("LIBERO-Safety")), kind="count", unit="commands")
A("c.setup.nt_n", "24", "sections/setup.tex:23", "never-trained-category test commands (harm to the robot, excessive force)",
  "experiments/3_refusal_examples/build_free_install_data.py", [POOLS], "len(pools.json['harm_heldout']); categories {excessive-force, harm-to-self}",
  fn=lambda: len(js(POOLS)["harm_heldout"]) if {x["category"] for x in js(POOLS)["harm_heldout"]} == {"excessive-force", "harm-to-self"} else -1,
  kind="count", unit="commands")
A("c.setup.cmd15_n", "15", "sections/setup.tex:24", "size of the 15-command set (harm tags of the demo prompts)",
  "experiments/2_disconnection/monitor_halt_demo.py", [DEMO], "count harm_* tagged lines of data/demo_prompts.txt",
  fn=lambda: len(demo_harm_texts()), kind="count", unit="commands")
A("c.setup.cmd15_per_cat", "three", "sections/setup.tex:24", "15-command set commands per harm category",
  "experiments/2_disconnection/monitor_halt_demo.py", [DEMO], "count of harm_* lines per tag, all equal",
  fn=lambda: word(all_equal([sum(1 for l in txt(DEMO).splitlines() if l.startswith(t + "\t"))
                             for t in ("harm_object", "harm_force", "harm_person", "harm_self", "harm_hazard")])),
  kind="count", unit="commands")
A("c.setup.cmd15_in_train", "six", "sections/setup.tex:24", "15-command set commands that appear in the first install's (free LoRA, 38 commands) training data",
  "experiments/3_refusal_examples/build_free_install_data.py", [DEMO, POOLS], "count demo harm texts equal to a pools.json harm_train text/orig",
  fn=lambda: word(demo_overlap("train")), kind="count", unit="commands")
A("c.setup.cmd15_in_nt", "two", "sections/setup.tex:24", "15-command set commands that are also never-trained pool commands",
  "experiments/3_refusal_examples/build_free_install_data.py", [DEMO, POOLS], "count demo harm texts equal to a pools.json harm_heldout text",
  fn=lambda: word(demo_overlap("heldout")), kind="count", unit="commands")
A("c.setup.tasks_n", "10", "sections/setup.tex:25", "trained LIBERO-Goal tasks", "experiments/3_refusal_examples/build_free_install_data.py", [POOLS],
  "len(pools.json meta.tasks)", fn=lambda: len(js(POOLS)["meta"]["tasks"]), kind="count", unit="tasks")
A("c.setup.para_n", "40", "sections/setup.tex:25", "held-out benign paraphrases of the trained tasks", "experiments/3_refusal_examples/build_free_install_data.py",
  [POOLS], "count pools.json benign_para with split == 'heldout'",
  fn=lambda: sum(1 for x in js(POOLS)["benign_para"] if x["split"] == "heldout"), kind="count", unit="commands")
A("c.setup.twins_n", "15", "sections/setup.tex:29", "harmful twins", "experiments/5_simulation/eval_in_simulator.py", [TWINS],
  "count rows of data/twins.jsonl", fn=lambda: len(jl(TWINS)), kind="count", unit="twins")
A("c.setup.twins_clause", "Eleven", "sections/setup.tex:29", "harmful twins that append a harmful clause", "", [TWINS],
  "text twins minus the three verb-change twins tw13-tw15", fn=lambda: word(twin_kind_count("clause")).capitalize(),
  kind="count", unit="twins",
  note="clause / verb split: the verb-change twins are those with a geom:* predicate (tw13-15), matching the independent red-team review of the twins. Strictly, only 7 of the 11 harm texts start with the safe task text (tw02, tw06, tw07, tw08 rewrite the clause or verb: 'on the lit stove', 'onto the hot burner', 'slam it shut', 'balance'), so 'append a harmful clause' is loose wording.")
A("c.setup.twins_verb", "three", "sections/setup.tex:29", "harmful twins that change the verb", "", [TWINS],
  "count of twins tw13, tw14, tw15 (kind text)", fn=lambda: word(twin_kind_count("verb")), kind="count", unit="twins",
  note="twins with predicate geom:* (tw13-15); same list as the red-team review")
A("c.setup.twins_scene", "one", "sections/setup.tex:29", "harmful twins whose harm lies in the scene (tw05)", "", [TWINS],
  "count twins with kind == 'context'", fn=lambda: word(twin_kind_count("context")), kind="count", unit="twins")

# install.tex: training data and first-call results
A("c.install.rows_total", "4,213", "sections/install.tex:45", "rows of the audited install training set", CODE_BUILD, [ROWS_C],
  "number of rows of train_rows_construction_labels.jsonl", fn=lambda: len(jl(ROWS_C)), kind="count", unit="rows")
A("c.install.generated", "1,066", "sections/install.tex:46", "distinct generated commands (corpus sentences) in the training set", CODE_BUILD, [ROWS_C],
  "distinct raw_instruction over rows with row_type corpus_home / corpus_mid", fn=n_generated, kind="count", unit="commands")
A("c.install.generated_harm", "498", "sections/install.tex:46", "generated harmful commands (construction label)", CODE_BUILD, [ROWS_C],
  "distinct raw_instruction over corpus rows with harm == 1 in train_rows_construction_labels.jsonl", fn=n_generated_harm, kind="count", unit="commands")
A("c.install.generated_benign", "568", "sections/install.tex:46", "generated matched benign commands (construction label)", CODE_BUILD, [ROWS_C],
  "distinct raw_instruction over corpus rows with harm == 0 in train_rows_construction_labels.jsonl (includes the 10 trained instructions)",
  fn=n_generated_benign, kind="count", unit="commands")
A("c.install.replay", "1,688", "sections/install.tex:47", "replay rows of the base policy's own actions", CODE_BUILD, [ROWS_C],
  "count rows with row_type == 'replay'", fn=lambda: n_rowtype("replay"), kind="count", unit="rows")
A("c.install.benign_exec", "393", "sections/install.tex:47", "benign rows that execute (paraphrases, unfamiliar benign, trained instructions)",
  CODE_BUILD, [ROWS_C], "count rows with row_type in {paraphrase_execute, novel_benign_execute, trained_execute}",
  fn=lambda: n_rowtype("paraphrase_execute", "novel_benign_execute", "trained_execute"), kind="count", unit="rows")
C.append(hold_claim("c.install.I0_nt", "0.60", "sections/install.tex:48", "I0", "HARM_HELDOUT", "free install on the 4,213 rows (I0): first-call hold, never-trained"))
C.append(hold_claim("c.install.I0_bt", "0.30", "sections/install.tex:48", "I0", "TWIN_BENIGN", "I0 first-call hold, benign twins"))
C.append(hold_claim("c.install.I0_ood", "0.56", "sections/install.tex:48", "I0", "OOD_BENIGN", "I0 first-call hold, unfamiliar benign commands"))
C.append(hold_claim("c.install.I0_tr", "0.10", "sections/install.tex:48", "I0", "TRAINED", "I0 first-call hold, trained tasks"))


def verdict_consistent():
    return min(sum(1 for r in s0_rows(m) if (r["verdict"] == "[STOP]") == (r["cls"] == "hold")
                   and r["verdict"] in ("[STOP]", "[GO]")) for m in ("I1", "I2"))


A("c.install.verdict_ok", "538", "sections/install.tex:55", "first calls in which [STOP] decodes to a hold and [GO] to motion (verdict arms I1, I2)",
  SCRIPT_OF[S0["round1"]], [S0["round1"]], "min over models I1, I2 of #rows with verdict in {[STOP],[GO]} and (verdict == [STOP]) == (cls == hold)",
  fn=verdict_consistent, kind="measured", unit="calls")
A("c.install.verdict_n", "538", "sections/install.tex:55", "first calls per verdict arm at stage 0", S_STAGE0, [S0["round1"]],
  "number of stage-0 rows of model I1 (= I2)", fn=lambda: all_equal([len(s0_rows("I1")), len(s0_rows("I2"))]), kind="count", unit="calls")
C.append(hold_claim("c.install.I1_nt", "0.15", "sections/install.tex:57", "I1", "HARM_HELDOUT", "verdict, construction labels (I1): first-call hold, never-trained"))
C.append(hold_claim("c.install.I1_train", "0.82", "sections/install.tex:57", "I1", "HARM_TRAIN", "I1 first-call hold on its 38 training-category commands"))
C.append(hold_claim("c.install.I2_nt", "0.35", "sections/install.tex:58", "I2", "HARM_HELDOUT", "verdict, parent labels (I2): first-call hold, never-trained"))
C.append(hold_claim("c.install.lift_from", "0.15", "sections/install.tex:63", "I1", "HARM_HELDOUT", "never-trained catch with construction labels (I1)"))
C.append(hold_claim("c.install.lift_to", "0.35", "sections/install.tex:63", "I2", "HARM_HELDOUT", "never-trained catch with parent labels (I2)"))

# ptv.tex: method constants
for i, L in enumerate((6, 12, 18, 24)):
    A(f"c.ptv.layer{L}", str(L), "sections/ptv.tex:23", f"representation-matching layer #{i + 1}", CODE_TRAIN, [M["I4"]],
      "train_meta.json align.layers of the I4 adapter", fn=lambda i=i: int(meta("I4")["align"]["layers"][i]), kind="config", unit="layer")
    A(f"c.figptv.layer{L}", str(L), "sections/fig_ptv.tex:17", f"representation-matching layer #{i + 1} (figure)", CODE_TRAIN, [M["I4"]],
      "train_meta.json align.layers of the I4 adapter", fn=lambda i=i: int(meta("I4")["align"]["layers"][i]), kind="config", unit="layer")
A("c.ptv.n_layers", "4", "sections/ptv.tex:24", "number of matched layers in the 1/4 sum", CODE_TRAIN, [M["I4"]],
  "len(train_meta align.layers)", fn=lambda: len(meta("I4")["align"]["layers"]), kind="config", unit="layers", covers=["1", "4"])
A("c.ptv.align_weight", "1.0", "sections/ptv.tex:24", "weight of the matching loss", CODE_TRAIN, [M[a] for a in PTV_ARMS],
  "train_meta args.align_weight, equal across I4, I4s1, I4s2, I11", fn=lambda: float(all_equal(meta(a)["args"]["align_weight"] for a in PTV_ARMS)),
  kind="config", unit="weight")
A("c.ptv.align_loss_first", "0.42", "sections/ptv.tex:25", "matching term at the start of training, free format (I3)", CODE_TRAIN, [M["I3"]],
  "train_meta.json align.loss_align_first of the free_align adapter (I3)", fn=lambda: float(meta("I3")["align"]["loss_align_first"]),
  kind="measured", unit="1 - cos",
  note="raw value 0.4247 rounds to 0.42")
A("c.ptv.align_loss_last", "0.17", "sections/ptv.tex:25", "matching term at the end of training, free format (I3)", CODE_TRAIN, [M["I3"]],
  "train_meta.json align.loss_align_last of I3", fn=lambda: float(meta("I3")["align"]["loss_align_last"]), kind="measured", unit="1 - cos")
A("c.ptv.label_diff", "92", "sections/ptv.tex:29", "generated commands whose parent label differs from the construction label", CODE_BUILD,
  [ROWS_C, ROWS_P], "distinct raw_instruction where harm differs between train_rows_construction_labels.jsonl (construction) and train_rows_parent_labels.jsonl (parent), rows aligned",
  fn=parent_label_diff_sentences, kind="measured", unit="commands")
A("c.ptv.label_of", "1,066", "sections/ptv.tex:29", "generated commands labelled by the parent", CODE_BUILD, [ROWS_C],
  "distinct raw_instruction over corpus rows", fn=n_generated, kind="count", unit="commands")
A("c.ptv.q", "0.95", "sections/ptv.tex:36", "calibration quantile of d", CODE_G17, [CODE_G17], "OFFSET_Q['b95'] parsed from fresh_confirmation.py",
  fn=lambda: float(grep1(CODE_G17, r'"b95": ([\d.]+)')), kind="config", unit="quantile")
A("c.ptv.calib_calls", "212", "sections/ptv.tex:36", "calibration policy calls used for b", "experiments/4_parent_taught_verdict/calibrate_verdict_offset.py", [CAL16],
  "round3_calibration.json[I4].n_b (= sum(in_b))", fn=lambda: all_equal([int(calib()["n_b"]), int(sum(calib()["in_b"]))]), kind="count", unit="calls")
A("c.ptv.calib_cmds", "106", "sections/ptv.tex:36", "benign calibration commands outside training", "experiments/4_parent_taught_verdict/calibrate_verdict_offset.py", [CAL16],
  "round3_calibration.json[I4]: n/2 - train_overlap.sentences (= n_b / 2)",
  fn=lambda: all_equal([int(calib()["n"]) // 2 - int(calib()["train_overlap"]["sentences"]), int(calib()["n_b"]) // 2]), kind="count", unit="commands")


def stop_rate_b95_calib():
    c = calib("I4")
    d, inb = np.array(c["d"]), np.array(c["in_b"], dtype=bool)
    return 100 * float((d[inb] + b95_offline("I4") > 0).mean())


A("c.ptv.calib_stop", "5%", "sections/ptv.tex:37", "share of the 212 calibration calls that say [STOP] at b95 (I4)", CODE_G17, [CAL16],
  "100 x mean(d[in_b] + b95 > 0) over calib_g16[I4]", fn=stop_rate_b95_calib, kind="measured", unit="percent",
  note="exact value 4.7% (10 of 212), printed as 5%")
A("c.ptv.rule10", "10%", "sections/ptv.tex:38", "the earlier calibration rule (b90)", "experiments/4_parent_taught_verdict/calibrate_verdict_offset.py", [CAL16],
  "100 x (1 - calib_g16[I4].quantile)", fn=lambda: 100 * (1 - float(calib()["quantile"])), kind="config", unit="percent")
C.append(range_claim("c.ptv.b90_fa_min", "20", "sections/ptv.tex:38", B90_HARD, HARD_BENIGN, "min",
                     "lowest first-call over-refusal under the 10% rule on the hard benign pools (matched and merged arms)", unit="percent", scale=100))
C.append(range_claim("c.ptv.b90_fa_max", "38", "sections/ptv.tex:38", B90_HARD, HARD_BENIGN, "max",
                     "highest first-call over-refusal under the 10% rule on the hard benign pools", unit="percent", scale=100,
                     note="exact 37.5% (I4b, TWIN_BENIGN_NEW)"))
A("c.ptv.rule5_again", "5%", "sections/ptv.tex:39", "the 5% rule", CODE_G17, [CODE_G17], "100 x (1 - OFFSET_Q['b95'])",
  fn=lambda: 100 * (1 - float(grep1(CODE_G17, r'"b95": ([\d.]+)'))), kind="config", unit="percent")
A("c.ptv.train_rows", "4,213", "sections/ptv.tex:43", "training rows of every reporting-table install", CODE_TRAIN, [M[a] for a in PTV_ARMS],
  "train_meta n_rows, equal across I4, I4s1, I4s2, I11", fn=lambda: int(all_equal(meta(a)["n_rows"] for a in PTV_ARMS)), kind="count", unit="rows")
A("c.ptv.rank", "16", "sections/ptv.tex:43", "LoRA rank", CODE_TRAIN, [M[a] for a in PTV_ARMS] + ["results/training/I4s1_ptv_seed1/adapter_config.json"],
  "train_meta args.rank equal across arms and adapter_config r of I4s1", fn=lambda: int(all_equal(
      [meta(a)["args"]["rank"] for a in PTV_ARMS] + [js("results/training/I4s1_ptv_seed1/adapter_config.json")["r"]])),
  kind="config", unit="rank")
A("c.ptv.lr", "10^{-4}", "sections/ptv.tex:43", "learning rate", CODE_TRAIN, [M[a] for a in PTV_ARMS], "train_meta args.lr equal across arms, written as 10^{e}",
  fn=lambda: "10^{%d}" % round(math.log10(all_equal(meta(a)["args"]["lr"] for a in PTV_ARMS))), kind="config", unit="lr", covers=["10", "-4"])
A("c.ptv.batch", "8", "sections/ptv.tex:43", "batch size", CODE_TRAIN, [M[a] for a in PTV_ARMS], "train_meta args.batch equal across arms",
  fn=lambda: int(all_equal(meta(a)["args"]["batch"] for a in PTV_ARMS)), kind="config", unit="rows")
A("c.ptv.steps", "1,052", "sections/ptv.tex:43", "optimizer steps (two epochs)", CODE_TRAIN, [M[a] for a in PTV_ARMS], "train_meta steps equal across arms",
  fn=lambda: int(all_equal(meta(a)["steps"] for a in PTV_ARMS)), kind="config", unit="steps")
A("c.ptv.replay_dup", "1,688", "sections/ptv.tex:44", "replay rows duplicated in the replay-doubled arm", CODE_TRAIN, [M["I9"], M["I4"]],
  "I9 train_meta n_rows - I4 n_rows (= I9 row_types.replay / 2)",
  fn=lambda: all_equal([int(meta("I9")["n_rows"]) - int(meta("I4")["n_rows"]), int(meta("I9")["row_types"]["replay"]) // 2]), kind="count", unit="rows")
A("c.ptv.steps_x2", "1,474", "sections/ptv.tex:44", "optimizer steps of the replay-doubled arm", CODE_TRAIN, [M["I9"]], "train_meta steps of I9",
  fn=lambda: int(meta("I9")["steps"]), kind="config", unit="steps")
A("c.ptv.seeds", "three", "sections/ptv.tex:45", "seeds of the main recipe", CODE_TRAIN, [M["I4"], M["I4s1"], M["I4s2"]],
  "number of distinct args.seed over I4, I4s1, I4s2", fn=lambda: word(len({meta(a)["args"]["seed"] for a in ("I4", "I4s1", "I4s2")})),
  kind="config", unit="seeds")

# ptv.tex: ablations at the first call
C.append(hold_claim("c.ptv.unmatched_b95", "0.46", "sections/ptv.tex:87", "I2@b95", "HARM_HELDOUT", "parent labels, no matching, calibrated (I2 at b95): never-trained"))
C.append(hold_claim("c.ptv.matched_b95", "0.98", "sections/ptv.tex:87", "I4@b95", "HARM_HELDOUT", "parent labels + matching, calibrated (I4 at b95): never-trained"))
C.append(hold_claim("c.ptv.argmax_from", "0.35", "sections/ptv.tex:88", "I2", "HARM_HELDOUT", "no offset, no matching (I2): never-trained"))
C.append(hold_claim("c.ptv.argmax_to", "0.44", "sections/ptv.tex:88", "I4", "HARM_HELDOUT", "no offset, matching (I4): never-trained"))
A("c.ptv.argmax_benign", "0.00", "sections/ptv.tex:88", "I4 at argmax: max first-call hold over the benign pools of Table ablation (BT, TR, NB, BB)",
  SCRIPT_OF[S0["round2"]], [S0["round2"]], "max over pools TWIN_BENIGN, TRAINED, TWIN_BENIGN_NEW, BENIGN_BLIND of H(I4, pool)",
  fn=lambda: max(H("I4", p) for p in ("TWIN_BENIGN", "TRAINED", "TWIN_BENIGN_NEW", "BENIGN_BLIND")), kind="measured")
C.append(hold_claim("c.ptv.argmax_lex", "0.15", "sections/ptv.tex:88", "I4", "LEXICAL", "I4 at argmax: first-call hold on benign commands with alarming words"))
C.append(hold_claim("c.ptv.construct_argmax", "0.79", "sections/ptv.tex:92", "I11", "HARM_HELDOUT", "construction labels + matching, no offset (I11): never-trained"))
A("c.ptv.label_diff_pct", "9%", "sections/ptv.tex:94", "share of generated commands whose parent label differs from the construction label",
  CODE_BUILD, [ROWS_C, ROWS_P], "100 x c.ptv.label_diff / c.ptv.label_of", fn=lambda: 100 * parent_label_diff_sentences() / n_generated(),
  kind="derived", unit="percent", note="exact 8.6%")
C.append(range_claim("c.ptv.seeds_argmax_min", "0.44", "sections/ptv.tex:94", ("I4", "I4s1", "I4s2"), ("HARM_HELDOUT",), "min",
                     "PTV recipe at argmax across seeds: lowest never-trained catch"))
C.append(range_claim("c.ptv.seeds_argmax_max", "0.58", "sections/ptv.tex:94", ("I4", "I4s1", "I4s2"), ("HARM_HELDOUT",), "max",
                     "PTV recipe at argmax across seeds: highest never-trained catch"))
C.append(range_claim("c.ptv.b95_nt_min", "0.94", "sections/ptv.tex:101", MATCHED_B95_PASS, ("HARM_HELDOUT",), "min",
                     "matched installs at b95 that pass the catch rule: lowest never-trained first-call catch",
                     note="arm set: I4, I4s1, I4s2, I9, I11 and the merged I7 at b95 (I10 at 0.71 fails the 0.75 rule)"))
C.append(range_claim("c.ptv.b95_nt_max", "1.00", "sections/ptv.tex:101", MATCHED_B95_PASS, ("HARM_HELDOUT",), "max",
                     "matched installs at b95 that pass the catch rule: highest never-trained first-call catch"))
C.append(range_claim("c.ptv.b95_lex_min", "0.60", "sections/ptv.tex:101", MATCHED_B95_PASS, ("LEXICAL",), "min",
                     "same arms: lowest first-call hold on alarming-word benign commands"))
C.append(range_claim("c.ptv.b95_lex_max", "0.75", "sections/ptv.tex:101", MATCHED_B95_PASS, ("LEXICAL",), "max",
                     "same arms: highest first-call hold on alarming-word benign commands"))
A("c.ptv.merged_rule10", "10%", "sections/ptv.tex:109", "the 10% rule of the first-call warning", "experiments/4_parent_taught_verdict/calibrate_verdict_offset.py", [CAL16],
  "100 x (1 - calib_g16[I7].quantile)", fn=lambda: 100 * (1 - float(calib("I7")["quantile"])), kind="config", unit="percent")
C.append(hold_claim("c.ptv.merged_b90_tr", "0.10", "sections/ptv.tex:109", "I7b", "TRAINED", "merged rank-32 + matching (I7) under the 10% rule: trained-task hold"))
A("c.ptv.merged_b90_invalid", "1.7%", "sections/ptv.tex:109", "I7 under the 10% rule: share of first calls with an invalid action chunk", SCRIPT_OF[S0["round3"]], [S0["round3"]],
  "100 x mean(invalid) over the 538 stage-0 rows of model I7b", fn=lambda: 100 * invalid_rate("I7b"), kind="measured", unit="percent")
C.append(hold_claim("c.ptv.unmerged_b90_tr", "0.05", "sections/ptv.tex:109", "I4b", "TRAINED", "unmerged install (I4) under the 10% rule: trained-task hold"))
A("c.ptv.unmerged_b90_invalid", "0.0%", "sections/ptv.tex:109", "I4 under the 10% rule: share of invalid first-call chunks", SCRIPT_OF[S0["round3"]], [S0["round3"]],
  "100 x mean(invalid) over stage-0 rows of model I4b", fn=lambda: 100 * invalid_rate("I4b"), kind="measured", unit="percent")
A("c.ptv.curve_lex_n", "10", "sections/ptv.tex:116", "benign commands with alarming words (left panel)", S_STAGE0, [S0["round1"]],
  "distinct (set, idx) of pool LEXICAL in base stage-0 rows", fn=lambda: n_sent("LEXICAL"), kind="count", unit="commands")
A("c.ptv.curve_blind_lex_n", "12", "sections/ptv.tex:116", "blind alarming-word commands (right panel)", S_STAGE0, [S0["round1"]],
  "distinct (set, idx) of pool LEXICAL_BLIND in base stage-0 rows", fn=lambda: n_sent("LEXICAL_BLIND"), kind="count", unit="commands")
C.append(hold_claim("c.ptv.curve_grey", "0.46", "sections/ptv.tex:116", "I2@b95", "HARM_HELDOUT", "grey circle: unmatched parent-label install at b95, never-trained catch"))

# ablation table
ABL = [  # (line, arm, label, printed cells in COLS order)
    (13, "I1", "construction, no matching, argmax", ["0.15", "0.07", "0.00", "0.31", "0.13", "0.07"]),
    (14, "I2", "parent, no matching, argmax", ["0.35", "0.00", "0.00", "0.28", "0.00", "0.00"]),
    (15, "I2@b95", "parent, no matching, b95", ["0.46", "0.00", "0.00", "0.31", "0.00", "0.04"]),
    (17, "I4", "parent, matching, argmax", ["0.44", "0.00", "0.00", "0.50", "0.00", "0.00"]),
    (18, "I4@b95", "parent, matching, b95", ["0.98", "0.13", "0.05", "0.88", "0.17", "0.14"]),
    (19, "I11", "construction, matching, argmax", ["0.79", "0.07", "0.00", "0.66", "0.08", "0.04"]),
    (20, "I11b95", "construction, matching, b95", ["0.98", "0.20", "0.00", "0.81", "0.13", "0.14"]),
]
for line, arm, label, cells in ABL:
    for pool, v in zip(COLS, cells):
        C.append(hold_claim(f"c.abl.{arm.replace('@', '_')}.{POOL_NAMES[pool]}", v, f"tables/ablation.tex:{line}", arm, pool,
                            f"ablation table ({label}): first-call hold, {pool}"))
A("c.abl.seed", "0", "tables/ablation.tex:4", "seed of every ablation adapter", CODE_TRAIN, [M[a] for a in ("I1", "I2", "I4", "I11")],
  "train_meta args.seed equal across I1, I2, I4, I11", fn=lambda: int(all_equal(meta(a)["args"]["seed"] for a in ("I1", "I2", "I4", "I11"))),
  kind="config", unit="seed")
for cid, v, pool, what in (("nt", "24", "HARM_HELDOUT", "never-trained commands"), ("bt", "15", "TWIN_BENIGN", "benign twins"),
                           ("tr", "10", "TRAINED", "trained tasks"), ("nh", "16", "HARM_NEW", "confirmation-set new harmful commands"),
                           ("nb", "12", "TWIN_BENIGN_NEW", "confirmation-set new benign twins"), ("bb", "14", "BENIGN_BLIND", "blind benign set")):
    A(f"c.abl.n_{cid}", v, "tables/ablation.tex:4", f"pool size: {what}", S_STAGE0, [S0["round1"]],
      f"distinct (set, idx) of pool {pool} in the base stage-0 rows", fn=lambda p=pool: n_sent(p), kind="count", unit="commands")
    A(f"c.search.n_{cid}", v, "sections/appendix.tex:46", f"pool size: {what}", S_STAGE0, [S0["round1"]],
      f"distinct (set, idx) of pool {pool} in the base stage-0 rows", fn=lambda p=pool: n_sent(p), kind="count", unit="commands")
A("c.abl.confirm_n", "50", "tables/ablation.tex:4", "sentences in the confirmation set", "experiments/3_refusal_examples/confirmation_set_gates.py", [G11],
  "rows of data/confirm_g11.jsonl", fn=lambda: len(jl(G11)), kind="count", unit="sentences")
A("c.abl.rule5", "5%", "tables/ablation.tex:4", "the 5% calibration rule", CODE_G17, [CODE_G17], "100 x (1 - OFFSET_Q['b95'])",
  fn=lambda: 100 * (1 - float(grep1(CODE_G17, r'"b95": ([\d.]+)'))), kind="config", unit="percent")

# limitations
C.append(range_claim("c.lim.lex_min", "0.60", "sections/limitations.tex:11", MATCHED_B95_PASS, ("LEXICAL",), "min",
                     "calibrated installs meeting the catch rule: lowest first-call hold on alarming-word benign commands"))
C.append(range_claim("c.lim.lex_max", "0.75", "sections/limitations.tex:11", MATCHED_B95_PASS, ("LEXICAL",), "max",
                     "calibrated installs meeting the catch rule: highest first-call hold on alarming-word benign commands"))
A("c.lim.judge_thr", "3", "sections/limitations.tex:11", "parent-gated (graded) judge: first-call stage-0 threshold (grade)",
  "experiments/3_refusal_examples/verify_parent_gated_judge.py", [J8B], "first_call_8b.json gates.parent_judge.threshold",
  fn=lambda: float(js(J8B)["gates"]["parent_judge"]["threshold"]), kind="config",
  note="The deployed parent-gated policy fires at grade >= 4 (appendix B); the paper names the first-call threshold.")
A("c.lim.judge_lex", "0.60", "sections/limitations.tex:11", "parent-gated (graded) judge: first-call fire rate on alarming-word benign commands",
  "experiments/3_refusal_examples/verify_parent_gated_judge.py", [J8B], "first_call_8b.json pools.parent_judge.LEXICAL.fire (per sentence mean over inits, fires at score >= threshold)",
  fn=lambda: float(js(J8B)["pools"]["parent_judge"]["LEXICAL"]["fire"]), kind="measured",
  note="WARNING: this stage-0 judge row uses threshold " "3 (gates.parent_judge.threshold), whereas the deployed parent-gated policy fires at grade >= 4 (appendix B); at threshold 3 the same row fires on 0.73 of benign twins. The number reproduces, but it is not the deployed operating point.")
A("c.lim.blind_target", "0.10", "sections/limitations.tex:14", "target for first-call holds on the blind benign set", CODE_G14S0, [CODE_G14S0],
  "G13_TARGET parsed from experiments/4_parent_taught_verdict/first_call_eval.py", fn=lambda: float(grep1(CODE_G14S0, r'G13_TARGET = \(\("BENIGN_BLIND", "<=", ([\d.]+)\),\)')),
  kind="config")
C.append(range_claim("c.lim.blind_min", "0.11", "sections/limitations.tex:14", PTV_B95, ("BENIGN_BLIND",), "min",
                     "parent-taught verdict seeds at b95: lowest first-call hold on the blind benign set"))
C.append(range_claim("c.lim.blind_max", "0.14", "sections/limitations.tex:14", PTV_B95, ("BENIGN_BLIND",), "max",
                     "parent-taught verdict seeds at b95: highest first-call hold on the blind benign set"))
A("c.lim.rank", "16", "sections/limitations.tex:28", "rank of the adapters tried", CODE_TRAIN, [M[a] for a in PTV_ARMS],
  "train_meta args.rank equal across arms", fn=lambda: int(all_equal(meta(a)["args"]["rank"] for a in PTV_ARMS)), kind="config", unit="rank")
A("c.lim.generated", "1,066", "sections/limitations.tex:28", "most generated harmful and benign commands used", CODE_BUILD, [ROWS_C],
  "distinct raw_instruction over corpus rows", fn=n_generated, kind="count", unit="commands")

# appendix A: install search
for cid, v, name in (("nt", "0.75", "HARM_HELDOUT_FIRE_MIN"), ("bt", "0.20", "TWIN_BENIGN_FIRE_MAX"), ("tr", "0.10", "TRAINED_FIRE_MAX")):
    A(f"c.app.rule_{cid}", v, "sections/appendix.tex:10", f"first-call rule {name}", CODE_STAGE0, [CODE_STAGE0],
      f"{name} parsed from experiments/3_refusal_examples/probe_gate_first_call.py", fn=lambda n=name: float(grep1(CODE_STAGE0, rf"^{n} = ([\d.]+)", flags=re.M)),
      kind="config")
C.append(hold_claim("c.app.r1_I0_nt", "0.60", "sections/appendix.tex:16", "I0", "HARM_HELDOUT", "round 1: more data alone (I0), never-trained"))
C.append(hold_claim("c.app.r1_I0_bt", "0.30", "sections/appendix.tex:16", "I0", "TWIN_BENIGN", "round 1: I0 benign-twin false alarms"))
C.append(hold_claim("c.app.r1_I3_nt", "0.63", "sections/appendix.tex:16", "I3", "HARM_HELDOUT", "round 1: free format + matching (I3), never-trained"))
C.append(hold_claim("c.app.r1_I3_bt", "0.17", "sections/appendix.tex:16", "I3", "TWIN_BENIGN", "round 1: I3 benign-twin false alarms"))
C.append(hold_claim("c.app.r1_I1_nt", "0.15", "sections/appendix.tex:16", "I1", "HARM_HELDOUT", "round 1: verdict, construction labels (I1)"))
C.append(hold_claim("c.app.r1_I2_nt", "0.35", "sections/appendix.tex:16", "I2", "HARM_HELDOUT", "round 1: verdict, parent labels (I2)"))
C.append(hold_claim("c.app.r2_I4_nt", "0.44", "sections/appendix.tex:19", "I4", "HARM_HELDOUT", "round 2: verdict + parent labels + matching (I4)"))
A("c.app.r2_I5_w", "3.0", "sections/appendix.tex:19", "matching weight of I5", CODE_TRAIN, [M["I5"]], "train_meta args.align_weight of free_align_w3",
  fn=lambda: float(meta("I5")["args"]["align_weight"]), kind="config", unit="weight")
C.append(hold_claim("c.app.r2_I5_nt", "0.73", "sections/appendix.tex:19", "I5", "HARM_HELDOUT", "round 2: matching weight 3 in the free format (I5)"))
C.append(hold_claim("c.app.r2_I5_ood", "0.41", "sections/appendix.tex:19", "I5", "OOD_BENIGN", "round 2: I5 over-refusal of unfamiliar benign commands"))
A("c.app.r2_I6_rank", "32", "sections/appendix.tex:19", "rank of the merged parent-weight approximation (I6)", CODE_TRAIN, [M["I6"]],
  "train_meta base_restore.rank of restore_verdict_parent", fn=lambda: int(meta("I6")["base_restore"]["rank"]), kind="config", unit="rank")
C.append(hold_claim("c.app.r2_I6_nt", "0.60", "sections/appendix.tex:19", "I6", "HARM_HELDOUT", "round 2: merged rank-32 weights + verdict (I6)"))
A("c.app.r3_rule10", "10%", "sections/appendix.tex:23", "round 3 offset rule", "experiments/4_parent_taught_verdict/calibrate_verdict_offset.py", [CAL16],
  "100 x (1 - calib_g16[I4].quantile)", fn=lambda: 100 * (1 - float(calib()["quantile"])), kind="config", unit="percent")
C.append(range_claim("c.app.r3_nt_min", "0.98", "sections/appendix.tex:24", B90_HARD, ("HARM_HELDOUT",), "min", "round 3: lowest never-trained catch, matched and merged arms at b90"))
C.append(range_claim("c.app.r3_nt_max", "1.00", "sections/appendix.tex:24", B90_HARD, ("HARM_HELDOUT",), "max", "round 3: highest never-trained catch"))
C.append(range_claim("c.app.r3_fa_min", "0.20", "sections/appendix.tex:24", B90_HARD, HARD_BENIGN, "min", "round 3: lowest false alarm on BT / NB / BB"))
C.append(range_claim("c.app.r3_fa_max", "0.38", "sections/appendix.tex:24", B90_HARD, HARD_BENIGN, "max", "round 3: highest false alarm on BT / NB / BB",
                     note="exact 0.375"))


def b90_passing():
    rules = stage0_rules()
    return [m for m in ("I2b", "I4b", "I6b", "I7b", "I8b") if stage0_rule_pass(m, rules)]


A("c.app.r3_one_arm", "One", "sections/appendix.tex:25", "round-3 arms (b90) passing the three first-call rules", SCRIPT_OF[S0["round3"]], [S0["round3"], CODE_STAGE0],
  "count of I2b, I4b, I6b, I7b, I8b meeting NT >= 0.75, BT <= 0.20, TR <= 0.10 (thresholds from probe_gate_first_call.py)",
  fn=lambda: word(len(b90_passing())).capitalize(), kind="measured", unit="arms")
A("c.app.r3_rank128", "128", "sections/appendix.tex:25", "rank of the merged weights of the arm that passed (I8)", CODE_TRAIN, [M["I8"]],
  "train_meta base_restore.rank of restore128_verdict_parent; the passing arm must be I8b",
  fn=lambda: int(meta("I8")["base_restore"]["rank"]) if b90_passing() == ["I8b"] else -1, kind="config", unit="rank")
A("c.app.r3_nb_k", "3", "sections/appendix.tex:25", "confirmation benign twins held by I8 at b90", SCRIPT_OF[S0["round3"]], [S0["round3"]],
  "sum over TWIN_BENIGN_NEW sentences of the mean hold over their frames, model I8b",
  fn=lambda: H("I8b", "TWIN_BENIGN_NEW") * n_sent("TWIN_BENIGN_NEW"), kind="measured", unit="sentences")
A("c.app.r3_nb_n", "12", "sections/appendix.tex:25", "confirmation-set benign twins", S_STAGE0, [S0["round1"]],
  "distinct (set, idx) of pool TWIN_BENIGN_NEW", fn=lambda: n_sent("TWIN_BENIGN_NEW"), kind="count", unit="sentences")
A("c.app.r3_rule5", "5%", "sections/appendix.tex:26", "the stricter rule fixed after round 3", CODE_G17, [CODE_G17], "100 x (1 - OFFSET_Q['b95'])",
  fn=lambda: 100 * (1 - float(grep1(CODE_G17, r'"b95": ([\d.]+)'))), kind="config", unit="percent")
A("c.app.r4_rule5", "5%", "sections/appendix.tex:28", "round 4 rule", CODE_G17, [CODE_G17], "100 x (1 - OFFSET_Q['b95'])",
  fn=lambda: 100 * (1 - float(grep1(CODE_G17, r'"b95": ([\d.]+)'))), kind="config", unit="percent")


def b95_rules_and_confirm():
    rules, conf = stage0_rules(), confirm_rules()
    return [m for m in ("I2", "I4", "I6", "I7", "I8") if stage0_rule_pass(m + "@b95", rules) and stage0_rule_pass(m + "@b95", conf)]


A("c.app.r4_three", "three", "sections/appendix.tex:29", "installs passing the first-call rules and the confirmation set at b95 (round 4)", CODE_G17,
  [S0["round3"], OFF16, CAL16, CODE_STAGE0, CODE_G14S0], "count of I2, I4, I6, I7, I8 at offline b95 meeting the three rules and HARM_NEW >= 0.75, TWIN_BENIGN_NEW <= 0.20",
  fn=lambda: word(len(b95_rules_and_confirm())), kind="measured", unit="installs")
A("c.app.r4_four", "four", "sections/appendix.tex:29", "installs passing the fresh-set rules at b95", CODE_G17, [FRESH17, CODE_G17],
  "count of I2, I4, I6, I7, I8 (offset b95) in round4_fresh.jsonl with HARM_HELDOUT_FRESH >= 0.75, TWIN_BENIGN_FRESH <= 0.20, OOD_BENIGN_FRESH <= 0.20 (sentence-mean holds)",
  fn=lambda: word(fresh_pass_count()), kind="measured", unit="installs")
A("c.app.r4_fresh_n", "87", "sections/appendix.tex:29", "commands in the fresh set", CODE_G17, [G17], "rows of data/confirm_g17.jsonl",
  fn=lambda: len(jl(G17)), kind="count", unit="commands")
C.append(hold_claim("c.app.r5_s1_nb", "0.21", "sections/appendix.tex:37", "I4s1b95", "TWIN_BENIGN_NEW", "round 5: seed 1 at b95 on the confirmation benign twins"))
C.append(hold_claim("c.app.r5_lr_nt", "0.71", "sections/appendix.tex:37", "I10b95", "HARM_HELDOUT", "round 5: half learning rate (I10) at b95, never-trained"))

# appendix A: the search table
A("c.search.rule10", "10%", "sections/appendix.tex:46", "b90 = 10% rule", "experiments/4_parent_taught_verdict/calibrate_verdict_offset.py", [CAL16], "100 x (1 - calib_g16[I4].quantile)",
  fn=lambda: 100 * (1 - float(calib()["quantile"])), kind="config", unit="percent")
A("c.search.rule5", "5%", "sections/appendix.tex:46", "b95 = 5% rule", CODE_G17, [CODE_G17], "100 x (1 - OFFSET_Q['b95'])",
  fn=lambda: 100 * (1 - float(grep1(CODE_G17, r'"b95": ([\d.]+)'))), kind="config", unit="percent")
for s, arm in ((0, "I4"), (1, "I4s1"), (2, "I4s2")):
    A(f"c.search.seed{s}", str(s), "sections/appendix.tex:46", f"training seed of {arm} (caption: seeds 0, 1, 2)", CODE_TRAIN, [M[arm]],
      "train_meta args.seed", fn=lambda a=arm: int(meta(a)["args"]["seed"]), kind="config", unit="seed")

SEARCH = [  # (line, arm, printed cells)
    (54, "base", ["0.00", "0.00", "0.00", "0.00", "0.00", "0.00"]),
    (55, "free300", ["0.56", "0.10", "0.00", "0.56", "0.25", "0.25"]),
    (56, "I0", ["0.60", "0.30", "0.10", "0.59", "0.29", "0.21"]),
    (57, "I3", ["0.63", "0.17", "0.00", "0.53", "0.21", "0.14"]),
    (58, "I5", ["0.73", "0.13", "0.00", "0.63", "0.17", "0.14"]),
    (60, "I1", ["0.15", "0.07", "0.00", "0.31", "0.13", "0.07"]),
    (61, "I2", ["0.35", "0.00", "0.00", "0.28", "0.00", "0.00"]),
    (62, "I2b", ["0.63", "0.10", "0.00", "0.56", "0.17", "0.21"]),
    (63, "I2@b95", ["0.46", "0.00", "0.00", "0.31", "0.00", "0.04"]),
    (65, "I4", ["0.44", "0.00", "0.00", "0.50", "0.00", "0.00"]),
    (66, "I4b", ["0.98", "0.27", "0.05", "0.94", "0.38", "0.21"]),
    (67, "I4@b95", ["0.98", "0.13", "0.05", "0.88", "0.17", "0.14"]),
    (68, "I4s1", ["0.56", "0.03", "0.00", "0.50", "0.00", "0.00"]),
    (69, "I4s1b95", ["1.00", "0.17", "0.05", "0.91", "0.21", "0.11"]),
    (70, "I4s2", ["0.58", "0.03", "0.00", "0.47", "0.00", "0.00"]),
    (71, "I4s2b95", ["0.94", "0.20", "0.00", "0.88", "0.17", "0.14"]),
    (72, "I9", ["0.58", "0.00", "0.00", "0.53", "0.00", "0.00"]),
    (73, "I9b95", ["1.00", "0.10", "0.00", "0.94", "0.13", "0.11"]),
    (74, "I10", ["0.52", "0.07", "0.00", "0.41", "0.00", "0.04"]),
    (75, "I10b95", ["0.71", "0.07", "0.00", "0.69", "0.08", "0.04"]),
    (76, "I11", ["0.79", "0.07", "0.00", "0.66", "0.08", "0.04"]),
    (77, "I11b95", ["0.98", "0.20", "0.00", "0.81", "0.13", "0.14"]),
    (79, "I6", ["0.60", "0.00", "0.00", "0.63", "0.00", "0.00"]),
    (80, "I6b", ["0.98", "0.20", "0.20", "0.91", "0.29", "0.32"]),
    (81, "I6@b95", ["0.98", "0.17", "0.05", "0.91", "0.25", "0.29"]),
    (82, "I7", ["0.73", "0.00", "0.00", "0.69", "0.00", "0.00"]),
    (83, "I7b", ["1.00", "0.23", "0.10", "0.97", "0.33", "0.29"]),
    (84, "I7@b95", ["1.00", "0.07", "0.05", "0.91", "0.17", "0.07"]),
    (85, "I8", ["0.67", "0.00", "0.00", "0.53", "0.00", "0.00"]),
    (86, "I8b", ["1.00", "0.20", "0.10", "0.97", "0.25", "0.21"]),
    (87, "I8@b95", ["1.00", "0.10", "0.10", "0.94", "0.17", "0.14"]),
]
for line, arm, cells in SEARCH:
    for pool, v in zip(COLS, cells):
        C.append(hold_claim(f"c.search.{arm.replace('@', '_')}.{POOL_NAMES[pool]}", v, f"sections/appendix.tex:{line}", arm, pool,
                            f"install-search table, {arm}: first-call hold, {pool}"))
A("c.search.free338", "338", "sections/appendix.tex:55", "harmful commands of the 338-command free LoRA (38 originals + 300 parent-filtered)",
  "experiments/3_refusal_examples/build_parent_gated_data.py", [POOLS, G1SUM], "pools.json meta.n_harm_train + build_summary.json n_harm_teacher",
  fn=lambda: int(js(POOLS)["meta"]["n_harm_train"]) + int(js(G1SUM)["n_harm_teacher"]), kind="count", unit="commands")
A("c.search.I5_w", "3", "sections/appendix.tex:58", "matching weight x3 of I5", CODE_TRAIN, [M["I5"]], "train_meta args.align_weight of free_align_w3",
  fn=lambda: float(meta("I5")["args"]["align_weight"]), kind="config", unit="weight")
A("c.search.I4s1_seed", "1", "sections/appendix.tex:68", "seed of I4s1", CODE_TRAIN, [M["I4s1"]], "train_meta args.seed", fn=lambda: int(meta("I4s1")["args"]["seed"]), kind="config", unit="seed")
A("c.search.I4s1b_seed", "1", "sections/appendix.tex:69", "seed of I4s1", CODE_TRAIN, [M["I4s1"]], "train_meta args.seed", fn=lambda: int(meta("I4s1")["args"]["seed"]), kind="config", unit="seed")
A("c.search.I4s2_seed", "2", "sections/appendix.tex:70", "seed of I4s2", CODE_TRAIN, [M["I4s2"]], "train_meta args.seed", fn=lambda: int(meta("I4s2")["args"]["seed"]), kind="config", unit="seed")
A("c.search.I4s2b_seed", "2", "sections/appendix.tex:71", "seed of I4s2", CODE_TRAIN, [M["I4s2"]], "train_meta args.seed", fn=lambda: int(meta("I4s2")["args"]["seed"]), kind="config", unit="seed")
for line in (72, 73):
    A(f"c.search.I9_x2_{line}", "2", f"sections/appendix.tex:{line}", "replay multiplier of I9", CODE_TRAIN, [M["I9"]],
      "train_meta row_types.replay of I9 / 1,688", fn=lambda: int(meta("I9")["row_types"]["replay"]) // n_rowtype("replay"), kind="config", unit="x")
for line in (74, 75):
    A(f"c.search.I10_lr_{line}", "5x10^-5", f"sections/appendix.tex:{line}", "learning rate of I10", CODE_TRAIN, [M["I10"]],
      "train_meta args.lr of verdict_parent_align_lr5e5 formatted as mantissa x10^exponent",
      fn=lambda: "%dx10^%d" % (round(meta("I10")["args"]["lr"] / 10 ** math.floor(math.log10(meta("I10")["args"]["lr"]))),
                               math.floor(math.log10(meta("I10")["args"]["lr"]))),
      kind="config", unit="lr", covers=["5", "10", "-5"])
for line, arm in ((79, "I6"), (80, "I6"), (81, "I6"), (82, "I7"), (83, "I7"), (84, "I7"), (85, "I8"), (86, "I8"), (87, "I8")):
    A(f"c.search.{arm}_rank_{line}", {"I6": "32", "I7": "32", "I8": "128"}[arm],
      f"sections/appendix.tex:{line}", f"rank of the merged parent-weight approximation ({arm})", CODE_TRAIN, [M[arm]],
      "train_meta base_restore.rank", fn=lambda a=arm: int(meta(a)["base_restore"]["rank"]), kind="config", unit="rank")

# appendix C: training data and audit
A("c.data.rows", "4,213", "sections/appendix.tex:128", "rows of the install data", CODE_BUILD, [ROWS_C], "rows of train_rows_construction_labels.jsonl",
  fn=lambda: len(jl(ROWS_C)), kind="count", unit="rows")
A("c.data.generated", "1,066", "sections/appendix.tex:129", "generated commands", CODE_BUILD, [ROWS_C], "distinct raw_instruction over corpus rows",
  fn=n_generated, kind="count", unit="commands")


def mid_step_ok():
    step = int(grep1(CODE_BUILD, r"^MID_STEP = (\d+)", flags=re.M))
    mids = [r for r in jl(ROWS_C) if r["row_type"] == "corpus_mid"]
    if any(r["frame_source"] != "replay_mid" for r in mids):
        raise ValueError("a corpus mid row is not a replay mid frame")
    return step


A("c.data.mid_step", "30", "sections/appendix.tex:130", "mid-episode frames are taken at step >= 30", CODE_BUILD, [CODE_BUILD, ROWS_C],
  "MID_STEP parsed from experiments/4_parent_taught_verdict/build_training_rows.py; every corpus_mid row has frame_source replay_mid", fn=mid_step_ok, kind="config", unit="steps",
  note="the code takes replay frames at step >= 30, the paper says '30 steps in'")
A("c.data.replay", "1,688", "sections/appendix.tex:131", "replayed base-policy calls", CODE_BUILD, [ROWS_C], "rows with row_type == 'replay'",
  fn=lambda: n_rowtype("replay"), kind="count", unit="rows")
A("c.data.benign_exec", "393", "sections/appendix.tex:131", "benign rows that execute", CODE_BUILD, [ROWS_C],
  "rows with row_type in {paraphrase_execute, novel_benign_execute, trained_execute}",
  fn=lambda: n_rowtype("paraphrase_execute", "novel_benign_execute", "trained_execute"), kind="count", unit="rows")
A("c.data.taskfrac_min", "0.45", "sections/appendix.tex:136", "lowest harm fraction of any nearest task (construction labels)", CODE_BUILD, [BUILD],
  "min of build_summary.json structural.I1.task_harm_fraction", fn=lambda: min(js(BUILD)["structural"]["I1"]["task_harm_fraction"].values()), kind="measured")
A("c.data.taskfrac_max", "0.55", "sections/appendix.tex:136", "highest harm fraction of any nearest task", CODE_BUILD, [BUILD],
  "max of build_summary.json structural.I1.task_harm_fraction", fn=lambda: max(js(BUILD)["structural"]["I1"]["task_harm_fraction"].values()), kind="measured")
A("c.data.feat_auroc_min", "0.58", "sections/appendix.tex:136", "non-meaning features -> label, 5-fold AUROC (non-trained-instruction rows)", "experiments/4_parent_taught_verdict/review_training_rows.py",
  [AUDIT], "parse '[audit 11] ... AUROC on non-trained-instruction rows: x' from audit_lexical_shortcuts.txt",
  fn=lambda: float(grep1(AUDIT, r"AUROC on non-trained-instruction rows: ([\d.]+)")), kind="log")
A("c.data.feat_auroc_max", "0.59", "sections/appendix.tex:136", "non-meaning features -> label, 5-fold AUROC (corpus rows)", "experiments/4_parent_taught_verdict/review_training_rows.py",
  [AUDIT], "parse '[audit 11] ... AUROC on corpus rows: x'", fn=lambda: float(grep1(AUDIT, r"AUROC on corpus rows: ([\d.]+)")), kind="log")
A("c.data.bow_fa", "10%", "sections/appendix.tex:137", "calibration false-alarm rate of the bag-of-words threshold", "experiments/4_parent_taught_verdict/review_training_rows.py", [AUDIT],
  "parse 'theta at N% calibration FA' from the audit's bag-of-words line", fn=lambda: float(grep1(AUDIT, r"bag-of-words .*theta at (\d+)% calibration FA")),
  kind="log", unit="percent")
for cid, v, key in (("nt", "0.67", "HARM_HELDOUT"), ("nh", "0.50", "G11/HARM_NEW"), ("bt", "0.07", "TWIN_BENIGN")):
    A(f"c.data.bow_{cid}", v, "sections/appendix.tex:137", f"bag-of-words baseline hold rate on {key}", "experiments/4_parent_taught_verdict/review_training_rows.py", [AUDIT],
      f"parse '{key} x' from the audit's bag-of-words line", fn=lambda k=key: float(grep1(AUDIT, r"bag-of-words [^\n]*?(?<![/A-Z_])" + re.escape(k) + r" ([\d.]+)")),
      kind="log", note="the logistic model is not refitted here (audit output only)")
A("c.data.nt_shared", "3", "sections/appendix.tex:138", "never-trained commands sharing a non-scene content word with a harmful training row", "", [POOLS, ROWS_C],
  "words = [a-z]+ tokens; drop function words (incl. until, can), SCENE_NOUNS words and the object-part words middle, top, box; count harm_heldout sentences whose remaining words meet the words of any train_rows_construction_labels.jsonl row with harm == 1 (all hits share 'push')",
  fn=lambda: len(nt_content_overlap()) if all("push" in s for s in nt_content_overlap()) else -1, kind="measured", unit="commands",
  note="reconstruction: the log (2026-09-22 G18 reading) states the result but names no script; the stop-word list is ours")
A("c.data.nt_of", "24", "sections/appendix.tex:138", "never-trained commands", "experiments/3_refusal_examples/build_free_install_data.py", [POOLS], "len(pools.json harm_heldout)",
  fn=lambda: len(js(POOLS)["harm_heldout"]), kind="count", unit="commands")

# appendix D: sets
for cid, v, pool in (("nt", "24", "HARM_HELDOUT"), ("htr", "38", "HARM_TRAIN"), ("cmd15", "15", "DEMO_HARM"), ("th", "15", "TWIN_HARM"),
                     ("tb", "15", "TWIN_BENIGN"), ("tr", "10", "TRAINED"), ("para", "40", "BENIGN_PARA_HELDOUT"),
                     ("ood", "17", "OOD_BENIGN"), ("lex", "10", "LEXICAL")):
    A(f"c.sets.{cid}", v, "sections/appendix.tex:145", f"selection-era first-call pool size: {pool}", S_STAGE0, [S0["round1"]],
      f"distinct (set, idx) of pool {pool} in base stage-0 rows", fn=lambda p=pool: n_sent(p), kind="count", unit="commands")
A("c.sets.confirm_n", "50", "sections/appendix.tex:146", "confirmation set size", "experiments/3_refusal_examples/confirmation_set_gates.py", [G11], "rows of data/confirm_g11.jsonl",
  fn=lambda: len(jl(G11)), kind="count", unit="sentences")
for cid, v, pool in (("nh", "16", "HARM_NEW"), ("thn", "12", "TWIN_HARM_NEW"), ("tbn", "12", "TWIN_BENIGN_NEW"), ("lexn", "10", "LEXICAL_NEW")):
    A(f"c.sets.{cid}", v, "sections/appendix.tex:146", f"confirmation-set pool size: {pool}", S_STAGE0, [S0["round1"]],
      f"distinct (set, idx) of pool {pool} in base stage-0 rows", fn=lambda p=pool: n_sent(p), kind="count", unit="commands")
for cid, v, pool in (("bb", "14", "BENIGN_BLIND"), ("lb", "12", "LEXICAL_BLIND")):
    A(f"c.sets.{cid}", v, "sections/appendix.tex:147", f"blind set pool size: {pool}", S_STAGE0, [S0["round1"], G13],
      f"distinct (set, idx) of pool {pool} in base stage-0 rows; equals rows of confirm_g13 with that pool",
      fn=lambda p=pool: all_equal([n_sent(p), sum(1 for r in jl(G13) if r.get("pool") == p)]), kind="count", unit="commands")
A("c.sets.fresh_n", "87", "sections/appendix.tex:148", "fresh set size", CODE_G17, [G17], "rows of data/confirm_g17.jsonl",
  fn=lambda: len(jl(G17)), kind="count", unit="commands")
A("c.sets.rule5", "5%", "sections/appendix.tex:148", "the 5% rule", CODE_G17, [CODE_G17], "100 x (1 - OFFSET_Q['b95'])",
  fn=lambda: 100 * (1 - float(grep1(CODE_G17, r'"b95": ([\d.]+)'))), kind="config", unit="percent")
A("c.sets.calib_n", "180", "sections/appendix.tex:149", "benign calibration commands", "experiments/4_parent_taught_verdict/calibrate_verdict_offset.py", [CAL16], "calib_g16[I4].n / 2 (two frames each)",
  fn=lambda: int(calib()["n"]) // 2, kind="count", unit="commands")
A("c.sets.calib_overlap", "74", "sections/appendix.tex:149", "calibration commands that occur in the install data", "experiments/4_parent_taught_verdict/calibrate_verdict_offset.py", [CAL16],
  "calib_g16[I4].train_overlap.sentences", fn=lambda: int(calib()["train_overlap"]["sentences"]), kind="count", unit="commands")
A("c.sets.calib_used", "106", "sections/appendix.tex:149", "calibration commands used for b", "experiments/4_parent_taught_verdict/calibrate_verdict_offset.py", [CAL16],
  "calib_g16[I4].n_b / 2", fn=lambda: int(calib()["n_b"]) // 2, kind="count", unit="commands")

# appendix F: install rules
for cid, v, name in (("nt", "0.75", "G17_SIM_HARM_REFUSE_MIN"), ("bt", "0.20", "G17_SIM_TWIN_BENIGN_REFUSE_MAX"),
                     ("lib", "0.80", "G17_SIM_LIBERO_SUCCESS_MIN"), ("para", "0.10", "G17_SIM_PARA_OVERREFUSE_MAX")):
    A(f"c.prereg.sim_{cid}", v, "sections/appendix.tex:175", f"simulator claim rule {name}", CODE_F2, [CODE_F2],
      f"{name} parsed from experiments/5_simulation/eval_in_simulator.py", fn=lambda n=name: float(grep1(CODE_F2, rf"^{n} = ([\d.]+)", flags=re.M)), kind="config")
A("c.prereg.r3_rule10", "10%", "sections/appendix.tex:177", "round 3 offset rule", "experiments/4_parent_taught_verdict/calibrate_verdict_offset.py", [CAL16],
  "100 x (1 - calib_g16[I4].quantile)", fn=lambda: 100 * (1 - float(calib()["quantile"])), kind="config", unit="percent")
A("c.prereg.r3_one", "one", "sections/appendix.tex:177", "round-3 candidates passing the first-call rules", SCRIPT_OF[S0["round3"]], [S0["round3"], CODE_STAGE0],
  "count of b90 arms meeting the three rules", fn=lambda: word(len(b90_passing())), kind="measured", unit="arms")

# appendix G: judge prompt
A("c.judge.scale_min", "0", "sections/appendix.tex:199", "lowest grade of the graded judge prompt", CODE_JUDGE, [CODE_JUDGE],
  "parse 'scale from N (completely safe)' from GRADED in experiments/1_presence/parent_harm_judge.py",
  fn=lambda: int(grep1(CODE_JUDGE, r"scale from (\d+) \(completely safe\)")), kind="config", unit="grade")
A("c.judge.scale_max", "10", "sections/appendix.tex:199", "highest grade of the graded judge prompt", CODE_JUDGE, [CODE_JUDGE],
  "parse 'to N (very harmful)' from GRADED", fn=lambda: int(grep1(CODE_JUDGE, r"to (\d+) \(very harmful\)")), kind="config", unit="grade")

CLAIMS = C

# ---------------------------------------------------------------- non-claims on my lines


def _notation_ignores():
    """b_{90} / b_{95} subscripts are offset names, not numbers; generated from the tex so every occurrence counts."""
    files = {"sections/ptv.tex": [37, 95, 101, 102, 116], "tables/ablation.tex": [4, 15, 18, 20],
             "sections/limitations.tex": [11, 14],
             "sections/appendix.tex": [26, 29, 46, 62, 63, 66, 67, 69, 71, 73, 75, 77, 80, 81, 83, 84, 86, 87]}
    out = []
    for rel, lines in files.items():
        text = (ROOT / ARXIV / rel).read_text().splitlines()
        for ln in lines:
            for m in re.finditer(r"b_\{(9[05])\}", text[ln - 1]):
                out.append((rel, ln, m.group(1), "offset name b_" + m.group(1) + " (notation)"))
    return out


IGNORE = [   # tex coverage only; the b_90 / b_95 notation entries are generated from the tex in the private build
    ("sections/ptv.tex", 24, "1", "the constant 1 in the cosine distance 1 - cos (formula)"),
    ("sections/ptv.tex", 116, "0", "'b = 0' marks the argmax point (notation)"),
    ("sections/ptv.tex", 116, "0", "'seed 0' curve label (seed id, recorded in c.search.seed0)"),
    ("sections/ptv.tex", 116, "2", "'seed 2' curve label (seed id)"),
    ("sections/ptv.tex", 116, "1", "'seed 1' label (seed id)"),
    ("sections/appendix.tex", 10, "1", "round numbers 'rounds 1 to 5' (section labels)"),
    ("sections/appendix.tex", 10, "5", "round numbers 'rounds 1 to 5' (section labels)"),
    ("sections/appendix.tex", 13, "1", "round label"),
    ("sections/appendix.tex", 18, "2", "round label"),
    ("sections/appendix.tex", 22, "3", "round label"),
    ("sections/appendix.tex", 28, "4", "round label"),
    ("sections/appendix.tex", 37, "1", "'Seed 1' arm label"),
    ("sections/appendix.tex", 46, "0", "'argmax (b=0)' notation"),
    ("sections/appendix.tex", 176, "1", "round label"),
    ("sections/appendix.tex", 176, "2", "round label"),
    ("sections/appendix.tex", 177, "3", "round label"),
]
