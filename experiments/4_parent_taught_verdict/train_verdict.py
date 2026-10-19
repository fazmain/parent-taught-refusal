#!/usr/bin/env python3
"""Round 6 of the install search (pre-registered; internally "G19", the name kept in file names such as
stage0_g19.jsonl and calib_g19.json): the REPORTING set.

Round numbers used below, as in the file names, flags and constants: round 1 = "G14" (training rows and the
first-call rules), round 2 = "G15" (the I4 recipe; its adapters are `--adapters-g15`), round 3 = "G16" (the
calibrated offset rule; `--calib-g16`), round 4 = "G17" (the b_95 rule and the fresh confirmation set;
`--fresh-g17`), round 5 = "G18" (seeds and replay doubled; `--adapters-g18`, `--calib-g18`, `--fresh-g18`). "G11" is
the confirmation set and "G13" the blind benign set; the calibration prefills are those of the probe-gate decider
("G4", Appendix B).

The 310 episodes of the parent-gated reference's plan on init states 10-14 were used to pick among four verdict
installs, so they are the SELECTION set. This round re-runs the SAME plan, the same sentences and the same four rules
on init states 15-19, which no model has ever run, for six arms: the parent-taught verdict (the I4 recipe) at three
seeds, the replay-doubling arm, the same seed-2 adapter at the ARGMAX (the calibration ablation), and one newly
trained arm.

  I11  NEW, trained here: the I1 rows (`train_verdict.jsonl`, CONSTRUCTION verdict labels) + the I3 alignment term
       (weight 1.0, targets bound to `train_free_align.jsonl`), seed 0, 1,052 steps, lr 1e-4. Everything else is
       I4's recipe, so I11 - I4 isolates the LABELLER (construction rules vs the parent judge) from the
       representation target. Its offline gates (first call + G11 + the round-4 fresh rules) are REPORTED, not a
       gate: the pre-registration simulates it either way.

This file does for I11 exactly what `train_verdict_seeds.py` did for its four arms - the same calibration rule
(round 3's `calibrate` on the 360 probe-gate prefills, the quantile over only the 212 whose sentence is not a v4
training text), the same pre-registered offset b_95 (`fresh_confirmation.offsets_from_calib`, which re-derives
round 3's own b_90 and asserts it equals the b the calibration fitted), the same first-call jobs and the same fresh
set - and then writes ONE verdict gate file per reporting arm, including the two that need no new training (round 2's
I4, round 5's I4s1 / I4s2 / I9) and the argmax gate of I4s2.

  check   CPU. The approved I1 rows: sha1 against the round-1 approval list, the alignment binding (the rows with an
          align_key match train_free_align.jsonl key for key, as I4's --align-rows-ref asserts at training time),
          the step count and the 212 non-training calibration prefills.      -> <out>/i1data_meta_g19.json
  stage0  GPU. I11: load (train_meta checked against the recipe and the approved row hash), calibrate, then the
          round-1 first-call jobs (probe-gate pools + G11 + G13, init 10 / 11) at the argmax AND at b_95.
                                       -> <out>/stage0/{stage0_g19.jsonl, offset_log_g19.jsonl, calib_g19.json,
                                                        stage0_g19.json, stage0_table_g19.txt, g19_stage0_go.json}
  fresh   GPU. I11 on the round-4 fresh confirmation set (87 sentences x 2 frames) at the argmax AND at b_95, the
          pre-registered round-4 fresh rules, then g19_go.json - REPORTED, never a gate.
                                       -> <out>/fresh/{fresh_g19.jsonl, offsets_g19.json, summary_g19.json,
                                                       table_g19.txt, g19_go.json}
  gates   CPU. One verdict gate file per reporting arm, each at its own calibrated offset, recomputed from the
          calibration record of the run that trained it (round 3 for I4, round 5 for I4s1 / I4s2 / I9, this run for
          I11) and checked by eval_in_simulator's own loader.
                                       -> <out>/gates/{verdict_gate_*.json, g19_gates.json, gates_table_g19.txt}
  frames  GPU (LIBERO, no policy call). The home frames of the reporting init states 15-19 for all ten trained
          tasks, rendered by `verify_parent_gated_judge.ensure_eval_frames` into the folder the offline arms read.
                                       -> <frames-out>/home_frames_eval/t*_i1[5-9].png, <out>/frames_g19.json

  python3 experiments/4_parent_taught_verdict/train_verdict.py --self-test                                  # CPU, no GPU
  python3 experiments/4_parent_taught_verdict/train_verdict.py --stage check  --out results/g19             # CPU
  python3 experiments/4_parent_taught_verdict/train_verdict.py --stage stage0 --out results/g19 [--smoke]   # GPU
  python3 experiments/4_parent_taught_verdict/train_verdict.py --stage fresh  --out results/g19 [--smoke]   # GPU (exit 0 either way)
  python3 experiments/4_parent_taught_verdict/train_verdict.py --stage gates  --out results/g19             # CPU
  python3 experiments/4_parent_taught_verdict/train_verdict.py --stage frames --out results/g19             # GPU (LIBERO)
  python3 experiments/4_parent_taught_verdict/train_verdict.py --self-check-cpu results/g19                 # CPU, on the GPU run's outputs

RULES are round 1's (first call + the G11 confirmation, G13 reported) and round 4's (the fresh set), unchanged and
applied at
b_95. No log line printed here contains command text: sentences are identified by set, pool, index and hash only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from refusal.paths import DATA_DIR, RESULTS_DIR, display, repo_path, add_experiment_dirs  # noqa: E402
add_experiment_dirs()
from train_install_lora import same_rows_as_ref  # noqa: E402
from build_training_rows import GO, STOP, VERDICT_IDS_REF, cpu_shim, verdict_token_ids  # noqa: E402
from first_call_eval import EVAL_INITS, apply_rules, build_items  # noqa: E402
from first_call_eval import check_rows as check_stage0_rows  # noqa: E402
from first_call_eval import jobs_of, load_json, model_block, summarize, verdict_stats  # noqa: E402
from train_verdict_alignment import COMPARE, LORA_ARGS, check_meta, load_arm  # noqa: E402
from calibrate_verdict_offset import QUANTILE as CALIB_QUANTILE  # noqa: E402
from calibrate_verdict_offset import SOURCES as G16_SOURCES  # noqa: E402
from calibrate_verdict_offset import N_B_FULL, calib_line, calibration_jobs, decision_ids, read_jsonl  # noqa: E402
from calibrate_verdict_offset import offset_stats as stage0_offset_stats  # noqa: E402
from calibrate_verdict_offset import run_stage0  # noqa: E402
from fresh_confirmation import CONFIRM, CONFIRM_SHA256, FRESH_RULES, OFFSET_Q, POOLS, RULE_OFFSET  # noqa: E402
from fresh_confirmation import build_items as build_fresh_items  # noqa: E402
from fresh_confirmation import check_rows as check_fresh_rows  # noqa: E402
from fresh_confirmation import jobs_of as fresh_jobs_of  # noqa: E402
from fresh_confirmation import offset_stats as fresh_offset_stats  # noqa: E402
from fresh_confirmation import rule_text as fresh_rule_text  # noqa: E402
from fresh_confirmation import (arm_name, gate_path, key_of, offsets_from_calib, pool_rates,  # noqa: E402
                                   run_fresh, write_gate)
from train_verdict_seeds import SOURCES as G18_SOURCES  # noqa: E402
from probe_gate_first_call import Report, file_line, sha  # noqa: E402

EXP = "g19_install"
ARMS = ("I11",)                                # the one arm trained here
I11_ROWS = "train_verdict.jsonl"               # the round-1 I1 rows: verdict format, CONSTRUCTION labels
ALIGN_ROWS = "train_free_align.jsonl"          # the file the I3 alignment targets are bound to
APPROVED_I1_SHA1 = "a1dcf776d937966ec261024c7b36d4742828b52d"      # round-1 approval list
APPROVED_ALIGN_SHA1 = "5ef2b12747000862e1fd18d98c45ff1dcfa1439a"   # round-1 approval list
SOURCES = {"I11": {"name": "verdict_construct_align", "rows": I11_ROWS, "align_weight": 1.0,
                   "align_ref": ALIGN_ROWS, "restore": None, "lora_args": dict(LORA_ARGS),
                   "text": "I1 rows (construction labels) + I3 alignment term, weight 1.0"}}
MODELS: dict = {"I11": ("I11", False), arm_name("I11"): ("I11", True)}
OFFSETS = ("b0", RULE_OFFSET)                  # the argmax row and the pre-registered one
REPORT_OFFSET = "b97"
CALIB_SOURCE = "experiments/g19_install.py --stage stage0, calib_g19.json (G16's rule, this pod's adapter)"

# ------------------------------------------------------------------ the reporting arms and where each one comes from
# (arm name in the simulator, calibration source, the run that trained the adapter, the offset it runs at).
# The run keys are internal run labels kept as data keys: "pod15" = round 2, "pod18" = round 5, "pod19" = this round.
ARGMAX_ARM = "I4s2argmax"
REPORT_GATES = (("I4b95", "I4", "pod15", RULE_OFFSET),
                ("I4s1b95", "I4s1", "pod18", RULE_OFFSET),
                ("I4s2b95", "I4s2", "pod18", RULE_OFFSET),
                ("I9b95", "I9", "pod18", RULE_OFFSET),
                (ARGMAX_ARM, "I4s2", "pod18", "b0"),
                (arm_name("I11"), "I11", "pod19", RULE_OFFSET))
RECIPES = {"I4": G16_SOURCES["I4"], **{k: G18_SOURCES[k] for k in ("I4s1", "I4s2", "I9")}, **SOURCES}
I4_B95 = 3.500                                 # round 3's calibration; recomputed and asserted, never assumed
B_TOL = 1e-6
# the reporting init states (eval_in_simulator carries the same three constants; the self-test asserts they agree)
REPORT_GATE_INITS = (15, 16, 17, 18, 19)
REPORT_PAIR_INITS = (15, 16)
REPORT_ONE_INIT = (15,)
REPORT_PLAN_N = 310
REPORT_PLAN_PER_TEST = {"gate": 50, "nonattempt": 96, "twins": 60, "overrefusal_para": 80, "abort": 24}
SHIPPED_TRAIN_ROWS = RESULTS_DIR / "raw/pod14/g14_v4/rows/train_verdict.jsonl"
SHIPPED_ALIGN_ROWS = RESULTS_DIR / "raw/pod14/g14_v4/rows/train_free_align.jsonl"
SHIPPED_ROWS_SUMMARY = RESULTS_DIR / "raw/pod14/g14_v4/rows/build_summary_g14.json"
SHIPPED_EVAL_FRAMES = RESULTS_DIR / "raw/pod8/g1/home_frames_eval"
SHIPPED_CALIB_G16 = RESULTS_DIR / "raw/pod16/g16/stage0/calib_g16.json"
SHIPPED_CALIB_G18 = RESULTS_DIR / "raw/pod18/g18/stage0/calib_g18.json"
SHIPPED_FRESH_G17 = RESULTS_DIR / "raw/pod17/g17/fresh/summary_g17.json"
SHIPPED_FRESH_G18 = RESULTS_DIR / "raw/pod18/g18/fresh/summary_g18.json"
SHIPPED_ADAPTERS_G15 = RESULTS_DIR / "raw/pod15/g15/adapters"
SHIPPED_ADAPTERS_G18 = RESULTS_DIR / "raw/pod18/g18/adapters"


def paths(out) -> dict:
    o = Path(out)
    return {"root": o, "adapters": o / "adapters", "meta": o / "i1data_meta_g19.json",
            "dir": o / "stage0", "rows": o / "stage0/stage0_g19.jsonl", "log": o / "stage0/offset_log_g19.jsonl",
            "calib": o / "stage0/calib_g19.json", "summary": o / "stage0/stage0_g19.json",
            "table": o / "stage0/stage0_table_g19.txt", "go0": o / "stage0/g19_stage0_go.json",
            "fdir": o / "fresh", "frows": o / "fresh/fresh_g19.jsonl", "foffsets": o / "fresh/offsets_g19.json",
            "fcalib": o / "fresh/calib_ref_g19.json", "fsummary": o / "fresh/summary_g19.json",
            "ftable": o / "fresh/table_g19.txt", "go": o / "fresh/g19_go.json",
            "gdir": o / "gates", "gates": o / "gates/g19_gates.json", "gtable": o / "gates/gates_table_g19.txt",
            "frames": o / "frames_g19.json"}


def stage0_paths(P: dict) -> dict:
    return {"dir": P["dir"], "rows": P["rows"], "calib": P["calib"], "log": P["log"]}


def fresh_paths(P: dict) -> dict:
    return {"dir": P["fdir"], "rows": P["frows"]}


def fresh_pairs() -> list[tuple[str, str]]:
    return [(arm, o) for arm in ARMS for o in OFFSETS]


def b95_of(src: str, c: dict) -> float:
    """The offset every arm is evaluated at: b_95 over the 212 non-training calibration prefills, derived from the
    decision logits the calibration rule stored (`offsets_from_calib` re-checks that record's own b_90 against the
    b the calibration fitted, so the two cannot drift apart)."""
    return float(offsets_from_calib({src: c}, sources=(src,))[src][RULE_OFFSET])


def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def steps_of(a) -> dict:
    base = int(load_json(a.rows_summary)["steps_all_arms"])
    if base <= 0:
        raise SystemExit(f"{a.rows_summary}: steps_all_arms is {base}")
    return {arm: base for arm in ARMS}


def checked_rows_sha1(a, P: dict) -> str:
    """The row hash the check stage recorded, which the adapter's train_meta must carry. On a full run that hash
    is the approved round-1 one (the check stage refuses anything else); a smoke run carries the smoke bytes."""
    doc = load_json(P["meta"])
    if bool(doc["smoke"]) != bool(a.smoke):
        raise SystemExit(f"{P['meta']}: smoke {doc['smoke']}, this run has smoke {a.smoke}")
    if not a.smoke and doc["rows_sha1"] != APPROVED_I1_SHA1:
        raise SystemExit(f"{P['meta']}: the checked rows are {doc['rows_sha1'][:16]}, not the pod-14 approved "
                         f"{APPROVED_I1_SHA1[:16]}")
    return str(doc["rows_sha1"])


def check_train_meta(arm: str, meta: dict, steps: int, checkpoint: str, rows_sha1: str) -> None:
    """`train_verdict_alignment.check_meta` (row file, step count, LoRA arguments, alignment term and reference, no restore)
    plus the row hash: I11 must have trained the bytes the check stage hashed (which on a full run it also
    asserted to be the approved round-1 I1 file), not a file derived from them."""
    check_meta(arm, meta, steps, checkpoint, None, SOURCES[arm])
    bad = []
    if meta.get("rows_sha1") != rows_sha1:
        bad.append(f"rows sha1 {str(meta.get('rows_sha1'))[:16]} (want {rows_sha1[:16]})")
    if meta.get("rows_derived_from") is not None:
        bad.append("a rows_derived_from record on an arm trained on the approved file itself")
    if bad:
        raise SystemExit(f"{arm}: train_meta.json is not the pre-registered recipe: " + "; ".join(bad))


# --------------------------------------------------------------------------------- check (CPU, before training)

def stage_check(a) -> int:
    P = paths(a.out)
    src, ref = Path(a.train_rows), Path(a.align_rows)
    src_sha1, ref_sha1 = sha1_bytes(src.read_bytes()), sha1_bytes(ref.read_bytes())
    if not a.smoke:
        for p, got, want in ((src, src_sha1, APPROVED_I1_SHA1), (ref, ref_sha1, APPROVED_ALIGN_SHA1)):
            if got != want:
                raise SystemExit(f"{p} sha1 {got} != the pod-14 approved {want}")
    rows, ref_rows = read_jsonl(src), read_jsonl(ref)
    if Path(src).name != I11_ROWS and not a.smoke:
        raise SystemExit(f"--train-rows is {Path(src).name}; I11 trains {I11_ROWS} (the CONSTRUCTION labels)")
    same_rows_as_ref(rows, ref_rows, str(src), str(ref))      # the I3 targets must still bind, exactly as I4's do
    steps = steps_of(a)["I11"]
    if not a.smoke and steps != 1052:
        raise SystemExit(f"{a.rows_summary}: steps_all_arms is {steps}, the pre-registration says 1052")
    train_texts = {r["raw_instruction"] for r in rows}
    if not train_texts:
        raise SystemExit(f"{src}: no training rows")
    cal_jobs = calibration_jobs(Path(a.f1_data), Path(a.novel_benign), a.smoke)
    n_b = sum(j["text"] not in train_texts for j in cal_jobs)
    if not a.smoke and n_b != N_B_FULL:
        raise SystemExit(f"{n_b} calibration prefills outside the I1 training texts; the G16 NOTE fixes {N_B_FULL}")
    verdicts = Counter(r["verdict"] for r in rows)
    labels = Counter(r.get("label_source") for r in rows)
    doc = {"exp": f"{EXP}/check", "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "smoke": bool(a.smoke),
           "rows": str(src), "rows_sha1": src_sha1, "approved_sha1": APPROVED_I1_SHA1, "n_rows": len(rows),
           "align_rows_ref": str(ref), "align_rows_ref_sha1": ref_sha1, "approved_align_sha1": APPROVED_ALIGN_SHA1,
           "n_align_keys": len({r["align_key"] for r in rows if r.get("align_key")}),
           "row_types": dict(Counter(r["row_type"] for r in rows)), "verdicts": dict(verdicts),
           "label_sources": dict(labels), "steps": steps, "recipe": SOURCES["I11"],
           "n_calibration_prefills": len(cal_jobs), "n_b": n_b, "n_train_texts": len(train_texts)}
    P["root"].mkdir(parents=True, exist_ok=True)
    P["meta"].write_text(json.dumps(doc, indent=1))
    rep = Report()
    rep(f"G19 I11 data: {doc['n_rows']} approved I1 rows (sha1 {src_sha1[:16]} = the pod-14 approved file), row "
        f"types {doc['row_types']}; verdict labels {dict(verdicts)} (CONSTRUCTION labels, the I1 labeller)")
    rep(f"the {doc['n_align_keys']} rows with an align_key match {ref.name} (sha1 {ref_sha1[:16]}) key for key, so "
        f"the I3 targets stay bound to that file exactly as they do for I4")
    rep(f"I11 OPTIMIZER STEPS = {steps} (the pre-registered count, the same as I4); {n_b} of "
        f"{doc['n_calibration_prefills']} calibration prefills are outside the {len(train_texts)} I1 training texts")
    print("wrote\n" + file_line(P["meta"]), flush=True)
    return 0


# --------------------------------------------------------------------------------- stage 0

def load_inputs(a) -> tuple:
    items = build_items(Path(a.f1_data), Path(a.twins), Path(a.demo_prompts), a.smoke)
    cal_jobs = calibration_jobs(Path(a.f1_data), Path(a.novel_benign), a.smoke)
    train_texts = {r["raw_instruction"] for r in read_jsonl(a.train_rows)}
    if not train_texts:
        raise SystemExit(f"{a.train_rows}: no training rows")
    n_b = sum(j["text"] not in train_texts for j in cal_jobs)
    if not a.smoke and n_b != N_B_FULL:
        raise SystemExit(f"{n_b} calibration prefills outside the I1 training texts; the G16 NOTE fixes {N_B_FULL}")
    return items, cal_jobs, train_texts


def evaluate0(rows: list[dict], log: list[dict], calib: dict, items: list[dict], hold_ids=None) -> tuple:
    check_stage0_rows(rows, items, list(MODELS), VERDICT_IDS_REF, hold_ids=hold_ids)
    summ = summarize(rows, tuple(MODELS))
    ost = stage0_offset_stats(rows, log, calib, models=MODELS, b_of=b95_of)
    offsets = offsets_from_calib(calib, sources=ARMS)
    go = {arm: int(summ[arm_name(arm)]["eligible"]) for arm in ARMS}
    return summ, ost, offsets, go, [arm for arm in ARMS if go[arm]]


def offset_line(o: dict) -> str:
    return (f"  offsets (same {o['n_b']} prefills, same stored logits): b95 {o[RULE_OFFSET]:+.3f} (STOP rate after "
            f"it {o['stop_rate_b95']:.3f}), b97 {o[REPORT_OFFSET]:+.3f} ({o['stop_rate_b97']:.3f}, report only); "
            f"b90 recomputed {o['b90_check']:+.3f} = the fitted {o['b90_stored']:+.3f}")


def table0_lines(summ: dict, ost: dict, calib: dict, offsets: dict, steps: dict, smoke: bool) -> list[str]:
    L = [f"G19 stage 0: first policy call, RAW command, no gate / tag / parent, the G14 stage-0 jobs (hold = "
         f"f1_token_check.classify of the decoded first chunk, per sentence averaged over the init "
         f"{EVAL_INITS[0]} and {EVAL_INITS[1]} frames); the new labeller-ablation arm I11 at its argmax and at its "
         f"own calibrated b95" + ("; SMOKE (not a result)" if smoke else ""),
         "", "NOTE: these offline numbers are REPORTED for I11; the G19 pre-registration simulates it on the "
             "reporting set whatever they say."]
    for m, (src, cal) in MODELS.items():
        s = summ[m]
        rc = SOURCES[src]
        L += ["", f"== {m} ({rc['text']}{', calibrated offset b95' if cal else ', argmax'}; adapter {rc['name']}, "
                  f"rows {rc['rows']}, {steps[src]} steps, "
                  f"{', '.join(f'{k} {v}' for k, v in rc['lora_args'].items())}): {s['n_calls']} calls, invalid "
                  f"rate {s['invalid_rate']:.3f}"]
        L += model_block(s, True)
        if cal:
            o, st = offsets[src], ost[m]
            L.append(calib_line(src, calib[src]))
            L.append(offset_line(o))
            L.append(f"  offset in evaluation: applied on {st['n_open']}/{st['n']} generations (first token '['), "
                     f"decision between [STOP] and [GO] on {st['n_top2']}; GO -> STOP flips {st['flips']} (harm "
                     f"{st['flips_harm']}, benign {st['flips_benign']})")
        L.append(f"G19 STAGE 0 {m}: rules {'PASS' if s['rules_pass'] else 'FAIL'}, G11 confirmation "
                 f"{'PASS' if s['confirm_pass'] else 'FAIL'}, G13 target "
                 f"{'MET' if s['g13_target_met'] else 'MISSED'} -> "
                 + ("STAGE-0 PASS" if s["eligible"] else "not eligible") + " (reported, not a gate)")
    L += ["", "== hold rate on the rule, confirmation and target pools",
          f"  {'model':<10s}{'recipe':<52s}" + "".join(f"{p:>16s}" for _, p in COMPARE) + f"{'invalid':>9s}"]
    for m, (src, cal) in MODELS.items():
        s = summ[m]
        text = SOURCES[src]["text"] + (" + b95" if cal else " (argmax)")
        vals = [s["sets"][st].get(pool, {}).get("hold", float("nan")) for st, pool in COMPARE]
        L.append(f"  {m:<10s}{text[:52]:<52s}" + "".join(f"{v:16.3f}" for v in vals) + f"{s['invalid_rate']:9.3f}")
    elig = [arm for arm in ARMS if summ[arm_name(arm)]["eligible"]]
    L += ["", f"G19 STAGE 0: arms passing the rules and the G11 confirmation at b95 {elig} (reported)" if elig else
          "G19 STAGE 0: I11 does not pass the rules and the G11 confirmation at b95 (reported; it is simulated "
          "anyway, as pre-registered)"]
    return L


def finish0(a, items: list[dict], steps: dict, append_results: bool) -> int:
    P = paths(a.out)
    rows, log, calib = read_jsonl(P["rows"]), read_jsonl(P["log"]), load_json(P["calib"])
    summ, ost, offsets, go, elig = evaluate0(rows, log, calib, items)
    lines = table0_lines(summ, ost, calib, offsets, steps, a.smoke)
    Report()("\n".join(lines))
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    doc = {"exp": f"{EXP}/stage0", "ts": stamp, "smoke": bool(a.smoke), "arms": list(ARMS),
           "models": {m: {"source": s, "offset": (RULE_OFFSET if c else "b0"), "adapter": SOURCES[s]["name"],
                          "steps": steps[s]} for m, (s, c) in MODELS.items()},
           "sources": SOURCES, "steps": steps, "quantile_calibration": CALIB_QUANTILE,
           "quantile_rule": OFFSET_Q[RULE_OFFSET],
           "calibration": {s: {k: v for k, v in c.items() if k not in ("ids", "d", "in_b")}
                           for s, c in calib.items()},
           "offsets": offsets, "verdict_ids": VERDICT_IDS_REF, "n_items": len(items),
           "n_jobs": len(jobs_of(items)), "summary": summ, "offset": ost, "go": go, "eligible": elig,
           "gate": False}
    P["summary"].write_text(json.dumps(doc, indent=1, default=float))
    P["table"].write_text("\n".join(lines) + "\n")
    P["go0"].write_text(json.dumps({**go, "eligible": elig, "gate": False}, indent=1))
    if append_results and not a.smoke:
        with (RESULTS_DIR / "results.jsonl").open("a") as f:
            for m, s in summ.items():
                for st, pools in s["sets"].items():
                    for pool, v in pools.items():
                        f.write(json.dumps({"exp": f"{EXP}/stage0", "ts": stamp, "model": m,
                                            "adapter": SOURCES[MODELS[m][0]]["name"],
                                            "offset_b": offsets[MODELS[m][0]][RULE_OFFSET] if MODELS[m][1] else 0.0,
                                            "set": st, "pool": pool, **v}) + "\n")
    print("wrote\n" + "\n".join(file_line(P[k]) for k in ("rows", "log", "calib", "summary", "table", "go0")),
          flush=True)
    return 0


def forbid_parent_here() -> None:
    from single_model_judges import forbid_parent
    forbid_parent()


def stage_stage0(a) -> int:
    P = paths(a.out)
    rs = load_json(a.rows_summary)
    if bool(rs["smoke"]) != bool(a.smoke) or rs["stub_generator"]:
        raise SystemExit(f"{a.rows_summary}: smoke {rs['smoke']}, stub {rs['stub_generator']}; this run has smoke "
                         f"{a.smoke} and needs real rows")
    if rs["verdict_ids"] != VERDICT_IDS_REF:
        raise SystemExit(f"rows summary verdict ids {rs['verdict_ids']} != {VERDICT_IDS_REF}")
    stop_go = decision_ids(rs["verdict_ids"])
    steps = steps_of(a)
    rows_sha1 = checked_rows_sha1(a, paths(a.out))
    adapters = {arm: Path(a.adapters) / SOURCES[arm]["name"] for arm in ARMS}
    missing = [str(p) for p in adapters.values() if not (p / "adapter_config.json").exists()]
    if missing:
        raise SystemExit(f"adapters missing: {missing}")
    if a.smoke:
        print("[smoke] 2 sentences per pool, both frames; 10 calibration prefills. NOT results.", flush=True)
    items, cal_jobs, train_texts = load_inputs(a)
    print(f"[jobs] {len(jobs_of(items))} generations per model, {len(MODELS)} models ({len(ARMS)} arm x argmax + "
          f"b95); {len(cal_jobs)} calibration prefills; decision ids STOP {stop_go[0]} / GO {stop_go[1]}",
          flush=True)
    for arm in ARMS:
        print(f"[arm] {arm}: {SOURCES[arm]['text']}, adapter {SOURCES[arm]['name']}, rows {SOURCES[arm]['rows']}, "
              f"{steps[arm]} steps, {SOURCES[arm]['lora_args']}", flush=True)
    forbid_parent_here()
    rep = Report()

    def check_ids(policy):
        if decision_ids(verdict_token_ids(policy, rep)) != stop_go:
            raise SystemExit("the policy tokenizer's verdict ids differ from the approved rows")

    def factory(arm):
        check_train_meta(arm, load_json(adapters[arm] / "train_meta.json"), steps[arm], a.checkpoint, rows_sha1)
        return load_arm(a, arm, str(adapters[arm]), {"steps": steps[arm], "restore_doc": None, "restore": None},
                        rc=SOURCES[arm])

    run_stage0(a, items, cal_jobs, train_texts, factory, check_ids, stop_go, P=stage0_paths(P), sources=ARMS,
               models=MODELS, b_of=b95_of)
    return finish0(a, items, steps, append_results=True)


# --------------------------------------------------------------------------------- the fresh set

def summarize_fresh(rows: list[dict], offsets: dict, stop_go) -> dict:
    out = {}
    for src, offset in fresh_pairs():
        mr = [r for r in rows if r["model"] == src and r["offset"] == offset]
        res = {"source": src, "offset": offset, "b": float(offsets[src][offset]), "n_calls": len(mr),
               "invalid_rate": float(np.mean([r["cls"] == "invalid" for r in mr])) if mr else float("nan"),
               "pools": pool_rates(mr)}
        rules, ok = apply_rules(res["pools"], FRESH_RULES)
        res["rules"] = rules
        res["rules_pass"] = bool(ok)
        res["rules_apply"] = bool(offset == RULE_OFFSET)
        res["verdict"] = verdict_stats([{**r, "set": "g19"} for r in mr]) if mr else None
        res["offset_stats"] = fresh_offset_stats(mr, res["b"], stop_go[0]) if mr else None
        res["eligible"] = bool(res["rules_apply"] and ok)
        out[key_of(src, offset)] = res
    return out


def reference_row(path, key: str, label: str) -> dict | None:
    """One stored fresh row of an earlier run, printed beside I11's for comparison. Report only."""
    p = Path(path) if path else None
    if not p or not p.exists():
        return None
    s = load_json(p).get("summary", {}).get(key)
    return None if s is None else {"label": label, "b": s["b"], "pools": s["pools"],
                                   "rules_pass": s["rules_pass"], "file": str(p)}


def tablef_lines(summ: dict, offsets: dict, steps: dict, refs: list, smoke: bool) -> list[str]:
    L = [f"G19 fresh confirmation set ({CONFIRM}, sha256 {CONFIRM_SHA256[:16]}, the G17 set reused): first policy "
         f"call, RAW command, no gate / tag / parent; hold per sentence averaged over the init {EVAL_INITS[0]} and "
         f"{EVAL_INITS[1]} home frames of its nearest trained task"
         + ("; SMOKE (not a result)" if smoke else ""),
         "", "== hold rate per pool (one row per sentence, its two frames averaged first)",
         f"  {'model':<10s}{'offset':<8s}{'b':>8s}  {'recipe':<44s}" + "".join(f"{p:>20s}" for p in POOLS)
         + f"{'invalid':>9s}"]
    for src, offset in fresh_pairs():
        s = summ[key_of(src, offset)]
        vals = "".join(f"{s['pools'].get(p, {}).get('hold', float('nan')):20.3f}" for p in POOLS)
        L.append(f"  {src:<10s}{offset:<8s}{s['b']:+8.3f}  {SOURCES[src]['text'][:44]:<44s}{vals}"
                 f"{s['invalid_rate']:9.3f}")
    for ref in refs:
        vals = "".join(f"{ref['pools'].get(p, {}).get('hold', float('nan')):20.3f}" for p in POOLS)
        L.append(f"  {ref['label']:<10s}{RULE_OFFSET:<8s}{ref['b']:+8.3f}  {'earlier pod, report only':<44s}{vals}"
                 f"{float('nan'):9.3f}")
    if not refs:
        L.append("  (no earlier fresh row on this machine; --fresh-g17 / --fresh-g18 point at them)")
    L += ["", f"== the pre-registered G17 fresh rules, applied at {RULE_OFFSET} only (REPORTED for I11)"]
    for src in ARMS:
        s = summ[key_of(src, RULE_OFFSET)]
        L.append(f"-- {src} at {RULE_OFFSET} (b = {s['b']:+.3f}; {SOURCES[src]['text']}; adapter "
                 f"{SOURCES[src]['name']}, {steps[src]} steps)")
        L += [fresh_rule_text(r) for r in s["rules"]]
        o, v = s["offset_stats"] or {}, s["verdict"] or {}
        if o:
            L.append(f"  offset: applied on {o['n_open']}/{o['n']} generations (first token '['), decision between "
                     f"[STOP] and [GO] on {o['n_top2']}; GO -> STOP flips {o['flips']} (harm {o['flips_harm']}, "
                     f"benign {o['flips_benign']})")
        if v:
            L.append(f"  verdict: {STOP} {v['n_stop']}, {GO} {v['n_go']}, none {v['n_none']}; agreement (verdict "
                     f"{STOP} <=> behaviour hold) {v['agree']:.3f}")
        L.append(f"G19 FRESH {src}: rules {'PASS' if s['rules_pass'] else 'FAIL'} (reported; {src} is simulated on "
                 f"the reporting set either way)")
    return L


def build_go(a, summ0: dict, summf: dict, offsets: dict, steps: dict) -> dict:
    p = {}
    for arm in ARMS:
        s0, sf = summ0[arm_name(arm)], summf[key_of(arm, RULE_OFFSET)]
        p[arm] = {"stage0_rules": int(s0["rules_pass"]), "g11": int(s0["confirm_pass"]),
                  "g13": int(s0["g13_target_met"]), "fresh": int(sf["rules_pass"]),
                  "all": int(s0["rules_pass"] and s0["confirm_pass"] and sf["rules_pass"])}
    return {"exp": EXP, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "smoke": bool(a.smoke), "arms": list(ARMS),
            "pass": p, "offline_pass": [arm for arm in ARMS if p[arm]["all"]],
            "sim_arms": [arm_name(arm) for arm in ARMS],
            "gate": False, "gate_note": "REPORTED only: the G19 pre-registration simulates I11 on the reporting "
                                        "set whatever these offline rules say",
            "b": {arm: float(offsets[arm][RULE_OFFSET]) for arm in ARMS}, "quantile": OFFSET_Q[RULE_OFFSET],
            "steps": {arm: int(steps[arm]) for arm in ARMS}}


def go_lines(go: dict) -> list[str]:
    L = ["", "== G19 offline map for I11 (REPORTED, not a gate: the arm is simulated on the reporting set either "
             "way)",
         f"  {'arm':<8s}{'stage0':>8s}{'G11':>6s}{'G13':>6s}{'fresh':>7s}{'b95':>9s}{'steps':>7s}   simulator"]
    for arm in ARMS:
        p = go["pass"][arm]
        L.append(f"  {arm:<8s}{'PASS' if p['stage0_rules'] else 'FAIL':>8s}{'PASS' if p['g11'] else 'FAIL':>6s}"
                 f"{'MET' if p['g13'] else 'MISS':>6s}{'PASS' if p['fresh'] else 'FAIL':>7s}"
                 f"{go['b'][arm]:+9.3f}{go['steps'][arm]:7d}   YES ({arm_name(arm)}, pre-registered)")
    L.append("")
    L.append(f"G19: I11 passes every offline bar: {bool(go['offline_pass'])}; simulator arm "
             f"{go['sim_arms']} either way")
    return L


def finishf(a, items: list[dict], offsets: dict, stop_go, steps: dict, calib_sha: str,
            append_results: bool) -> int:
    P = paths(a.out)
    rows = read_jsonl(P["frows"])
    check_fresh_rows(rows, items, offsets, pairs=fresh_pairs(), verdict=lambda _s: True)
    summ = summarize_fresh(rows, offsets, stop_go)
    summ0 = load_json(P["summary"])["summary"]
    refs = [r for r in (reference_row(a.fresh_g17, key_of("I4", RULE_OFFSET), "I4b95"),
                        reference_row(a.fresh_g18, key_of("I4s2", RULE_OFFSET), "I4s2b95"),
                        reference_row(a.fresh_g18, key_of("I9", RULE_OFFSET), "I9b95")) if r]
    lines = tablef_lines(summ, offsets, steps, refs, a.smoke)
    go = build_go(a, summ0, summ, offsets, steps)
    lines += go_lines(go)
    Report()("\n".join(lines))
    stamp = go["ts"]
    doc = {"exp": f"{EXP}/fresh", "ts": stamp, "smoke": bool(a.smoke), "confirm": CONFIRM,
           "confirm_sha256": CONFIRM_SHA256, "pools": list(POOLS), "arms": list(ARMS),
           "offsets_scored": list(OFFSETS), "n_items": len(items), "n_jobs": len(fresh_jobs_of(items)),
           "models": {key_of(s, o): {"source": s, "offset": o, "adapter": SOURCES[s]["name"]}
                      for s, o in fresh_pairs()},
           "sources": SOURCES, "offsets": offsets, "rules": list(FRESH_RULES), "rule_offset": RULE_OFFSET,
           "verdict_ids": VERDICT_IDS_REF, "steps": steps, "calib_sha256": calib_sha,
           "references": refs, "summary": summ, "go": go}
    P["foffsets"].write_text(json.dumps({"exp": f"{EXP}/fresh", "ts": stamp, "quantiles": OFFSET_Q,
                                         "check_quantile": CALIB_QUANTILE, "calib": str(a.calib),
                                         "calib_sha256": calib_sha, "offsets": offsets}, indent=1, default=float))
    P["fsummary"].write_text(json.dumps(doc, indent=1, default=float))
    P["ftable"].write_text("\n".join(lines) + "\n")
    P["go"].write_text(json.dumps(go, indent=1, default=float))
    if append_results and not a.smoke:
        with (RESULTS_DIR / "results.jsonl").open("a") as f:
            for key, s in summ.items():
                for pool, v in s["pools"].items():
                    f.write(json.dumps({"exp": f"{EXP}/fresh", "ts": stamp, "model": key, "source": s["source"],
                                        "offset": s["offset"], "offset_b": s["b"], "pool": pool, **v}) + "\n")
    print("wrote\n" + "\n".join(file_line(P[k]) for k in ("frows", "foffsets", "fcalib", "fsummary", "ftable",
                                                          "go")), flush=True)
    return 0


def stage_fresh(a) -> int:
    P = paths(a.out)
    rs = load_json(a.rows_summary)
    if rs["verdict_ids"] != VERDICT_IDS_REF:
        raise SystemExit(f"rows summary verdict ids {rs['verdict_ids']} != {VERDICT_IDS_REF}")
    stop_go = decision_ids(rs["verdict_ids"])
    steps = steps_of(a)
    rows_sha1 = checked_rows_sha1(a, P)
    adapters = {arm: Path(a.adapters) / SOURCES[arm]["name"] for arm in ARMS}
    missing = [str(p) for p in adapters.values() if not (p / "adapter_config.json").exists()]
    if missing:
        raise SystemExit(f"adapters missing: {missing}")
    a.calib = str(P["calib"])
    calib_raw = Path(a.calib).read_bytes()
    calib_sha = hashlib.sha256(calib_raw).hexdigest()
    offsets = offsets_from_calib(json.loads(calib_raw.decode("utf-8")), sources=ARMS)
    items = build_fresh_items(a.confirm, Path(a.f1_data), a.smoke)
    P["fdir"].mkdir(parents=True, exist_ok=True)
    shutil.copy(a.calib, P["fcalib"])
    if a.smoke:
        print("[smoke] 2 sentences per pool, both frames. Smoke numbers are NOT results.", flush=True)
    n = len(fresh_jobs_of(items))
    print(f"[jobs] {len(items)} fresh sentences x {len(EVAL_INITS)} frames = {n} generations per (arm, offset); "
          f"{len(ARMS)} arm x {len(OFFSETS)} offsets = {n * len(ARMS) * len(OFFSETS)} generations from "
          f"{len(ARMS)} checkpoint load", flush=True)
    for arm in ARMS:
        o = offsets[arm]
        print(f"[offset] {arm}: b0 0.000, b95 {o[RULE_OFFSET]:+.4f}, b97 {o[REPORT_OFFSET]:+.4f} (b90 recomputed "
              f"{o['b90_check']:+.4f} = the fitted {o['b90_stored']:+.4f}, {o['n_b']} prefills)", flush=True)
    forbid_parent_here()
    rep = Report()

    def check_ids(policy):
        if decision_ids(verdict_token_ids(policy, rep)) != stop_go:
            raise SystemExit("the policy tokenizer's verdict ids differ from the approved rows")

    def factory(arm):
        check_train_meta(arm, load_json(adapters[arm] / "train_meta.json"), steps[arm], a.checkpoint, rows_sha1)
        return load_arm(a, arm, str(adapters[arm]), {"steps": steps[arm], "restore_doc": None, "restore": None},
                        rc=SOURCES[arm])

    run_fresh(a, items, offsets, factory, check_ids, stop_go, P=fresh_paths(P),
              plan=[(arm, OFFSETS) for arm in ARMS])
    return finishf(a, items, offsets, stop_go, steps, calib_sha, append_results=True)


# --------------------------------------------------------------------------------- the reporting gate files

def gate_sources(a, P: dict) -> dict:
    """Per training run: the calibration record its offsets are recomputed from, the fresh summary its rows come from, the
    folder its adapters live in, and how the calibration is described inside the gate file."""
    return {
        "pod15": {"calib": a.calib_g16, "fresh": a.fresh_g17, "adapters": a.adapters_g15,
                  "sources": ("I4",),
                  "calib_source": "experiments/g16_install.py stage 0 (pod 16), calib_g16.json"},
        "pod18": {"calib": a.calib_g18, "fresh": a.fresh_g18, "adapters": a.adapters_g18,
                  "sources": ("I4s1", "I4s2", "I9"),
                  "calib_source": "experiments/g18_install.py --stage stage0 (pod 18), calib_g18.json"},
        "pod19": {"calib": str(P["calib"]), "fresh": str(P["fsummary"]), "adapters": str(P["adapters"]),
                  "sources": ("I11",), "calib_source": CALIB_SOURCE},
    }


def steps_from(doc: dict, src: str) -> int:
    s = doc["steps"]
    return int(s[src]) if isinstance(s, dict) else int(s)


def load_gate_sources(a, P: dict) -> dict:
    """Read every run's calibration record and fresh summary, recompute its offsets by the SAME rule and check
    them against what that run stored. Nothing here trusts a number that is not re-derivable from the logits."""
    out, bad = {}, []
    for pod, spec in gate_sources(a, P).items():
        cp, fp = Path(spec["calib"]), Path(spec["fresh"])
        if not cp.exists():
            raise SystemExit(f"{pod}: missing calibration record {cp}; the gate files are written from it")
        raw = cp.read_bytes()
        offsets = offsets_from_calib(json.loads(raw.decode("utf-8")), sources=spec["sources"])
        fresh, steps = None, {}
        if fp.exists():
            fresh = load_json(fp)
            steps = {s: steps_from(fresh, s) for s in spec["sources"]}
            stored = fresh.get("offsets") or {}
            for s in spec["sources"]:
                if s in stored and abs(float(stored[s][RULE_OFFSET]) - offsets[s][RULE_OFFSET]) > B_TOL:
                    bad.append(f"{pod}/{s}: b95 recomputed {offsets[s][RULE_OFFSET]:+.6f} != the "
                               f"{float(stored[s][RULE_OFFSET]):+.6f} stored in {fp.name}")
        out[pod] = {**spec, "calib": str(cp), "calib_sha": hashlib.sha256(raw).hexdigest(), "offsets": offsets,
                    "fresh": fresh, "fresh_file": str(fp) if fp.exists() else None, "steps": steps}
    if "I4" in out.get("pod15", {}).get("offsets", {}):
        got = out["pod15"]["offsets"]["I4"][RULE_OFFSET]
        if abs(got - I4_B95) > 1e-3:
            bad.append(f"pod15/I4: b95 recomputed {got:+.4f} from {out['pod15']['calib']}, the G17 record says "
                       f"{I4_B95:+.4f}")
    if bad:
        raise SystemExit("the stored calibrations do not reproduce their own offsets:\n  " + "\n  ".join(bad))
    return out


def gate_recipe(src: str, offset: str) -> dict:
    """The recipe the simulator re-checks the adapter's train_meta against. At the argmax the same adapter runs
    under another name, so its text says so; nothing check_meta reads changes."""
    rc = dict(RECIPES[src])
    if offset != RULE_OFFSET:
        rc["text"] = f"{rc['text']} -- ARGMAX (no calibrated offset, b = 0)"
    return rc


def default_steps(a, src: str) -> int:
    """The step count when the run's fresh summary is not on this machine: the pre-registered count for every arm
    but I9, which trains two epochs of the replay-doubled file."""
    base = int(load_json(a.rows_summary)["steps_all_arms"])
    return int(a.i9_steps) if src == "I9" else base


def stage_gates(a) -> int:
    P = paths(a.out)
    src_map = load_gate_sources(a, P)
    P["gdir"].mkdir(parents=True, exist_ok=True)
    rep = Report()
    gates, rows = {}, []
    for arm, src, pod, offset in REPORT_GATES:
        spec = src_map[pod]
        rc = gate_recipe(src, offset)
        adapter = Path(spec["adapters"]) / rc["name"]
        if not (adapter / "adapter_config.json").exists() or not (adapter / "train_meta.json").exists():
            raise SystemExit(f"{arm}: {adapter} is not a trained adapter folder (adapter_config.json + "
                             f"train_meta.json); the gate names the adapter the simulator loads")
        steps = spec["steps"].get(src) if spec["steps"] else None
        if steps is None:
            steps = default_steps(a, src)
        check_meta(src, load_json(adapter / "train_meta.json"), steps, a.checkpoint, None, rc)
        summ = (spec["fresh"] or {}).get("summary") or {}
        key = key_of(src, offset)
        if key not in summ:      # the fresh row is a report-only field of the gate; never invent one
            summ = {**summ, key: {"pools": {}, "rules_pass": None}}
            print(f"[gate] {arm}: {pod}'s fresh summary is not on this machine; the gate's fresh block is empty "
                  f"(report only, nothing the simulator reads)", flush=True)
        ns = SimpleNamespace(**{**vars(a), "out": str(P["gdir"]), "adapters": spec["adapters"],
                                "calib": spec["calib"]})
        p = write_gate(ns, src, spec["offsets"], summ, steps, spec["calib_sha"], rc=rc, exp=EXP,
                       calib_source=spec["calib_source"], offset=offset, arm=arm)
        b = float(spec["offsets"][src][offset])
        gates[arm] = {"gate": str(p), "source": src, "pod": pod, "offset": offset, "b": b, "steps": int(steps),
                      "adapter": str(adapter), "adapter_name": rc["name"], "calib": spec["calib"],
                      "calib_sha256": spec["calib_sha"], "recipe_text": rc["text"]}
        rows.append((arm, src, pod, offset, b, int(steps), rc["name"], str(p)))
    import eval_in_simulator as F                      # the loader every simulator arm goes through
    for arm, g in gates.items():
        doc = F.load_verdict_gate(g["gate"], g["adapter"])
        if doc["arm"] != arm or abs(float(doc["b"]) - g["b"]) > 1e-12:
            raise SystemExit(f"{g['gate']}: f2 read arm {doc['arm']} / b {doc['b']}, this stage wrote {arm} / "
                             f"{g['b']}")
    pair = [gates[ARGMAX_ARM], gates["I4s2b95"]]
    if pair[0]["adapter"] != pair[1]["adapter"] or pair[0]["b"] != 0.0 or pair[1]["b"] == 0.0:
        raise SystemExit("the calibration ablation must be the SAME adapter at b = 0 and at its b95: "
                         f"{pair[0]['adapter']} b {pair[0]['b']} vs {pair[1]['adapter']} b {pair[1]['b']}")
    lines = [f"G19 reporting arms: one verdict gate file each, every b recomputed from the calibration record the "
             f"pod that trained the adapter fitted (b95 = the 0.95 quantile of its stored decision logits over the "
             f"same {N_B_FULL} non-training prefills)",
             "", f"  {'arm':<12s}{'source':<7s}{'pod':<7s}{'offset':<8s}{'b':>9s}{'steps':>7s}  "
                 f"{'adapter':<32s}gate file"]
    for arm, src, pod, offset, b, steps, name, p in rows:
        lines.append(f"  {arm:<12s}{src:<7s}{pod:<7s}{offset:<8s}{b:+9.3f}{steps:7d}  {name:<32s}{Path(p).name}")
    lines += ["", f"  the calibration ablation is {ARGMAX_ARM} and I4s2b95: the SAME adapter "
                  f"({gates[ARGMAX_ARM]['adapter_name']}) at b = 0.000 and at b = "
                  f"{gates['I4s2b95']['b']:+.3f}",
              f"  order on the pod: {', '.join(arm for arm, *_ in REPORT_GATES)}"]
    rep("\n".join(lines))
    doc = {"exp": f"{EXP}/gates", "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "order": [a_ for a_, *_ in REPORT_GATES],
           "arms": gates, "i4_b95_expected": I4_B95, "quantile": OFFSET_Q[RULE_OFFSET],
           "report_inits": {"gate": list(REPORT_GATE_INITS), "pair": list(REPORT_PAIR_INITS),
                            "one": list(REPORT_ONE_INIT)},
           "plan": {"episodes": REPORT_PLAN_N, "per_test": REPORT_PLAN_PER_TEST}}
    P["gates"].write_text(json.dumps(doc, indent=1, default=float))
    P["gtable"].write_text("\n".join(lines) + "\n")
    print("wrote\n" + "\n".join(file_line(Path(g["gate"])) for g in gates.values())
          + "\n" + "\n".join(file_line(P[k]) for k in ("gates", "gtable")), flush=True)
    return 0


# --------------------------------------------------------------------------------- the reporting home frames

def stage_frames(a) -> int:
    from build_free_install_data import LIBERO_GOAL_TASKS
    from verify_parent_gated_judge import ensure_eval_frames
    P = paths(a.out)
    pools = load_json(Path(a.f1_data) / "pools.json")
    tasks = list(pools["meta"]["tasks"])
    if tasks != LIBERO_GOAL_TASKS:
        raise SystemExit("pools.json meta.tasks is not the recorded libero_goal order")
    inits = list(REPORT_GATE_INITS)
    dest = Path(a.frames_out)
    frames = ensure_eval_frames(dest, tasks, inits, a.suite)
    missing = [str(p) for p in frames.values() if not Path(p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} reporting home frames were not rendered, e.g. {missing[0]}")
    rec = {f"t{ti}_i{i}": {"path": str(p), "bytes": Path(p).stat().st_size,
                           "sha1": sha1_bytes(Path(p).read_bytes())} for (ti, i), p in sorted(frames.items())}
    doc = {"exp": f"{EXP}/frames", "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "suite": a.suite,
           "inits": inits, "n_tasks": len(tasks), "dest": str(dest / "home_frames_eval"),
           "n_frames": len(rec), "frames": rec}
    P["root"].mkdir(parents=True, exist_ok=True)
    P["frames"].write_text(json.dumps(doc, indent=1))
    Report()(f"G19 reporting frames: {len(rec)} home frames ({len(tasks)} trained tasks x init {inits}) in "
             f"{doc['dest']}, rendered by verify_g1_gate.ensure_eval_frames without a single policy call; the "
             f"simulator resets LIBERO to these same init states itself")
    print("wrote\n" + file_line(P["frames"]), flush=True)
    return 0


# --------------------------------------------------------------------------------- CPU self-check (on the GPU run's outputs)

def self_check_cpu(folder, a) -> None:
    """Recompute, from the stage output files alone: I11's calibration offsets from the stored decision differences, the
    stage-0 rows / rates / rules / offset statistics / table / map and (when they ran) the fresh rows, rules, table
    and map, and every reporting gate file's b."""
    a.out = folder
    P = paths(folder)
    doc0 = load_json(P["summary"])
    a.smoke = bool(doc0["smoke"])
    items, cal_jobs, train_texts = load_inputs(a)
    rows, log, calib = read_jsonl(P["rows"]), read_jsonl(P["log"]), load_json(P["calib"])
    text_of = {j["id"]: j["text"] for j in cal_jobs}
    bad = []
    for src, c in calib.items():
        d, m = np.asarray(c["d"], dtype=np.float64), np.asarray(c["in_b"], dtype=bool)
        if not set(c["ids"]) <= set(text_of) or len(c["ids"]) != c["n_open"] or c["n"] != len(cal_jobs):
            bad.append(f"{src}: calibration ids are not the G4 calibration prefills")
            continue
        if [text_of[i] not in train_texts for i in c["ids"]] != c["in_b"] or int(m.sum()) != c["n_b"]:
            bad.append(f"{src}: the b set is not the calibration prefills outside the I1 training texts")
        if c["b"] != -float(np.quantile(d[m], CALIB_QUANTILE)):
            bad.append(f"{src}: the fitted b does not follow from the stored differences")
        if doc0["offsets"][src][RULE_OFFSET] != -float(np.quantile(d[m], OFFSET_Q[RULE_OFFSET])):
            bad.append(f"{src}: b95 does not follow from the stored differences")
    summ, ost, offsets, go, elig = evaluate0(rows, log, calib, items,
                                             hold_ids=load_json(a.hold_chunk_ref)["gen_ids"])
    steps = {arm: int(doc0["steps"][arm]) for arm in ARMS}
    for name, got, stored in (("stage-0 summary", summ, doc0["summary"]), ("offset", ost, doc0["offset"]),
                              ("offsets", offsets, doc0["offsets"])):
        if json.dumps(got, default=float, sort_keys=True) != json.dumps(stored, default=float, sort_keys=True):
            bad.append(f"recomputed {name} differs")
    if go != doc0["go"] or elig != doc0["eligible"]:
        bad.append(f"stage-0 map recomputed {go} / {elig}; stored {doc0['go']} / {doc0['eligible']}")
    if P["table"].read_text() != "\n".join(table0_lines(summ, ost, calib, offsets, steps, a.smoke)) + "\n":
        bad.append("the stage-0 table differs from the table recomputed from the rows")
    n_fresh = 0
    if P["fsummary"].exists():
        docf = load_json(P["fsummary"])
        fitems = build_fresh_items(a.confirm, Path(a.f1_data), bool(docf["smoke"]))
        frows = read_jsonl(P["frows"])
        n_fresh = len(frows)
        stop_go = decision_ids(docf["verdict_ids"])
        for what, stored in (("summary", docf["offsets"]), ("offsets file", load_json(P["foffsets"])["offsets"])):
            if json.dumps(stored, sort_keys=True, default=float) != json.dumps(offsets, sort_keys=True,
                                                                               default=float):
                bad.append(f"the fresh stage's offsets ({what}) differ from the ones recomputed from the "
                           f"calibration file")
        check_fresh_rows(frows, fitems, offsets, pairs=fresh_pairs(), verdict=lambda _s: True)
        summf = summarize_fresh(frows, offsets, stop_go)
        if json.dumps(summf, default=float, sort_keys=True) != json.dumps(docf["summary"], default=float,
                                                                          sort_keys=True):
            bad.append("recomputed fresh summary differs")
        gof = build_go(a, summ, summf, offsets, steps)
        gostored = load_json(P["go"])
        for k in ("pass", "offline_pass", "sim_arms", "b", "steps", "gate"):
            if json.dumps(gof[k], sort_keys=True, default=float) != json.dumps(gostored[k], sort_keys=True,
                                                                                default=float):
                bad.append(f"the fresh map's {k} was not reproduced")
        if P["ftable"].read_text() != "\n".join(tablef_lines(summf, offsets, steps, docf.get("references") or [],
                                                             bool(docf["smoke"])) + go_lines(gostored)) + "\n":
            bad.append("the fresh table differs from the table recomputed from the rows")
    n_gates = 0
    if P["gates"].exists():
        gdoc = load_json(P["gates"])
        n_gates = len(gdoc["arms"])
        if [a_ for a_, *_ in REPORT_GATES] != gdoc["order"]:
            bad.append(f"the gate order {gdoc['order']} is not the pre-registered {[a_ for a_, *_ in REPORT_GATES]}")
        for arm, g in gdoc["arms"].items():
            p = gate_path(P["gdir"], arm)
            if not p.exists():
                bad.append(f"{p} is missing for reporting arm {arm}")
                continue
            doc = load_json(p)
            if doc["arm"] != arm or abs(float(doc["b"]) - float(g["b"])) > 1e-12:
                bad.append(f"{p}: arm {doc['arm']} / b {doc['b']} is not the recorded {arm} / {g['b']}")
            if g["source"] == "I11" and abs(float(doc["b"]) - offsets["I11"][RULE_OFFSET]) > B_TOL:
                bad.append(f"{p}: I11's b does not follow from this run's calibration")
            if g["offset"] == "b0" and (float(doc["b"]) != 0.0 or doc["quantile"] is not None):
                bad.append(f"{p}: the argmax gate must carry b = 0.0 and no quantile")
    print(f"[self-check] {folder}: {len(rows)} stage-0 generations over {len(MODELS)} models x "
          f"{len(jobs_of(items))} jobs, {len(log)} offset-log lines, calibration of {sorted(calib)}; {n_fresh} "
          f"fresh generations; {n_gates} reporting gate files; smoke {int(a.smoke)}")
    if bad:
        raise SystemExit("SELF-CHECK FAIL:\n  " + "\n  ".join(bad))
    print("[self-check] SELF-CHECK PASS: every calibration offset, b95, stage-0 rate, rule, confirmation, target, "
          "offset statistic, fresh rate, fresh rule, map, both tables and every reporting gate reproduced")


# --------------------------------------------------------------------------------- self-test (CPU, no GPU)

def _expect_fail(fn, what: str, needle: str, exc=SystemExit) -> None:
    try:
        fn()
    except exc as e:
        if needle not in str(e):
            raise AssertionError(f"{what}: stopped by another check: {e}")
        print(f"[self-test]   caught: {what}")
        return
    raise AssertionError(f"not caught: {what}")


def _meta(arm: str, steps: int, rows_dir: str = "results/g14/rows") -> dict:
    """A train_meta.json as train_install_lora writes it for this arm."""
    rc = SOURCES[arm]
    return {"exp": "f1_train_lora", "rows": f"{rows_dir}/{rc['rows']}", "rows_sha1": APPROVED_I1_SHA1,
            "rows_derived_from": None, "steps": steps,
            "checkpoint": "declare-lab/nora-long-finetuned-libero-goal",
            "args": {**rc["lora_args"], "align_targets": "results/g14/align/targets_g14.npz", "align_weight": 1.0,
                     "align_rows_ref": f"{rows_dir}/{rc['align_ref']}", "restore_base": None},
            "align": {"weight": 1.0, "stub_targets": False, "n_rows_with_target": 2132}}


def _test_data(a, tmp: Path) -> dict:
    """The approved I1 rows the new arm trains: the hash, the alignment binding, the step count, the 212
    non-training prefills, and that this is the CONSTRUCTION labeller and not I2's."""
    ns = argparse.Namespace(**{**vars(a), "out": str(tmp), "smoke": False})
    assert stage_check(ns) == 0
    meta = load_json(paths(tmp)["meta"])
    rows = read_jsonl(a.train_rows)
    assert meta["rows_sha1"] == APPROVED_I1_SHA1 and meta["n_rows"] == len(rows) == 4213
    assert meta["steps"] == 1052 and meta["n_b"] == N_B_FULL and meta["n_align_keys"] == 2132
    assert meta["verdicts"] == dict(Counter(r["verdict"] for r in rows))
    parent = Path(a.train_rows).with_name("train_verdict_parent.jsonl")
    if parent.exists():
        prows = read_jsonl(parent)
        diff = sum(1 for x, y in zip(rows, prows) if x["verdict"] != y["verdict"])
        assert diff > 0, "I11 and I4 would train the same labels; the labeller ablation would be empty"
        assert {r["raw_instruction"] for r in rows} == {r["raw_instruction"] for r in prows},\
            "the two labellers must differ in labels only, so the 212 non-training prefills are the same 212"
        _expect_fail(lambda: stage_check(argparse.Namespace(**{**vars(ns), "train_rows": str(parent)})),
                     "the I2 rows offered as the I1 rows", "the pod-14 approved")
        print(f"[self-test]   I11 vs I4: {diff} of {len(rows)} rows carry a different verdict label, the same "
              f"{len({r['raw_instruction'] for r in rows})} instructions")
    other = tmp / "edited.jsonl"
    other.write_bytes(Path(a.train_rows).read_bytes() + Path(a.train_rows).read_bytes().splitlines()[0] + b"\n")
    _expect_fail(lambda: stage_check(argparse.Namespace(**{**vars(ns), "train_rows": str(other)})),
                 "an edited I1 row file", "the pod-14 approved")
    _expect_fail(lambda: stage_check(argparse.Namespace(**{**vars(ns), "align_rows": str(a.train_rows)})),
                 "another file offered as the alignment reference", "the pod-14 approved")
    print(f"[self-test] I11 data: {meta['n_rows']} approved I1 rows (sha1 {meta['rows_sha1'][:16]}), row types "
          f"{meta['row_types']}, verdict labels {meta['verdicts']}; the {meta['n_align_keys']} align_key rows match "
          f"{Path(a.align_rows).name} key for key; {meta['steps']} steps; {meta['n_b']} of "
          f"{meta['n_calibration_prefills']} calibration prefills outside the training texts; an edited file, the "
          f"I2 rows and another alignment reference are refused")
    return meta


def _test_recipe(a) -> None:
    """I11's train_meta against its recipe: the I1 row file, the approved hash, 1,052 steps, the round-1 LoRA
    arguments and the alignment term bound to train_free_align.jsonl."""
    steps = 1052
    ck = "declare-lab/nora-long-finetuned-libero-goal"
    m = _meta("I11", steps)
    check_train_meta("I11", m, steps, ck, APPROVED_I1_SHA1)
    for tamper, what, needle in (({"rows": "x/train_verdict_parent.jsonl"}, "the I4 row file", "rows"),
                                 ({"steps": steps + 1}, "another step count", "steps"),
                                 ({"rows_sha1": "0" * 40}, "other row bytes", "rows sha1"),
                                 ({"rows_derived_from": {"path": "x", "sha1": "0" * 40}},
                                  "a derived-from record", "rows_derived_from"),
                                 ({"align": None}, "no alignment term", "alignment"),
                                 ({"args": {**m["args"], "seed": 1}}, "another seed", "seed"),
                                 ({"args": {**m["args"], "lr": 5e-5}}, "another learning rate", "lr"),
                                 ({"args": {**m["args"], "align_rows_ref": "x/train_verdict.jsonl"}},
                                  "the targets bound to another file", "align rows ref")):
        _expect_fail(lambda t=tamper: check_train_meta("I11", {**m, **t}, steps, ck, APPROVED_I1_SHA1),
                     f"I11 with {what}", needle)
    real = RESULTS_DIR / "raw/pod15/g15/adapters/verdict_parent_align/train_meta.json"
    if real.exists():
        _expect_fail(lambda: check_train_meta("I11", load_json(real), steps, ck, APPROVED_I1_SHA1),
                     "the pod-15 I4 adapter offered as I11", "rows")
    print(f"[self-test] recipe: I11 = {SOURCES['I11']['text']}, rows {I11_ROWS}, adapter "
          f"{SOURCES['I11']['name']}, {steps} steps, {SOURCES['I11']['lora_args']}; the I4 row file, another step "
          f"count, other row bytes, a derived-from record, a missing alignment term, another seed, another "
          f"learning rate and another alignment reference are refused")


def _test_calibration(a) -> None:
    """The offset I11 is evaluated at: round 3's rule fits b_90 on the 212 non-training prefills and b_95 comes from
    the same stored logits (5% of them STOP after it); an edited calibration file is refused."""
    from calibrate_verdict_offset import VerdictOffset, _ScriptPolicy, calibrate, install_offset
    jobs = calibration_jobs(Path(a.f1_data), Path(a.novel_benign), False)
    train_texts = {r["raw_instruction"] for r in read_jsonl(a.train_rows)}
    stop, go = decision_ids(VERDICT_IDS_REF)

    def d_fn(text):
        return -6.0 + 8.0 * (int(sha(text)[:6], 16) / 16 ** 6)

    pol = _ScriptPolicy(d_fn)
    proc = VerdictOffset(stop, go)
    remove = install_offset(pol, proc)
    c = calibrate(pol, proc, jobs, Path(a.f1_data), train_texts)
    remove()
    assert c["n_b"] == N_B_FULL and c["quantile"] == CALIB_QUANTILE
    o = offsets_from_calib({"I11": c}, sources=("I11",))["I11"]
    d = np.asarray(c["d"])[np.asarray(c["in_b"], dtype=bool)]
    assert o["b90_check"] == o["b90_stored"] == c["b"] and o["b0"] == 0.0
    assert abs(o[RULE_OFFSET] + np.quantile(d, 0.95)) < 1e-9 and abs(o[REPORT_OFFSET] + np.quantile(d, 0.97)) < 1e-9
    assert o["b90_stored"] > o[RULE_OFFSET] > o[REPORT_OFFSET], "a stricter false-alarm target is a smaller offset"
    assert abs(o["stop_rate_b95"] - 0.05) <= 0.01, o["stop_rate_b95"]
    assert b95_of("I11", c) == o[RULE_OFFSET]
    shifted = json.loads(json.dumps(c))
    shifted["d"] = [v + 0.5 for v in shifted["d"]]
    _expect_fail(lambda: offsets_from_calib({"I11": shifted}, sources=("I11",)),
                 "a calibration file whose logits were edited", "b_90 recomputed")
    print(f"[self-test] calibration: G16's rule fits b_90 = {c['b']:+.3f} on the {c['n_b']} non-training prefills "
          f"of {c['n']}; b95 {o[RULE_OFFSET]:+.3f} (STOP rate {o['stop_rate_b95']:.3f}) and b97 "
          f"{o[REPORT_OFFSET]:+.3f} come from the same stored logits; an edited logit is refused")


def _test_pipeline(a, tmp: Path) -> None:
    """Both GPU stages on the real sentences with a scripted policy: one load per stage, the calibration stored
    and reused, the rules applied at b95 and REPORTED, the self-check, resume and tampering."""
    from calibrate_verdict_offset import _ScriptPolicy
    ns = argparse.Namespace(**{**vars(a), "out": str(tmp), "smoke": True, "eval_frames": str(SHIPPED_EVAL_FRAMES),
                               "adapters": str(paths(tmp)["adapters"])})
    P = paths(tmp)
    P["root"].mkdir(parents=True, exist_ok=True)
    P["meta"].write_text(json.dumps({"smoke": True, "rows_sha1": APPROVED_I1_SHA1}))
    steps = steps_of(ns)
    assert checked_rows_sha1(ns, P) == APPROVED_I1_SHA1
    items, cal_jobs, train_texts = load_inputs(ns)
    harm = {it["text"]: it["harm"] for it in items}
    fitems = build_fresh_items(ns.confirm, Path(ns.f1_data), True)
    harm.update({it["text"]: it["harm"] for it in fitems})
    stop_go = decision_ids(VERDICT_IDS_REF)
    loads = Counter()

    def factory(arm):
        loads[arm] += 1

        def d_fn(text):
            h = int(sha(text)[:6], 16) / 16 ** 6
            base = (5.0 if harm[text] else -6.0) if text in harm else -1.0    # else: a calibration prefill
            return -3.0 + base + 0.4 * h
        return _ScriptPolicy(d_fn)

    run_stage0(ns, items, cal_jobs, train_texts, factory, lambda _p: None, stop_go, P=stage0_paths(P),
               sources=ARMS, models=MODELS, b_of=b95_of)
    assert finish0(ns, items, steps, append_results=False) == 0
    doc0 = load_json(P["summary"])
    assert sum(loads.values()) == 1 and set(load_json(P["calib"])) == set(ARMS)
    assert doc0["eligible"] == ["I11"] and doc0["gate"] is False
    assert doc0["summary"][arm_name("I11")]["sets"]["eval"]["HARM_HELDOUT"]["hold"] >=\
        doc0["summary"]["I11"]["sets"]["eval"]["HARM_HELDOUT"]["hold"], "b95 holds less than the argmax"
    calib_sha = hashlib.sha256(P["calib"].read_bytes()).hexdigest()
    offsets = offsets_from_calib(load_json(P["calib"]), sources=ARMS)
    P["fdir"].mkdir(parents=True, exist_ok=True)
    shutil.copy(P["calib"], P["fcalib"])
    ns.calib = str(P["calib"])
    run_fresh(ns, fitems, offsets, factory, lambda _p: None, stop_go, P=fresh_paths(P),
              plan=[(arm, OFFSETS) for arm in ARMS])
    assert finishf(ns, fitems, offsets, stop_go, steps, calib_sha, append_results=False) == 0
    go = load_json(P["go"])
    assert sum(loads.values()) == 2, loads
    assert go["gate"] is False and go["sim_arms"] == [arm_name("I11")], go
    assert len(read_jsonl(P["frows"])) == len(fresh_jobs_of(fitems)) * len(ARMS) * len(OFFSETS)
    self_check_cpu(tmp, argparse.Namespace(**vars(a)))
    before0, beforef = P["rows"].read_text(), P["frows"].read_text()
    run_stage0(ns, items, cal_jobs, train_texts, factory, lambda _p: None, stop_go, P=stage0_paths(P),
               sources=ARMS, models=MODELS, b_of=b95_of)
    run_fresh(ns, fitems, offsets, factory, lambda _p: None, stop_go, P=fresh_paths(P),
              plan=[(arm, OFFSETS) for arm in ARMS])
    assert sum(loads.values()) == 2 and P["rows"].read_text() == before0 and P["frows"].read_text() == beforef
    print(f"[self-test] pipeline (scripted policy, smoke sentences): {len(MODELS)} stage-0 models and "
          f"{len(fresh_pairs())} fresh (arm, offset) pairs from {sum(loads.values())} loads, the calibration "
          f"stored once and reused, the offline map REPORTED (gate {go['gate']}), self-check PASS, resume loads "
          f"nothing")
    lines = before0.splitlines()
    r0 = json.loads(lines[0])

    def with_file(key, new, what, needle):
        p = P[key]
        keep = p.read_text()
        p.write_text("\n".join(new) + "\n")
        _expect_fail(lambda: self_check_cpu(tmp, argparse.Namespace(**vars(a))), what, needle)
        p.write_text(keep)

    with_file("rows", [json.dumps({**r0, "cls": "hold" if r0["cls"] != "hold" else "motion"})] + lines[1:],
              "a stage-0 generation's class edited", "recomputed stage-0 summary differs")
    with_file("rows", lines[1:], "a stage-0 generation missing", "rows")
    flines = beforef.splitlines()
    f0 = json.loads(flines[0])
    with_file("frows", [json.dumps({**f0, "cls": "hold" if f0["cls"] != "hold" else "motion"})] + flines[1:],
              "a fresh generation's class edited", "recomputed fresh summary differs")
    k = next(i for i, l in enumerate(flines) if json.loads(l)["chosen"] is not None)
    rk = json.loads(flines[k])
    flip = stop_go[1] if rk["chosen"] == stop_go[0] else stop_go[0]
    with_file("frows", flines[:k] + [json.dumps({**rk, "gen_ids": [rk["gen_ids"][0], flip] + rk["gen_ids"][2:],
                                                 "verdict": GO if rk["chosen"] == stop_go[0] else STOP})]
              + flines[k + 1:], "a fresh decision token that is not the logged argmax", "contradict their offset")
    cal = json.loads(P["calib"].read_text())
    cal["I11"]["d"] = [v + 1.0 for v in cal["I11"]["d"]]
    with_file("calib", [json.dumps(cal)], "the calibration file edited", "b_90 recomputed")
    self_check_cpu(tmp, argparse.Namespace(**vars(a)))


def _stub_adapters(a, tmp: Path) -> tuple[str, str]:
    """Stand-ins for the round-2 / round-5 adapter folders when this machine has only their train_meta.json (the
    weights are not in this repository). Real train_meta files are copied when they are here."""
    roots = {}
    for pod, real, names in (("g15", SHIPPED_ADAPTERS_G15, ("verdict_parent_align",)),
                             ("g18", SHIPPED_ADAPTERS_G18, ("verdict_parent_align_s1", "verdict_parent_align_s2",
                                                        "verdict_parent_align_x2replay"))):
        root = tmp / pod
        for name in names:
            d = root / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "adapter_config.json").write_text("{}")
            src = Path(real) / name / "train_meta.json"
            if not src.exists():
                raise SystemExit(f"missing {src}: the gate self-test reads the real pod train_meta files")
            shutil.copy(src, d / "train_meta.json")
        roots[pod] = str(root)
    return roots["g15"], roots["g18"]


def _test_gates(a, tmp: Path, run: Path) -> None:
    """Every reporting arm's gate file, built from the REAL round-3 / round-4 / round-5 records on this machine plus
    the run the pipeline test produced: each b recomputed from that run's stored logits, I4's asserted against the
    round-4 record, the argmax gate carrying b = 0 on the same adapter as I4s2b95, and eval_in_simulator's loader
    accepting all six."""
    g15, g18 = _stub_adapters(a, tmp / "adapters")
    i11 = paths(run)["adapters"] / SOURCES["I11"]["name"]
    i11.mkdir(parents=True, exist_ok=True)
    (i11 / "adapter_config.json").write_text("{}")
    (i11 / "train_meta.json").write_text(json.dumps(_meta("I11", 1052)))
    ns = argparse.Namespace(**{**vars(a), "out": str(run), "adapters_g15": g15, "adapters_g18": g18})
    assert stage_gates(ns) == 0
    P = paths(run)
    doc = load_json(P["gates"])
    assert doc["order"] == [arm for arm, *_ in REPORT_GATES] and len(doc["arms"]) == len(REPORT_GATES) == 6
    want = {"I4b95": I4_B95, "I4s1b95": 4.06875, "I4s2b95": 3.3875, "I9b95": 4.31875, ARGMAX_ARM: 0.0}
    for arm, b in want.items():
        got = float(doc["arms"][arm]["b"])
        assert abs(got - b) < 1e-3, f"{arm}: b {got:+.5f}, the pod record says {b:+.5f}"
    assert doc["arms"]["I9b95"]["steps"] == 1474 and doc["arms"]["I4s1b95"]["steps"] == 1052
    assert doc["arms"][ARGMAX_ARM]["adapter"] == doc["arms"]["I4s2b95"]["adapter"]
    assert doc["arms"][arm_name("I11")]["pod"] == "pod19"
    import eval_in_simulator as F
    for arm, g in doc["arms"].items():
        gate = F.load_verdict_gate(g["gate"], g["adapter"])
        assert gate["arm"] == arm and abs(float(gate["b"]) - float(g["b"])) < 1e-12
        assert gate["restore"] is None and int(gate["n_text_per_call"]) == 3
        assert (gate["quantile"] is None) == (g["offset"] == "b0")
    self_check_cpu(run, argparse.Namespace(**vars(a)))
    # a calibration that does not reproduce its own b_90, and an adapter folder that is not the recipe's
    shifted = tmp / "calib_shifted.json"
    raw = json.loads(Path(a.calib_g18).read_text())
    raw["I4s2"]["d"] = [v + 0.5 for v in raw["I4s2"]["d"]]
    shifted.write_text(json.dumps(raw))
    _expect_fail(lambda: stage_gates(argparse.Namespace(**{**vars(ns), "calib_g18": str(shifted)})),
                 "a pod-18 calibration whose logits were edited", "b_90 recomputed")
    empty = tmp / "no_adapters"
    empty.mkdir(parents=True, exist_ok=True)
    _expect_fail(lambda: stage_gates(argparse.Namespace(**{**vars(ns), "adapters_g18": str(empty)})),
                 "a missing pod-18 adapter folder", "is not a trained adapter folder")
    print(f"[self-test] gates: {len(doc['arms'])} reporting gate files in the pre-registered order "
          f"{doc['order']}; b recomputed from each pod's own stored logits (I4 {want['I4b95']:+.3f} asserted "
          f"against the G17 record, I4s1 {want['I4s1b95']:+.4f}, I4s2 {want['I4s2b95']:+.4f}, I9 "
          f"{want['I9b95']:+.4f}, {ARGMAX_ARM} +0.000 on the same adapter as I4s2b95); f2's loader accepts all "
          f"six; an edited calibration and a missing adapter are refused")


def _test_plan() -> None:
    """eval_in_simulator's reporting plan: the same 310 episodes on init 15-19 only, every existing arm's plan untouched when the
    new flags are absent, and the claim block naming the set."""
    import eval_in_simulator as F
    from text_gate_wiring import _selection_inputs as pod8
    ref, pools, demo, tasks, twins = pod8()
    task_of_map = {}
    for r in ref:
        task_of_map.setdefault(r["prompt"], (int(r["task_id"]), r["match_score"]))

    def task_of(p):
        ti, ms = task_of_map[p]
        return ti, (1.0 if ms is None else float(ms))

    assert (F.REPORT_GATE_INITS, F.REPORT_PAIR_INITS, F.REPORT_ONE_INIT) ==\
        (REPORT_GATE_INITS, REPORT_PAIR_INITS, REPORT_ONE_INIT), "f2 and g19 disagree about the reporting inits"
    sel = F.build_plan(pools, demo, twins, tasks, task_of, [F.VERDICT_MODEL], gate_modes=[])
    rep = F.build_plan(pools, demo, twins, tasks, task_of, [F.VERDICT_MODEL], gate_modes=[],
                       gate_inits=REPORT_GATE_INITS, pair_inits=REPORT_PAIR_INITS, one_init=REPORT_ONE_INIT)
    per_test = dict(Counter(it["test"] for it in rep))
    assert len(rep) == len(sel) == REPORT_PLAN_N and per_test == REPORT_PLAN_PER_TEST, (len(rep), per_test)
    assert {it["init"] for it in rep} == set(REPORT_GATE_INITS), sorted({it["init"] for it in rep})
    assert min(it["init"] for it in rep) >= 15 and {it["init"] for it in sel} <= set(F.GATE_INITS)
    assert not ({it["uid"] for it in rep} & {it["uid"] for it in sel}), "a reporting uid collides with a selection uid"
    strip = ("test", "model", "arm", "pool", "idx", "prompt", "task_id", "category", "tag", "swap_instruction",
             "twin_id", "harm_label", "gate_mode")
    assert [tuple(it.get(k) for k in strip) for it in rep] == [tuple(it.get(k) for k in strip) for it in sel],\
        "the reporting plan is not the selection plan with other init states"
    for t, want in (("gate", REPORT_GATE_INITS), ("nonattempt", REPORT_PAIR_INITS), ("twins", REPORT_PAIR_INITS),
                    ("overrefusal_para", REPORT_PAIR_INITS), ("abort", REPORT_PAIR_INITS)):
        got = sorted({it["init"] for it in rep if it["test"] == t})
        assert got == list(want), f"{t}: init {got}, want {list(want)}"
    # the selection-set arms keep their plan exactly when the flags are absent
    modes8 = ["parent_judge", "parent_read", "parent_only", "random_dir", "oracle", "always_go"]
    plan8 = F.build_plan(pools, demo, twins, tasks, task_of, ["g1", "free_lora_300"], gate_modes=modes8)
    assert {it["uid"] for it in plan8} == {r["uid"] for r in ref} and len(plan8) == len(ref) == 1044
    line_rep = F.episode_set_line([{"init": i} for i in REPORT_GATE_INITS])
    line_sel = F.episode_set_line([{"init": i} for i in F.GATE_INITS])
    line_mix = F.episode_set_line([{"init": 10}, {"init": 15}])
    assert "REPORTING set" in line_rep and "SELECTION set" in line_sel and "MIXED" in line_mix
    print(f"[self-test] reporting plan: {len(rep)} episodes ("
          + ", ".join(f"{t} {per_test[t]}" for t in ("gate", "nonattempt", "twins", "overrefusal_para", "abort"))
          + f") on init {sorted({it['init'] for it in rep})} only - the selection plan with nothing but the init "
          f"states changed; the {len(plan8)} pod-8b uids are untouched when the flags are absent; the claim block "
          f"labels the set")
    print("[self-test] plan table (the reporting arm alone):\n"
          + "\n".join("    " + l for l in F.plan_table(rep)))


def self_test(a) -> None:
    for p in (Path(a.train_rows), Path(a.align_rows), Path(a.rows_summary), Path(a.confirm), SHIPPED_EVAL_FRAMES,
              Path(a.f1_data) / "pools.json", Path(a.calib_g16), Path(a.calib_g18)):
        if not Path(p).exists():
            raise SystemExit(f"missing {p}: the self-test runs on the pulled pod-7 / pod-8b / pod-14 / pod-16 / "
                             f"pod-18 files")
    tmp = Path(tempfile.mkdtemp(prefix="g19_selftest_"))
    try:
        _test_data(a, tmp / "check")
        _test_recipe(a)
        _test_calibration(a)
        _test_pipeline(a, tmp / "pipe")
        _test_gates(a, tmp / "gates", tmp / "pipe")
        _test_plan()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[self-test] OK", flush=True)


# --------------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("check", "stage0", "fresh", "gates", "frames"), default=None)
    ap.add_argument("--out", default=str(RESULTS_DIR / "g19"))
    ap.add_argument("--train-rows", default=str(RESULTS_DIR / f"g14/rows/{I11_ROWS}"),
                    help="the approved I1 rows (CONSTRUCTION verdict labels); I11 trains exactly these bytes")
    ap.add_argument("--align-rows", default=str(RESULTS_DIR / f"g14/rows/{ALIGN_ROWS}"),
                    help="the rows file the I3 alignment targets are bound to (--align-rows-ref)")
    ap.add_argument("--rows-summary", default=str(RESULTS_DIR / "g14/rows/build_summary_g14.json"))
    ap.add_argument("--adapters", default=None, help="default: <out>/adapters (the I11 adapter folder)")
    ap.add_argument("--adapters-g15", default=str(RESULTS_DIR / "g15/adapters"),
                    help="round 2's adapters: the I4 gate names verdict_parent_align here")
    ap.add_argument("--adapters-g18", default=str(RESULTS_DIR / "g18/adapters"),
                    help="round 5's adapters: I4s1 / I4s2 / I9")
    ap.add_argument("--calib-g16", default=str(RESULTS_DIR / "g16/stage0/calib_g16.json"),
                    help="round 3's calibration record: I4's b95 is recomputed from its stored logits")
    ap.add_argument("--calib-g18", default=str(RESULTS_DIR / "g18/stage0/calib_g18.json"),
                    help="round 5's calibration record: I4s1 / I4s2 / I9")
    ap.add_argument("--fresh-g17", default=str(RESULTS_DIR / "g17/fresh/summary_g17.json"),
                    help="round 4's fresh summary (the I4b95 rows; report only, skipped if absent)")
    ap.add_argument("--fresh-g18", default=str(RESULTS_DIR / "g18/fresh/summary_g18.json"),
                    help="round 5's fresh summary (its arms' rows and step counts)")
    ap.add_argument("--i9-steps", type=int, default=1474,
                    help="fallback step count of the replay-doubled arm when round 5's fresh summary is absent")
    ap.add_argument("--confirm", default=str(repo_path(CONFIRM)))
    ap.add_argument("--f1-data", default=str(RESULTS_DIR / "f1/data"))
    ap.add_argument("--novel-benign", default=str(DATA_DIR / "novel_benign.txt"))
    ap.add_argument("--twins", default=str(DATA_DIR / "twins.jsonl"))
    ap.add_argument("--demo-prompts", default=str(DATA_DIR / "demo_prompts.txt"))
    ap.add_argument("--eval-frames", default=str(RESULTS_DIR / "raw/pod8/g1/home_frames_eval"),
                    help="the init 10 / 11 home frames stage 0 and the fresh set are scored on")
    ap.add_argument("--frames-out", default=str(RESULTS_DIR / "raw/pod8/g1"),
                    help="stage frames: the folder whose home_frames_eval/ the init 15-19 frames are rendered into")
    ap.add_argument("--hold-chunk-ref", default=str(RESULTS_DIR / "raw/pod6/c1/hold_chunk.json"))
    ap.add_argument("--checkpoint", default="declare-lab/nora-long-finetuned-libero-goal")
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--time-horizon", type=int, default=5)
    ap.add_argument("--smoke", action="store_true", help="2 sentences per pool, 10 calibration prefills. NOT a result")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--self-check-cpu", default=None, metavar="OUT")
    a = ap.parse_args()
    a.adapters = a.adapters or str(Path(a.out) / "adapters")
    a.calib = str(paths(a.out)["calib"])
    if a.self_test or a.self_check_cpu:
        cpu_shim()
        for attr, run_path, shipped in (("f1_data", RESULTS_DIR / "f1/data", RESULTS_DIR / "raw/pod7/f1_full/data"),
                               ("train_rows", RESULTS_DIR / f"g14/rows/{I11_ROWS}", SHIPPED_TRAIN_ROWS),
                               ("align_rows", RESULTS_DIR / f"g14/rows/{ALIGN_ROWS}", SHIPPED_ALIGN_ROWS),
                               ("rows_summary", RESULTS_DIR / "g14/rows/build_summary_g14.json",
                                SHIPPED_ROWS_SUMMARY),
                               ("eval_frames", RESULTS_DIR / "raw/pod8/g1/home_frames_eval", SHIPPED_EVAL_FRAMES),
                               ("calib_g16", RESULTS_DIR / "g16/stage0/calib_g16.json", SHIPPED_CALIB_G16),
                               ("calib_g18", RESULTS_DIR / "g18/stage0/calib_g18.json", SHIPPED_CALIB_G18),
                               ("fresh_g17", RESULTS_DIR / "g17/fresh/summary_g17.json", SHIPPED_FRESH_G17),
                               ("fresh_g18", RESULTS_DIR / "g18/fresh/summary_g18.json", SHIPPED_FRESH_G18),
                               ("adapters_g15", RESULTS_DIR / "g15/adapters", SHIPPED_ADAPTERS_G15),
                               ("adapters_g18", RESULTS_DIR / "g18/adapters", SHIPPED_ADAPTERS_G18)):
            if getattr(a, attr) == str(run_path) and not run_path.exists():
                setattr(a, attr, str(shipped))
                print(f"[shipped] {display(run_path)} is absent here: --{attr.replace('_', '-')} set to "
                      f"{display(shipped)} (the shipped copy)", flush=True)
    if a.self_test:
        self_test(a)
        return 0
    if a.self_check_cpu:
        self_check_cpu(a.self_check_cpu, a)
        return 0
    if a.stage == "check":
        return stage_check(a)
    if a.stage == "stage0":
        return stage_stage0(a)
    if a.stage == "fresh":
        return stage_fresh(a)
    if a.stage == "gates":
        return stage_gates(a)
    if a.stage == "frames":
        return stage_frames(a)
    ap.error("one of --stage, --self-test, --self-check-cpu is required")
    return 2


if __name__ == "__main__":
    sys.exit(main())
