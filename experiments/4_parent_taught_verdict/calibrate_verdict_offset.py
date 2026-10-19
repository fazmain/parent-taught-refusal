#!/usr/bin/env python3
"""Round 3 of the install search (pre-registered; internally "G16", the name kept in file names such as
calib_g16.json and in the `--g14-stage0` / `--g15-stage0` flags): a calibrated decision offset for the installed
verdict, and two new restore-based verdict installs, on the SAME approved training rows of round 1 ("G14 v4").
Round 2 is "G15"; the calibration prefills are those of the probe-gate decider ("G4", Appendix B).

The verdict arms generate "[STOP]" / "[GO]" (ids [58, 50669, 60] / [58, 15513, 60]) before the action chunk. The
decision is the SECOND generated token, after "[": 50669 vs 15513. A calibrated model adds a fixed offset b to the
[STOP] logit at that position only (`VerdictOffset`, a LogitsProcessor run after the default ones), in the policy's
own greedy generation; nothing else changes. The 360 probe-gate calibration prefills (180 benign calibration
sentences x 2 home frames of init 0-9, as probe_gate_first_call builds them) are read through the same generation
path at b = 0; b = -(the 0.90 quantile of logit[STOP] - logit[GO]) over ONLY the 212 of them whose sentence is not a
v4 training text (a recorded amendment to the pre-registration: 74 calibration sentences are training rows labelled
[GO]). A second pass with b checks that the generated decision is STOP exactly when logit[STOP] + b > logit[GO] and
reports the STOP rate after the offset on the 212 (0.10 by construction) and on all 360.

Models (stage-0 table order; priority for the simulator: I7, I8, then I2b, I4b, I6b, I7b, I8b):
  I2b  round-1 I2 verdict_parent + offset                   I7   rank-32 restore + I2 rows + alignment 1.0 (argmax)
  I4b  round-2 I4 verdict_parent_align + offset             I7b  I7 + offset
  I6b  round-2 I6 restore-32 + verdict_parent + offset      I8   rank-128 restore + I2 rows (argmax)
                                                            I8b  I8 + offset
Rank 32 is round 2's restore (its factors and fingerprints); rank 128 is computed here by the same stage
(train_verdict_alignment.run_restore) against plain NORA. The six first-call rows of round 1 and the three of round 2
are reused alongside.

  restore  GPU. Rank-128 factors of W_parent - W_NORA, plain NORA.   -> <out>/restore/{factors_r128_g16.safetensors, ...}
  stage0   GPU. Per adapter: load (plain / restore-32 / restore-128, the adapter's train_meta checked against its
           recipe), argmax generations (I7, I8), calibration (360 two-token generations at b = 0, then 360 with b),
           generations with the offset. -> <out>/stage0/{stage0_g16.jsonl, offset_log_g16.jsonl, calib_g16.json,
                                                          stage0_g16.json, stage0_table_g16.txt, g16_go.json}

  python3 experiments/4_parent_taught_verdict/calibrate_verdict_offset.py --self-test                                      # CPU, no GPU
  python3 experiments/4_parent_taught_verdict/calibrate_verdict_offset.py --stage restore --out results/g16                # GPU
  python3 experiments/4_parent_taught_verdict/calibrate_verdict_offset.py --stage stage0 --out results/g16 [--smoke]       # GPU: 0 = eligible arm, 3 = NO-GO
  python3 experiments/4_parent_taught_verdict/calibrate_verdict_offset.py --self-check-cpu results/g16                     # CPU, on the GPU run's outputs

RULES, CONFIRMATION, TARGET and the simulator condition are round 1's. No log line printed here contains command text.
"""
from __future__ import annotations

import argparse
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
from refusal.paths import DATA_DIR, RESULTS_DIR, display, add_experiment_dirs  # noqa: E402
add_experiment_dirs()
from first_call_token_check import classify  # noqa: E402
from build_training_rows import GO, STOP, VERDICT_IDS_REF, cpu_shim, verdict_token_ids  # noqa: E402
from first_call_eval import MODELS as G14_MODELS  # noqa: E402
from first_call_eval import (build_items, check_rows, free_cuda, jobs_of, load_json, model_block,  # noqa: E402
                                    parse_verdict, summarize)
from train_verdict_alignment import ARMS as G15_ARMS  # noqa: E402
from train_verdict_alignment import (COMPARE, G14_TEXT, LORA_ARGS, RECIPE as G15_RECIPE, build_factors,  # noqa: E402
                                     check_meta, fingerprint, load_arm, reference_rows, restore_base, run_restore,
                                     write_restore)
from probe_gate_first_call import Report, file_line, sha  # noqa: E402
from single_model_judges import ParentTensors, forbid_parent, restore_targets  # noqa: E402
from verify_parent_gated_judge import assign_scenes, calib_sentences, load_png  # noqa: E402

EXP = "g16_install"
RANK = 128
QUANTILE = 0.90
N_CALIB, N_HOME = 180, 2                  # the probe-gate calibration prefills: 180 sentences x 2 home frames of init 0-9
SMOKE_CALIB = 10
N_B_FULL = 212                            # recorded amendment: calibration prefills whose sentence is not a v4 training text
OPEN, CLOSE = 58, 60                      # "[" and "]"
D_TOL = 1e-3                              # the b = 0 pass and the offset pass read the same decision logits
BOUNDARY = 1e-4                           # |d + b| below this: fp32 rounding may decide either way (reported)
SOURCES = {
    "I2": {**G15_RECIPE["I4"], "name": "verdict_parent", "align_weight": None, "align_ref": None, "restore": None,
           "text": "G14 I2 verdict_parent (parent labels)"},
    "I4": {**G15_RECIPE["I4"], "name": "verdict_parent_align", "text": "G15 I4: I2 rows + alignment 1.0"},
    "I6": {**G15_RECIPE["I6"], "name": "restore_verdict_parent", "text": "G15 I6: restore-32 + I2 rows"},
    "I7": {"name": "restore32_verdict_parent_align", "rows": "train_verdict_parent.jsonl", "align_weight": 1.0,
           "align_ref": "train_free_align.jsonl", "restore": 32, "text": "restore-32 + I2 rows + alignment 1.0"},
    "I8": {"name": "restore128_verdict_parent", "rows": "train_verdict_parent.jsonl", "align_weight": None,
           "align_ref": None, "restore": 128, "text": "restore-128 + I2 rows"},
}
MODELS = {"I2b": ("I2", True), "I4b": ("I4", True), "I6b": ("I6", True), "I7": ("I7", False), "I7b": ("I7", True),
          "I8": ("I8", False), "I8b": ("I8", True)}           # name -> (adapter source, calibrated offset)
PRIORITY = ("I7", "I8", "I2b", "I4b", "I6b", "I7b", "I8b")
NO_GO_EXIT = 3
SHIPPED_G14_STAGE0 = RESULTS_DIR / "raw/pod14/g14_v4/stage0"
SHIPPED_G15_STAGE0 = RESULTS_DIR / "raw/pod15/g15/stage0"
SHIPPED_TRAIN_ROWS = RESULTS_DIR / "raw/pod14/g14_v4/rows/train_verdict_parent.jsonl"
SHIPPED_EVAL_FRAMES = RESULTS_DIR / "raw/pod8/g1/home_frames_eval"


def paths(out) -> dict:
    o = Path(out)
    return {"restore_dir": o / "restore", "factors": o / f"restore/factors_r{RANK}_g16.safetensors",
            "restore": o / f"restore/restore_base_r{RANK}_g16.json",
            "restore_table": o / f"restore/restore_table_r{RANK}_g16.txt", "adapters": o / "adapters",
            "dir": o / "stage0", "rows": o / "stage0/stage0_g16.jsonl", "log": o / "stage0/offset_log_g16.jsonl",
            "calib": o / "stage0/calib_g16.json", "summary": o / "stage0/stage0_g16.json",
            "table": o / "stage0/stage0_table_g16.txt", "go": o / "stage0/g16_go.json"}


# --------------------------------------------------------------------------------- the offset

def decision_ids(verdict_ids: dict) -> tuple[int, int]:
    """(STOP id, GO id) of the decision token: both verdicts are "[", one token, "]" and differ only there."""
    s, g = list(verdict_ids[STOP]), list(verdict_ids[GO])
    if len(s) != 3 or len(g) != 3 or s[0] != OPEN or g[0] != OPEN or s[2] != CLOSE or g[2] != CLOSE or s[1] == g[1]:
        raise SystemExit(f"verdict ids {verdict_ids} are not [58, x, 60] / [58, y, 60] with x != y")
    return s[1], g[1]


class VerdictOffset:
    """LogitsProcessor: at the verdict decision position (the second generated token, when the first generated token
    is "["), add `b` to the [STOP] logit; every other position and every other logit passes through untouched (the
    same tensor object is returned). One record per generate call: the raw decision difference d = logit[STOP] -
    logit[GO], whether the two top raw logits are [STOP] and [GO], the raw argmax and the argmax after the offset."""

    def __init__(self, stop_id: int, go_id: int, b: float = 0.0):
        self.stop_id, self.go_id, self.b = int(stop_id), int(go_id), float(b)
        self.log: list[dict] = []
        self.cur: dict | None = None

    def begin(self) -> None:
        self.cur = {"prompt_len": None, "calls": 0, "applied_at": [], "first": None, "d": None, "top2": None,
                    "raw": None, "chosen": None, "b": self.b}

    def __call__(self, input_ids, scores):
        c = self.cur
        if c is None or input_ids.shape[0] != 1:
            raise RuntimeError("VerdictOffset: generate through install_offset, one sequence per call")
        if c["prompt_len"] is None:
            c["prompt_len"] = int(input_ids.shape[1])
        step = int(input_ids.shape[1]) - c["prompt_len"]
        if step != c["calls"]:
            raise RuntimeError(f"VerdictOffset called at step {step} after {c['calls']} calls")
        c["calls"] += 1
        if step == 1:
            c["first"] = int(input_ids[0, -1])
        if step != 1 or c["first"] != OPEN:
            return scores
        s = scores[0]
        c["d"] = float(s[self.stop_id] - s[self.go_id])
        c["top2"] = sorted(int(i) for i in s.topk(2).indices.tolist()) == sorted((self.stop_id, self.go_id))
        c["raw"] = int(s.argmax())
        out = scores.clone()
        out[:, self.stop_id] += self.b
        c["chosen"] = int(out[0].argmax())
        c["applied_at"].append(step)
        return out

    def end(self, n_new: int) -> None:
        c = self.cur
        if c["calls"] != n_new or any(s != 1 for s in c["applied_at"]) or len(c["applied_at"]) > 1:
            raise RuntimeError(f"VerdictOffset: {c['calls']} calls for {n_new} generated tokens, applied at "
                               f"{c['applied_at']} (only the decision position 1 is allowed)")
        self.log.append(c)
        self.cur = None


def install_offset(policy, proc: VerdictOffset):
    """Route policy.model.generate through `proc` and check every act(): the generated decision token is the
    offset argmax. Returns the function that removes both wrappers."""
    from transformers import LogitsProcessorList
    model, orig_generate, orig_act = policy.model, policy.model.generate, policy.act

    def generate(*args, **kw):
        if kw.get("logits_processor"):
            raise RuntimeError("generate already carries logits processors")
        proc.begin()
        out = orig_generate(*args, logits_processor=LogitsProcessorList([proc]), **kw)
        proc.end(int(out.shape[1]) - proc.cur["prompt_len"])
        return out

    def act(image, text, **kw):
        r = orig_act(image, text, **kw)
        e = proc.log[-1]
        if e["applied_at"] and (len(r["gen_ids"]) < 2 or int(r["gen_ids"][1]) != e["chosen"]):
            raise RuntimeError("the generated decision token is not the argmax after the offset")
        return r

    model.generate, policy.act = generate, act

    def remove():
        del model.generate
        del policy.act
    return remove


# --------------------------------------------------------------------------------- calibration

def calibration_jobs(f1_data: Path, novel: Path, smoke: bool) -> list[dict]:
    """The probe-gate calibration prefills: calib_sentences (TRAINED + BENIGN_PARA_TRAIN + NOVEL_BENIGN), scene = nearest
    trained task, sentence k on the home frames of init (k + j) % 10, j = 0, 1 (probe_gate_first_call)."""
    pools = load_json(Path(f1_data) / "pools.json")
    cal = calib_sentences(pools, Path(novel))
    assign_scenes(cal, pools["meta"]["tasks"])
    if len(cal) != N_CALIB:
        raise SystemExit(f"{len(cal)} calibration sentences, G4 has {N_CALIB}")
    jobs = [{"id": f"cal{k:03d}_i{(k + j) % 10}", "pool": it["pool"], "text": it["text"],
             "frame": f"home_frames/t{it['task_id']}_i{(k + j) % 10}.png"} for k, it in enumerate(cal)
            for j in range(N_HOME)]
    missing = [j["frame"] for j in jobs if not (Path(f1_data) / j["frame"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} calibration frames missing, e.g. {missing[0]}")
    return jobs[:: len(jobs) // SMOKE_CALIB][:SMOKE_CALIB] if smoke else jobs


def decision_pass(policy, proc: VerdictOffset, jobs: list[dict], f1_data: Path) -> list[dict]:
    """Two generated tokens per prefill through the installed offset; the processor's record per prefill."""
    import torch
    out = []
    for j in jobs:
        inputs = policy._inputs(load_png(Path(f1_data) / j["frame"]), j["text"])
        with torch.inference_mode():
            policy.model.generate(**inputs, max_new_tokens=2, do_sample=False)
        e = proc.log[-1]
        out.append({"id": j["id"], "first": e["first"], "d": e["d"], "top2": e["top2"], "chosen": e["chosen"]})
    return out


def calibrate(policy, proc: VerdictOffset, jobs: list[dict], f1_data: Path, train_texts: set) -> dict:
    """Pass 1 (b = 0) over all calibration prefills; b = -(the 0.90 quantile of d) over ONLY the prefills whose
    sentence is not a v4 training text (the recorded amendment: 212 of the 360); pass 2 with b over all of them: same decision
    logits, generated decision = STOP exactly when d + b > 0 (except within BOUNDARY). STOP rates after the offset on
    the b set (0.10 by construction) and on all prefills."""
    t0 = time.time()
    proc.b = 0.0
    p1 = decision_pass(policy, proc, jobs, f1_data)
    ok = [k for k, r in enumerate(p1) if r["d"] is not None]
    in_b = [k for k in ok if jobs[k]["text"] not in train_texts]
    if not in_b:
        raise SystemExit("no calibration prefill outside the v4 training texts generated '[' first")
    d = np.asarray([p1[k]["d"] for k in ok], dtype=np.float64)
    mask = np.asarray([jobs[k]["text"] not in train_texts for k in ok])
    q = float(np.quantile(d[mask], QUANTILE))
    b = -q
    proc.b = b
    p2 = decision_pass(policy, proc, jobs, f1_data)
    bad, boundary = [], 0
    for r1, r2 in zip(p1, p2):
        if (r1["d"] is None) != (r2["d"] is None) or (r1["d"] is not None and abs(r1["d"] - r2["d"]) > D_TOL):
            bad.append(f"{r1['id']}: decision logits differ between the two passes")
        elif r2["d"] is not None and r2["top2"]:
            if abs(r2["d"] + b) < BOUNDARY:
                boundary += 1
            elif (r2["chosen"] == proc.stop_id) != (r2["d"] + b > 0):
                bad.append(f"{r1['id']}: generated {r2['chosen']} with d + b = {r2['d'] + b:+.4f}")
    if bad:
        raise SystemExit(f"the offset pass contradicts the calibration: {bad[:3]} ({len(bad)} prefills)")
    pred = d + b > 0
    gen = np.asarray([p2[k]["chosen"] == proc.stop_id for k in ok])
    in_train = sorted({j["text"] for j in jobs} & train_texts)
    res = {"b": b, "q": q, "quantile": QUANTILE, "b_set": "calibration prefills whose sentence is not a v4 training text",
           "n": len(jobs), "n_open": len(ok), "n_b": len(in_b), "n_top2": int(sum(bool(p1[k]["top2"]) for k in ok)),
           "stop_rate_b": {"predicted": float(pred[mask].mean()), "generated": float(gen[mask].mean())},
           "stop_rate_all": {"predicted": float(pred.mean()), "generated": float(gen.mean())},
           "n_boundary": boundary, "train_overlap": {"sentences": len(in_train), "prefills": len(ok) - len(in_b)},
           "ids": [jobs[k]["id"] for k in ok], "d": [float(x) for x in d], "in_b": [bool(x) for x in mask],
           "wall_s": round(time.time() - t0, 1)}
    print(f"[calib] b = {b:+.4f} (q{QUANTILE:.2f} of logit[STOP] - logit[GO] = {q:+.4f}) over the {len(in_b)} calibration "
          f"prefills whose sentence is not a v4 training text; STOP rate after the offset on those "
          f"{res['stop_rate_b']['predicted']:.3f} (generated {res['stop_rate_b']['generated']:.3f}), on all "
          f"{len(ok)}/{len(jobs)} {res['stop_rate_all']['predicted']:.3f} (generated "
          f"{res['stop_rate_all']['generated']:.3f}); {res['wall_s']:.0f}s", flush=True)
    return res


# --------------------------------------------------------------------------------- stage 0

def run_jobs(name: str, policy, jobs: list, eval_frames: Path, fh, flog, proc: VerdictOffset | None) -> int:
    """One generation per (sentence, frame) job, the round-1 first-call row; with an offset, one offset-log line each."""
    t0, cache = time.time(), {}
    for it, init in jobs:
        frame = f"t{it['task_id']}_i{init}.png"
        if frame not in cache:
            cache[frame] = load_png(Path(eval_frames) / frame)
        r = policy.act(cache[frame], it["text"])
        gen = [int(t) for t in r["gen_ids"]]
        row = {"model": name, "set": it["set"], "idx": it["idx"], "pool": it["pool"], "harm": it["harm"],
               "text_sha1": it["text_sha1"], "task_id": it["task_id"], "init": init, "frame": frame,
               "cls": classify(r), "verdict": parse_verdict(gen, VERDICT_IDS_REF), "n_text": int(r["n_text"]),
               "invalid": bool(r["invalid"]), "gen_ids": gen}
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        if proc is not None:
            e = proc.log[-1]
            flog.write(json.dumps({"model": name, "set": it["set"], "idx": it["idx"], "init": init, "b": e["b"],
                                   "first": e["first"], "d": e["d"], "top2": e["top2"], "raw": e["raw"],
                                   "chosen": e["chosen"]}) + "\n")
            flog.flush()
    print(f"[{name}] {len(jobs)} generations in {time.time() - t0:.0f}s", flush=True)
    return len(jobs)


def read_jsonl(p) -> list[dict]:
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def run_stage0(a, items: list[dict], cal_jobs: list[dict], train_texts: set, factory, check_ids, stop_go,
               P: dict | None = None, sources=None, models: dict | None = None, b_of=None) -> None:
    """Per adapter source: load once, argmax generations, calibration (stored, reused on resume), offset
    generations. `factory(src)` returns the loaded policy.

    Round 5 (train_verdict_seeds) drives the same loop over other arms and another offset: `P` (a paths dict with dir / rows / calib / log),
    `sources` (the iteration order), `models` ({model name: (source, calibrated)}) and `b_of(src, calibration
    record) -> b` (default: the b the calibration stage fitted) are the only things that differ."""
    P = P or paths(a.out)
    models = models or MODELS
    sources = sources or list(SOURCES)
    P["dir"].mkdir(parents=True, exist_ok=True)
    jobs = jobs_of(items)
    done = {(r["model"], r["set"], r["idx"], r["init"]) for r in read_jsonl(P["rows"])}
    calib = load_json(P["calib"]) if P["calib"].exists() else {}
    checked = False
    with P["rows"].open("a") as fh, P["log"].open("a") as flog:
        for src in sources:
            names = [m for m, (s, _) in models.items() if s == src]
            todo = {m: [j for j in jobs if (m, j[0]["set"], j[0]["idx"], j[1]) not in done] for m in names}
            if not any(todo.values()) and (src in calib or not any(models[m][1] for m in names)):
                print(f"[{src}] every generation present; skipped", flush=True)
                continue
            policy = factory(src)
            if not checked:
                check_ids(policy)
                checked = True
            for m in names:
                if not models[m][1] and todo[m]:
                    run_jobs(m, policy, todo[m], Path(a.eval_frames), fh, flog, None)
            cal_models = [m for m in names if models[m][1]]
            if cal_models:
                proc = VerdictOffset(*stop_go)
                remove = install_offset(policy, proc)
                if src not in calib:
                    calib[src] = calibrate(policy, proc, cal_jobs, Path(a.f1_data), train_texts)
                    P["calib"].write_text(json.dumps(calib, indent=1))
                proc.b = float(calib[src]["b"]) if b_of is None else float(b_of(src, calib[src]))
                for m in cal_models:
                    if todo[m]:
                        run_jobs(m, policy, todo[m], Path(a.eval_frames), fh, flog, proc)
                remove()
            del policy
            free_cuda()


def offset_stats(rows: list[dict], log: list[dict], calib: dict, models: dict | None = None, b_of=None) -> dict:
    """Per calibrated model, from the offset log checked against the rows: every row has one log line with its
    source's b; the row's decision token is the logged argmax after the offset. `models` / `b_of` are round 5's hooks,
    exactly as in `run_stage0`."""
    by_key = {}
    for e in log:
        k = (e["model"], e["set"], e["idx"], e["init"])
        if k in by_key:
            raise SystemExit(f"two offset-log lines for {k}")
        by_key[k] = e
    out = {}
    for m, (src, cal) in (models or MODELS).items():
        if not cal:
            continue
        b_want = float(calib[src]["b"]) if b_of is None else float(b_of(src, calib[src]))
        mr = [r for r in rows if r["model"] == m]
        n_open = n_top2 = flips = fh = fb = 0
        for r in mr:
            e = by_key.get((m, r["set"], r["idx"], r["init"]))
            if e is None or e["b"] != b_want:
                raise SystemExit(f"{m}: no offset-log line with b = {b_want} for {r['set']}/{r['idx']}")
            if e["first"] == OPEN:
                if len(r["gen_ids"]) < 2 or r["gen_ids"][1] != e["chosen"]:
                    raise SystemExit(f"{m}: row {r['set']}/{r['idx']}/i{r['init']} decision is not the logged one")
                n_open += 1
                n_top2 += bool(e["top2"])
                if e["raw"] != e["chosen"]:
                    flips += 1
                    fh += int(r["harm"] == 1)
                    fb += int(r["harm"] == 0)
        out[m] = {"n": len(mr), "n_open": n_open, "n_top2": n_top2, "flips": flips, "flips_harm": fh,
                  "flips_benign": fb}
    extra = set(by_key) - {(r["model"], r["set"], r["idx"], r["init"]) for r in rows}
    if extra:
        raise SystemExit(f"{len(extra)} offset-log lines without a row")
    return out


def evaluate(rows, log, calib, ref14, ref15, items, hold_ids=None):
    check_rows(rows, items, list(MODELS), VERDICT_IDS_REF, hold_ids=hold_ids)
    summ = summarize(rows, tuple(MODELS))
    refs = {"G14": summarize(ref14, G14_MODELS), "G15": summarize(ref15, G15_ARMS)}
    for ref in refs.values():
        for s in ref.values():
            s["eligible"] = False
    ost = offset_stats(rows, log, calib)
    go = {m: int(summ[m]["eligible"]) for m in PRIORITY}
    return summ, refs, ost, go, [m for m in PRIORITY if go[m]]


def calib_line(src: str, c: dict) -> str:
    rb, ra = c["stop_rate_b"], c["stop_rate_all"]
    return (f"  calibration ({src}): b = {c['b']:+.3f} = -(q{c['quantile']:.2f} of logit[STOP] - logit[GO] = "
            f"{c['q']:+.3f}) over the {c['n_b']} calibration prefills whose sentence is not a v4 training text; STOP "
            f"rate after the offset on those {rb['predicted']:.3f} (generated {rb['generated']:.3f}), on all "
            f"{c['n_open']}/{c['n']} {ra['predicted']:.3f} (generated {ra['generated']:.3f}); decision between [STOP] "
            f"and [GO] on {c['n_top2']}, {c['n_boundary']} at the boundary; {c['train_overlap']['sentences']} "
            f"calibration sentences ({c['train_overlap']['prefills']} prefills) are v4 training texts")


def table_lines(summ, refs, ost, calib, smoke: bool) -> list[str]:
    L = ["G16 stage 0: first policy call, RAW command, no gate / tag / parent, the G14 stage-0 jobs; 'b' models add "
         "the calibrated offset b to the [STOP] logit at the verdict decision position (argmax elsewhere); G14 and "
         "G15 rows reused" + ("; SMOKE (not a result)" if smoke else "")]
    for m, (src, cal) in MODELS.items():
        s = summ[m]
        L += ["", f"== {m} ({SOURCES[src]['text']}{', calibrated offset' if cal else ', argmax'}; adapter "
                  f"{SOURCES[src]['name']}): {s['n_calls']} calls, invalid rate {s['invalid_rate']:.3f}"]
        L += model_block(s, True)
        if cal:
            o = ost[m]
            L.append(calib_line(src, calib[src]))
            L.append(f"  offset in evaluation: applied on {o['n_open']}/{o['n']} generations (first token '['), "
                     f"decision between [STOP] and [GO] on {o['n_top2']}; GO -> STOP flips {o['flips']} (harm "
                     f"{o['flips_harm']}, benign {o['flips_benign']})")
        L.append(f"G16 STAGE 0 {m}: rules {'PASS' if s['rules_pass'] else 'FAIL'}, G11 confirmation "
                 f"{'PASS' if s['confirm_pass'] else 'FAIL'}, G13 target {'MET' if s['g13_target_met'] else 'MISSED'} -> "
                 + ("SIMULATOR ELIGIBLE" if s["eligible"] else "not eligible"))
    L += ["", "== hold rate on the rule, confirmation and target pools; G14 / G15 rows reused (never eligible here)",
          f"  {'model':<7s}{'recipe':<62s}" + "".join(f"{p:>16s}" for _, p in COMPARE) + f"{'invalid':>9s}"]
    both = ([(m, G14_TEXT[m], refs["G14"][m]) for m in G14_MODELS]
            + [(m, "G15 " + G15_RECIPE[m]["text"], refs["G15"][m]) for m in G15_ARMS]
            + [(m, SOURCES[src]["text"] + (" + offset" if cal else ""), summ[m]) for m, (src, cal) in MODELS.items()])
    for m, text, s in both:
        vals = [s["sets"][st].get(pool, {}).get("hold", float("nan")) for st, pool in COMPARE]
        L.append(f"  {m:<7s}{text:<62s}" + "".join(f"{v:16.3f}" for v in vals) + f"{s['invalid_rate']:9.3f}")
    elig = [m for m in PRIORITY if summ[m]["eligible"]]
    L += ["", f"G16 STAGE 0: simulator-eligible models in priority order {elig}" if elig else
          "G16 STAGE 0 NO-GO: no model passes the rules and the G11 confirmation; no simulator arm"]
    return L


def finish(a, items, ref14, ref15, append_results: bool) -> int:
    P = paths(a.out)
    rows, log, calib = read_jsonl(P["rows"]), read_jsonl(P["log"]), load_json(P["calib"])
    summ, refs, ost, go, elig = evaluate(rows, log, calib, ref14, ref15, items)
    lines = table_lines(summ, refs, ost, calib, a.smoke)
    Report()("\n".join(lines))
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    doc = {"exp": EXP, "ts": stamp, "smoke": bool(a.smoke), "models": {m: {"source": s, "offset": c,
                                                                            "adapter": SOURCES[s]["name"]}
                                                                        for m, (s, c) in MODELS.items()},
           "sources": SOURCES, "calibration": {s: {k: v for k, v in c.items() if k not in ("ids", "d", "in_b")}
                                               for s, c in calib.items()},
           "verdict_ids": VERDICT_IDS_REF, "n_items": len(items), "n_jobs": len(jobs_of(items)), "summary": summ,
           "summary_ref": refs, "offset": ost, "go": go, "eligible": elig}
    P["summary"].write_text(json.dumps(doc, indent=1, default=float))
    P["table"].write_text("\n".join(lines) + "\n")
    P["go"].write_text(json.dumps({**go, "eligible": elig}, indent=1))
    if append_results and not a.smoke:
        with (RESULTS_DIR / "results.jsonl").open("a") as f:
            for m, s in summ.items():
                for st, pools in s["sets"].items():
                    for pool, v in pools.items():
                        f.write(json.dumps({"exp": EXP, "ts": stamp, "model": m, "adapter": SOURCES[MODELS[m][0]]["name"],
                                            "offset_b": calib[MODELS[m][0]]["b"] if MODELS[m][1] else None,
                                            "set": st, "pool": pool, **v}) + "\n")
    print("wrote\n" + "\n".join(file_line(P[k]) for k in ("rows", "log", "calib", "summary", "table", "go")), flush=True)
    return 0 if (elig or a.smoke) else NO_GO_EXIT


def load_inputs(a) -> tuple:
    items = build_items(Path(a.f1_data), Path(a.twins), Path(a.demo_prompts), a.smoke)
    ref14, _ = reference_rows(a.g14_stage0, items, "g14", G14_MODELS)
    ref15, _ = reference_rows(a.g15_stage0, items, "g15", G15_ARMS)
    cal_jobs = calibration_jobs(Path(a.f1_data), Path(a.novel_benign), a.smoke)
    train_texts = {r["raw_instruction"] for r in read_jsonl(a.train_rows)}
    if not train_texts:
        raise SystemExit(f"{a.train_rows}: no training rows")
    n_b = sum(j["text"] not in train_texts for j in cal_jobs)
    if not a.smoke and n_b != N_B_FULL:
        raise SystemExit(f"{n_b} calibration prefills outside the v4 training texts; the G16 NOTE fixes {N_B_FULL}")
    return items, ref14, ref15, cal_jobs, train_texts


def stage_stage0(a) -> int:
    P = paths(a.out)
    rs = load_json(a.rows_summary)
    if bool(rs["smoke"]) != bool(a.smoke) or rs["stub_generator"]:
        raise SystemExit(f"{a.rows_summary}: smoke {rs['smoke']}, stub {rs['stub_generator']}; this run has smoke "
                         f"{a.smoke} and needs real rows")
    if rs["verdict_ids"] != VERDICT_IDS_REF:
        raise SystemExit(f"rows summary verdict ids {rs['verdict_ids']} != {VERDICT_IDS_REF}")
    stop_go = decision_ids(rs["verdict_ids"])
    rdoc_path = {32: Path(a.restore32), RANK: P["restore"]}
    rdoc = {r: load_json(p) for r, p in rdoc_path.items()}
    adapters = {src: P["adapters"] / rc["name"] for src, rc in SOURCES.items()}
    missing = [str(p) for p in adapters.values() if not (p / "adapter_config.json").exists()]
    if missing:
        raise SystemExit(f"adapters missing: {missing}")
    if a.smoke:
        print("[smoke] 2 sentences per pool, both frames; 10 calibration prefills. NOT results.", flush=True)
    items, ref14, ref15, cal_jobs, train_texts = load_inputs(a)
    print(f"[jobs] {len(jobs_of(items))} generations per model, {len(MODELS)} models; {len(cal_jobs)} calibration "
          f"prefills; decision ids STOP {stop_go[0]} / GO {stop_go[1]} after '[' {OPEN}", flush=True)
    forbid_parent()
    rep = Report()

    def check_ids(policy):
        if decision_ids(verdict_token_ids(policy, rep)) != stop_go:
            raise SystemExit("the policy tokenizer's verdict ids differ from the rows stage")

    steps = int(rs["steps_all_arms"])

    def factory(src):
        rc = SOURCES[src]
        ctx = {"steps": steps, "restore_doc": rdoc_path.get(rc["restore"]), "restore": rdoc.get(rc["restore"])}
        return load_arm(a, src, str(adapters[src]), ctx, rc=rc)

    run_stage0(a, items, cal_jobs, train_texts, factory, check_ids, stop_go)
    return finish(a, items, ref14, ref15, append_results=True)


def stage_restore(a) -> int:
    return run_restore(a, paths(a.out), RANK, "G16")


# --------------------------------------------------------------------------------- CPU self-check (on the GPU run's outputs)

def self_check_cpu(folder, a) -> None:
    """Recompute rows checks, every summary, the offset statistics, b from the stored calibration differences, the
    GO map and the table."""
    a.out = folder
    P = paths(folder)
    doc = load_json(P["summary"])
    a.smoke = bool(doc["smoke"])
    items, ref14, ref15, cal_jobs, train_texts = load_inputs(a)
    rows, log, calib = read_jsonl(P["rows"]), read_jsonl(P["log"]), load_json(P["calib"])
    text_of = {j["id"]: j["text"] for j in cal_jobs}
    bad = []
    for src, c in calib.items():
        d, m = np.asarray(c["d"], dtype=np.float64), np.asarray(c["in_b"], dtype=bool)
        if not set(c["ids"]) <= set(text_of) or len(c["ids"]) != c["n_open"] or c["n"] != len(cal_jobs):
            bad.append(f"{src}: calibration ids are not the G4 calibration prefills")
            continue
        if [text_of[i] not in train_texts for i in c["ids"]] != c["in_b"] or int(m.sum()) != c["n_b"]:
            bad.append(f"{src}: the b set is not the calibration prefills outside the v4 training texts")
        if (c["b"] != -float(np.quantile(d[m], QUANTILE)) or c["stop_rate_b"]["predicted"] != float((d[m] + c["b"] > 0).mean())
                or c["stop_rate_all"]["predicted"] != float((d + c["b"] > 0).mean())):
            bad.append(f"{src}: b or a STOP rate does not follow from the stored differences")
    summ, refs, ost, go, elig = evaluate(rows, log, calib, ref14, ref15, items,
                                         hold_ids=load_json(a.hold_chunk_ref)["gen_ids"])
    for name, got, stored in (("G16", summ, doc["summary"]), ("G14", refs["G14"], doc["summary_ref"]["G14"]),
                              ("G15", refs["G15"], doc["summary_ref"]["G15"]), ("offset", ost, doc["offset"])):
        if json.dumps(got, default=float, sort_keys=True) != json.dumps(stored, default=float, sort_keys=True):
            bad.append(f"recomputed {name} summary differs")
    if go != doc["go"] or elig != doc["eligible"] or load_json(P["go"]) != {**go, "eligible": elig}:
        bad.append(f"GO map recomputed {go} / {elig}; stored {doc['go']} / {doc['eligible']}")
    if P["table"].read_text() != "\n".join(table_lines(summ, refs, ost, calib, a.smoke)) + "\n":
        bad.append("the table file differs from the table recomputed from the rows")
    print(f"[self-check] {folder}: {len(rows)} generations over {len(MODELS)} models x {len(jobs_of(items))} jobs, "
          f"{len(log)} offset-log lines, calibration of {sorted(calib)}; eligible {elig}; smoke {int(a.smoke)}")
    if bad:
        raise SystemExit("SELF-CHECK FAIL:\n  " + "\n  ".join(bad))
    print("[self-check] SELF-CHECK PASS: rows, offsets, b, every per-pool rate, rule, confirmation, target, verdict "
          "statistic, the G14 / G15 references, the GO map and the table reproduced")


# --------------------------------------------------------------------------------- self-test (CPU, no GPU)

VOCAB, END, HOLD_T, MOVE_T, THIRD, PROMPT0 = 50700, 50699, 201, 301, 7, 20000
TINY128 = {"n_layers": 2, "d": 160, "dkv": 32, "dff": 192, "vocab": 200, "extra": 5}


def _expect_fail(fn, what: str, needle: str, exc=SystemExit) -> None:
    try:
        fn()
    except exc as e:
        if needle not in str(e):
            raise AssertionError(f"{what}: stopped by another check: {e}")
        print(f"[self-test]   caught: {what}")
        return
    raise AssertionError(f"not caught: {what}")


class _Script:
    """Fixes every step of a generation: "[" (or `first` instead), then [STOP] / [GO] logits +-d/2 (a third token
    lower), "]", a hold token after a generated [STOP] or a motion token otherwise, then END."""

    def __init__(self, d: float, first: int = OPEN):
        self.d, self.first, self.p = float(d), first, None

    def __call__(self, input_ids, scores):
        import torch
        if self.p is None:
            self.p = input_ids.shape[1]
        step = input_ids.shape[1] - self.p
        new = torch.full_like(scores, -1e4)
        stop, go = VERDICT_IDS_REF[STOP][1], VERDICT_IDS_REF[GO][1]
        if step == 0:
            new[:, self.first] = 0.0
        elif step == 1:
            new[:, stop], new[:, go], new[:, THIRD] = self.d / 2, -self.d / 2, -abs(self.d) - 5
        elif step == 2:
            new[:, CLOSE] = 0.0
        elif step == 3:
            new[:, HOLD_T if int(input_ids[0, self.p + 1]) == stop else MOVE_T] = 0.0
        else:
            new[:, END] = 0.0
        return new


class _ScriptLM:
    """`generate` over a tiny random GPT-2 with a _Script first and the caller's processors after it."""

    def __init__(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
        torch.manual_seed(0)
        self.lm = GPT2LMHeadModel(GPT2Config(vocab_size=VOCAB, n_positions=32, n_embd=8, n_layer=1, n_head=1)).eval()
        self.d: dict[int, float] = {}
        self.first: dict[int, int] = {}

    def generate(self, input_ids, max_new_tokens, do_sample=False, logits_processor=None, **_kw):
        import torch
        from transformers import LogitsProcessorList
        k = int(input_ids[0, 0])
        lp = LogitsProcessorList([_Script(self.d[k], self.first.get(k, OPEN))] + list(logits_processor or []))
        with torch.no_grad():
            return self.lm.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                    max_new_tokens=max_new_tokens, do_sample=do_sample, logits_processor=lp,
                                    pad_token_id=END, eos_token_id=END)


class _ScriptPolicy:
    """NoraPolicy stand-in: `_inputs` encodes the sentence as one prompt token (its d from `d_fn`), `act` generates
    through `model.generate` and returns a hold action exactly when the hold token was generated."""

    def __init__(self, d_fn, seen: list | None = None):
        self.model, self.d_fn, self.ids, self.seen = _ScriptLM(), d_fn, {}, seen

    def _inputs(self, image, text):
        import torch
        k = PROMPT0 + self.ids.setdefault(text, len(self.ids))
        self.model.d[k] = self.d_fn(text)
        if self.seen is not None:
            self.seen.append(text)
        return {"input_ids": torch.tensor([[k, 11, 12]])}

    def act(self, image, text, max_new_tokens: int = 8):
        inputs = self._inputs(image, text)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = [int(t) for t in out[0, inputs["input_ids"].shape[1]:]]
        a = np.zeros((5, 7))
        a[:, 6] = 1.0
        if HOLD_T not in gen:
            a[:, 0] = 0.2
        return {"action": a, "gen_ids": gen, "n_text": 3, "invalid": False}


def _test_offset() -> None:
    """The processor under a real transformers generate: moves only the decision logit, only after '[', and the continuation
    follows the generated verdict."""
    import torch
    from transformers import LogitsProcessorList
    stop, go = decision_ids(VERDICT_IDS_REF)
    assert (stop, go) == (50669, 15513)
    _expect_fail(lambda: decision_ids({STOP: [58, 50669, 60], GO: [58, 50669, 60]}), "identical decision ids",
                 "x != y")

    class Spy:
        def __init__(self):
            self.rec = []

        def __call__(self, input_ids, scores):
            self.rec.append(scores.clone())
            return scores

    pol = _ScriptPolicy(lambda t: {"go": -10.0, "flip": -3.0, "stop": 4.0}[t])
    lm = pol.model

    def gen(text, procs):
        ids = pol._inputs(None, text)["input_ids"]
        k = int(ids[0, 0])
        out = lm.lm.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=8, do_sample=False,
                             logits_processor=LogitsProcessorList([_Script(lm.d[k], lm.first.get(k, OPEN))] + procs),
                             pad_token_id=END, eos_token_id=END)
        return out[0, ids.shape[1]:].tolist()

    for text, b, want in (("go", 5.0, go), ("flip", 5.0, stop), ("stop", -2.0, stop), ("stop", -6.0, go)):
        s0, s1, p = Spy(), Spy(), VerdictOffset(stop, go, b)
        p.begin()
        g0 = gen(text, [s0])
        g1 = gen(text, [p, s1])
        p.end(len(g1))
        e = p.log[-1]
        assert g1[:3] == [OPEN, want, CLOSE] and g1[3] == (HOLD_T if want == stop else MOVE_T), (text, b, g1)
        assert e["applied_at"] == [1] and e["chosen"] == want and e["top2"] and abs(e["d"] - lm.d[PROMPT0 + pol.ids[text]]) < 1e-5
        diff = (s1.rec[1] - s0.rec[1]).abs()
        assert float(diff[0, stop]) == abs(b) and float(diff.sum()) == abs(b), "only the [STOP] logit moves, by b"
        n_same = 3 if want == g0[1] else 1          # after a flipped decision the continuation legitimately differs
        assert all(torch.equal(s0.rec[i], s1.rec[i]) for i in [0] + list(range(2, n_same + 1))), "other positions moved"
        assert (g0 == g1) == (want == g0[1])
    # no "[" first: the processor never applies, the generation is unchanged
    lm.first[PROMPT0 + pol.ids["flip"]] = THIRD
    p = VerdictOffset(stop, go, 50.0)
    p.begin()
    g = gen("flip", [p])
    p.end(len(g))
    assert p.log[-1]["applied_at"] == [] and p.log[-1]["d"] is None and g[0] == THIRD
    lm.first.clear()
    # install_offset: through policy.model.generate, one record per act, removal restores the originals
    p = VerdictOffset(stop, go, 5.0)
    remove = install_offset(pol, p)
    r = pol.act(None, "flip")
    assert r["gen_ids"][1] == stop and HOLD_T in r["gen_ids"] and len(p.log) == 1 and p.log[0]["raw"] == go
    _expect_fail(lambda: pol.model.generate(input_ids=torch.tensor([[PROMPT0, 1], [PROMPT0, 1]]), max_new_tokens=2),
                 "a batch of two", "one sequence per call", RuntimeError)
    p.cur = None
    remove()
    assert "generate" not in pol.model.__dict__ and "act" not in pol.__dict__
    assert pol.act(None, "flip")["gen_ids"][1] == go
    print("[self-test] offset (tiny GPT-2, HF generate): b moves only the [STOP] logit at the decision position "
          "(bitwise-equal scores elsewhere), never applies without '[' first, the continuation follows the verdict, "
          "install/remove wrap generate and act")


def _test_calibration(a) -> None:
    """The 360 probe-gate calibration prefills; b only from them; STOP rate 0.10 after the offset; agreement preserved in evaluation."""
    jobs = calibration_jobs(Path(a.f1_data), Path(a.novel_benign), False)
    assert len(jobs) == 360 and len({j["id"] for j in jobs}) == 360
    assert all(j["frame"].startswith("home_frames/") and int(j["frame"].split("_i")[1][:-4]) < 10 for j in jobs)
    assert Counter(j["pool"] for j in jobs) == {"BENIGN_PARA_TRAIN": 200, "NOVEL_BENIGN": 140, "TRAINED": 20}
    items = build_items(Path(a.f1_data), Path(a.twins), Path(a.demo_prompts), False)
    shared = {j["text"] for j in jobs} & {it["text"] for it in items}
    assert shared == {it["text"] for it in items if it["pool"] == "TRAINED"} and len(shared) == 10, len(shared)
    assert len(calibration_jobs(Path(a.f1_data), Path(a.novel_benign), True)) == SMOKE_CALIB
    print("[self-test] calibration prefills: 360 = 180 G4 sentences x 2 home frames of init 0-9 (TRAINED 20, "
          "BENIGN_PARA_TRAIN 200, NOVEL_BENIGN 140); the only sentences shared with the evaluation are the 10 trained "
          "instructions (G4 design, on other frames)")
    stop, go = decision_ids(VERDICT_IDS_REF)

    def d_fn(text):
        h = int(sha(text)[:6], 16) / 16 ** 6
        return (3.0 if text.startswith("H") else -8.0) + 6.0 * h

    seen = []
    pol = _ScriptPolicy(d_fn, seen)
    p = VerdictOffset(stop, go)
    remove = install_offset(pol, p)
    real_train = {r["raw_instruction"] for r in read_jsonl(a.train_rows)}
    assert sum(j["text"] not in real_train for j in jobs) == N_B_FULL, "the G16 NOTE's 212 non-training prefills"
    train = {j["text"] for j in jobs[:40]}
    c = calibrate(pol, p, jobs, Path(a.f1_data), train)
    d = np.asarray([d_fn(j["text"]) for j in jobs])
    nt = np.asarray([j["text"] not in train for j in jobs])
    assert set(seen) == {j["text"] for j in jobs} and len(seen) == 720, "calibration reads calibration prefills only"
    assert abs(c["b"] + np.quantile(d[nt], QUANTILE)) < 1e-4 and c["n_open"] == c["n_top2"] == 360 and c["n_b"] == 320
    assert c["stop_rate_b"]["predicted"] == c["stop_rate_b"]["generated"] == 0.1, c["stop_rate_b"]
    assert c["stop_rate_all"]["predicted"] == c["stop_rate_all"]["generated"] == float((d + c["b"] > 0).mean())
    assert c["train_overlap"] == {"sentences": 20, "prefills": 40} and c["in_b"] == nt.tolist()
    # b ignores the training prefills: moving only their decisions far toward STOP leaves b unchanged
    pol2 = _ScriptPolicy(lambda t: d_fn(t) + (50.0 if t in train else 0.0))
    p2 = VerdictOffset(stop, go)
    remove2 = install_offset(pol2, p2)
    c2 = calibrate(pol2, p2, jobs, Path(a.f1_data), train)
    remove2()
    assert c2["b"] == c["b"] and c2["stop_rate_b"] == c["stop_rate_b"]
    assert c2["stop_rate_all"]["predicted"] == (32 + 40) / 360, c2["stop_rate_all"]
    # evaluation with the offset: benign sentences above the calibration quantile flip; the hold follows the verdict
    tmp = Path(tempfile.mkdtemp(prefix="g16_selftest_"))
    try:
        sub = [(it, init) for it, init in jobs_of(items)][:120]
        p.b = c["b"]
        with (tmp / "rows.jsonl").open("w") as fh, (tmp / "log.jsonl").open("w") as flog:
            run_jobs("I2b", pol, sub, SHIPPED_EVAL_FRAMES, fh, flog, p)
        rows, log = read_jsonl(tmp / "rows.jsonl"), read_jsonl(tmp / "log.jsonl")
        assert len(rows) == len(log) == 120 and all(e["b"] == c["b"] for e in log)
        assert all((r["verdict"] == STOP) == (r["cls"] == "hold") for r in rows), "verdict / behaviour agreement"
        flips = [e for e in log if e["raw"] != e["chosen"]]
        assert flips and all(e["raw"] == go and e["chosen"] == stop for e in flips)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    remove()
    print(f"[self-test] calibration: b = -q0.90 over the non-training calibration prefills only ({c['b']:+.3f}; 320 "
          f"of 360 here, {N_B_FULL} on the real v4 rows), unchanged when the training prefills move; STOP rate after "
          f"the offset 0.100 on the b set and {c['stop_rate_all']['predicted']:.3f} on all 360, predicted = generated; "
          f"{len(flips)} GO -> STOP flips in 120 evaluation calls, every hold under [STOP], every motion under [GO]")


def _test_restore128(tmp: Path) -> dict:
    import torch
    from safetensors.torch import save_file
    from single_model_judges import _tiny_model
    t = TINY128
    par = _tiny_model(t["n_layers"], t["d"], t["dkv"], t["dff"], t["vocab"], 0, True, 2)
    folder = tmp / "parent"
    folder.mkdir(parents=True)
    save_file({n: p.detach().clone().contiguous() for n, p in par.named_parameters() if n != "lm_head.weight"},
              str(folder / "model-00001.safetensors"))
    base = _tiny_model(t["n_layers"], t["d"], t["dkv"], t["dff"], t["vocab"], t["extra"], True, 1)
    fac, rec = build_factors(base, list(base.model.layers), ParentTensors(folder), "cpu", Report(), t["vocab"], RANK)
    P = paths(tmp / "g16")
    doc = write_restore(P, fac, {"exp": f"{EXP}/restore", "checkpoint": "tiny/nora", "base_adapter": None, **rec},
                        Report())
    assert doc["rank"] == RANK and fac["L0.q_proj.US"].shape == (t["d"], RANK) and fac["emb.r128.US"].shape[1] == RANK
    assert fac["L0.k_proj.US"].shape == (t["dkv"], t["dkv"]) and 0 < doc["fraction_captured"]["min"] < 1
    m = _tiny_model(t["n_layers"], t["d"], t["dkv"], t["dff"], t["vocab"], t["extra"], True, 1)
    r = restore_base(m, list(m.model.layers), P["restore"], "tiny/nora", "cpu")
    assert r["rank"] == RANK and r["fingerprint_after"] == doc["fingerprint_after"]
    assert torch.equal(m.model.layers[1].mlp.up_proj.weight, base.model.layers[1].mlp.up_proj.weight)
    # the recipe check: I8 needs a rank-128 record, I7 a rank-32 one
    meta = {"out": "x", "rows": "results/g14/rows/train_verdict_parent.jsonl", "steps": 1052, "checkpoint": "tiny/nora",
            "args": {**LORA_ARGS, "align_rows_ref": None}, "base_restore": r}
    check_meta("I8", meta, 1052, "tiny/nora", doc, SOURCES["I8"])
    d32 = {**doc, "rank": 32}
    _expect_fail(lambda: check_meta("I8", meta, 1052, "tiny/nora", d32, SOURCES["I8"]), "I8 against a rank-32 doc",
                 "differs from the restore stage")
    _expect_fail(lambda: check_meta("I8", {**meta, "base_restore": {**r, "rank": 32}}, 1052, "tiny/nora", doc,
                                    SOURCES["I8"]), "I8 trained on a rank-32 base", "differs from the restore stage")
    _expect_fail(lambda: check_meta("I7", {**meta, "align": {"weight": 1.0, "stub_targets": False}, "args":
                                           {**meta["args"], "align_rows_ref": "r/train_free_align.jsonl"}},
                                    1052, "tiny/nora", doc, SOURCES["I7"]), "I7 trained on the rank-128 base",
                 "differs from the restore stage")
    _expect_fail(lambda: check_meta("I2", meta, 1052, "tiny/nora", None, SOURCES["I2"]), "I2 on a restored base",
                 "trained on plain NORA")
    # load order for I8: plain load, rank-128 restore, then the adapter
    events = []

    class FakePolicy:
        def __init__(self, checkpoint, unnorm_key=None, time_horizon=None, adapter=None):
            mm = _tiny_model(t["n_layers"], t["d"], t["dkv"], t["dff"], t["vocab"], t["extra"], True, 1)
            self.nora = SimpleNamespace(model=mm)
            self.model, self.layers, self.device = mm, list(mm.model.layers), "cpu"
            events.append(("load", adapter))

    def merge(model, adapter):
        events.append(("merge", fingerprint(restore_targets(model, list(model.model.layers)), t["vocab"])["all"]))
        return model

    ad = tmp / "adapters" / SOURCES["I8"]["name"]
    ad.mkdir(parents=True)
    (ad / "train_meta.json").write_text(json.dumps(meta))
    load_arm(SimpleNamespace(checkpoint="tiny/nora", suite="libero_goal", time_horizon=5), "I8", str(ad),
             {"steps": 1052, "restore_doc": P["restore"], "restore": doc},
             SimpleNamespace(policy=FakePolicy, merge=merge, layers_of=lambda mm: list(mm.model.layers)), SOURCES["I8"])
    assert events == [("load", None), ("merge", doc["fingerprint_after"]["all"])]
    print(f"[self-test] rank-128 restore: factors of W_parent - W_base (tiny model, {doc['n_matrices']} matrices, "
          f"embedding rank 128), merged by restore_base with both fingerprints; recipe ranks enforced; I8 loads plain, "
          f"restores at rank 128, then merges its adapter")
    return doc


def _test_pipeline(tmp: Path, a) -> None:
    """The whole stage 0 on the smoke jobs with scripted policies: argmax and offset models, calibration stored and
    reused, rows + offset log, references, table, self-check, tampering."""
    ns = argparse.Namespace(**{**vars(a), "out": str(tmp), "smoke": True, "eval_frames": str(SHIPPED_EVAL_FRAMES)})
    items, ref14, ref15, cal_jobs, train_texts = load_inputs(ns)
    harm = {it["text"]: it["harm"] for it in items}
    offs = {"I2": -9.0, "I4": -7.0, "I6": -5.0, "I7": 0.0, "I8": -20.0}
    loads = Counter()

    def factory(src):
        loads[src] += 1

        def d_fn(text):
            h = int(sha(text)[:6], 16) / 16 ** 6
            return offs[src] + (8.0 if harm.get(text) else -4.0) + h
        return _ScriptPolicy(d_fn)

    stop_go = decision_ids(VERDICT_IDS_REF)
    run_stage0(ns, items, cal_jobs, train_texts, factory, lambda _p: None, stop_go)
    rc = finish(ns, items, ref14, ref15, append_results=False)
    P = paths(tmp)
    doc = load_json(P["summary"])
    assert rc == 0 and doc["eligible"][:1] == ["I7"] and "I8" not in doc["eligible"], doc["eligible"]
    assert set(load_json(P["calib"])) == {"I2", "I4", "I6", "I7", "I8"} and sum(loads.values()) == 5
    assert all(doc["summary"][m]["verdict"]["agree"] == 1.0 for m in MODELS)
    assert doc["summary"]["I8"]["sets"]["eval"]["HARM_HELDOUT"]["hold"] == 0.0
    ns_check = argparse.Namespace(**vars(a))
    self_check_cpu(tmp, ns_check)
    before_rows, before_calib = P["rows"].read_text(), P["calib"].read_text()
    run_stage0(ns, items, cal_jobs, train_texts, factory, lambda _p: None, stop_go)
    assert sum(loads.values()) == 5 and P["rows"].read_text() == before_rows and P["calib"].read_text() == before_calib
    print(f"[self-test] pipeline (smoke jobs, scripted policies): 7 models from 5 loads, calibration stored and reused, "
          f"eligible {doc['eligible']} in priority order, self-check PASS, resume loads nothing")
    lines = P["log"].read_text().splitlines()
    e0 = json.loads(lines[0])
    P["log"].write_text("\n".join([json.dumps({**e0, "chosen": e0["raw"] if e0["chosen"] != e0["raw"] else 1})]
                                  + lines[1:]) + "\n")
    _expect_fail(lambda: self_check_cpu(tmp, argparse.Namespace(**vars(a))), "an offset-log decision edited",
                 "decision is not the logged one")
    P["log"].write_text("\n".join(lines) + "\n")
    cal = json.loads(before_calib)
    cal["I4"]["b"] += 0.5
    P["calib"].write_text(json.dumps(cal))
    _expect_fail(lambda: self_check_cpu(tmp, argparse.Namespace(**vars(a))), "a calibration offset edited",
                 "no offset-log line with b")
    P["calib"].write_text(before_calib)
    self_check_cpu(tmp, argparse.Namespace(**vars(a)))


def self_test(a) -> None:
    for p in (SHIPPED_EVAL_FRAMES, Path(a.g14_stage0) / "stage0_g14.jsonl", Path(a.g15_stage0) / "stage0_g15.jsonl",
              Path(a.train_rows)):
        if not Path(p).exists():
            raise SystemExit(f"missing {p}: the self-test runs on pulled pod-8b / pod-14 / pod-15 files")
    tmp = Path(tempfile.mkdtemp(prefix="g16_selftest_"))
    try:
        _test_offset()
        _test_calibration(a)
        _test_restore128(tmp / "restore")
        _test_pipeline(tmp / "stage0", a)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[self-test] OK", flush=True)


# --------------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("restore", "stage0"), default=None)
    ap.add_argument("--out", default=str(RESULTS_DIR / "g16"))
    ap.add_argument("--rows-summary", default=str(RESULTS_DIR / "g14/rows/build_summary_g14.json"))
    ap.add_argument("--train-rows", default=str(RESULTS_DIR / "g14/rows/train_verdict_parent.jsonl"),
                    help="the approved I2 rows: their sentences are counted in the calibration set (report only)")
    ap.add_argument("--restore32", default=str(RESULTS_DIR / "g15/restore/restore_base_g15.json"),
                    help="round 2's rank-32 restore doc, beside its factor file")
    ap.add_argument("--g14-stage0", default=str(RESULTS_DIR / "g14_ref/stage0"))
    ap.add_argument("--g15-stage0", default=str(RESULTS_DIR / "g15_ref/stage0"))
    ap.add_argument("--f1-data", default=str(RESULTS_DIR / "f1/data"))
    ap.add_argument("--novel-benign", default=str(DATA_DIR / "novel_benign.txt"))
    ap.add_argument("--twins", default=str(DATA_DIR / "twins.jsonl"))
    ap.add_argument("--demo-prompts", default=str(DATA_DIR / "demo_prompts.txt"))
    ap.add_argument("--eval-frames", default=str(RESULTS_DIR / "raw/pod8/g1/home_frames_eval"))
    ap.add_argument("--hold-chunk-ref", default=str(RESULTS_DIR / "raw/pod6/c1/hold_chunk.json"))
    ap.add_argument("--checkpoint", default="declare-lab/nora-long-finetuned-libero-goal")
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--time-horizon", type=int, default=5)
    ap.add_argument("--smoke", action="store_true", help="stage0: 2 sentences per pool, 10 calibration prefills")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--self-check-cpu", default=None, metavar="OUT")
    a = ap.parse_args()
    if a.self_test or a.self_check_cpu:
        cpu_shim()
        for attr, run_path, shipped in (("f1_data", RESULTS_DIR / "f1/data", RESULTS_DIR / "raw/pod7/f1_full/data"),
                               ("g14_stage0", RESULTS_DIR / "g14_ref/stage0", SHIPPED_G14_STAGE0),
                               ("g15_stage0", RESULTS_DIR / "g15_ref/stage0", SHIPPED_G15_STAGE0),
                               ("train_rows", RESULTS_DIR / "g14/rows/train_verdict_parent.jsonl", SHIPPED_TRAIN_ROWS)):
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
    if a.stage == "restore":
        return stage_restore(a)
    if a.stage == "stage0":
        return stage_stage0(a)
    ap.error("one of --stage, --self-test, --self-check-cpu is required")
    return 2


if __name__ == "__main__":
    sys.exit(main())
