#!/usr/bin/env python3
"""Round 2 of the install search (code name G15; pre-registered): three install recipes on the SAME approved round-1
(G14) v4 rows (no new data), then the round-1 first-call evaluation for the three new arms. I4 is the parent-taught
verdict (PTV), seed 0.

Arms (trained by `train_install_lora.py`, 1,052 steps each, the round-1 LoRA recipe):
  I4 verdict_parent_align    the I2 rows + the I3 alignment term (weight 1.0; the targets are bound to the I3 rows file,
                             `--align-rows-ref` asserts that the I2 rows with an align_key are the same frames and sentences)
  I5 free_align_w3           the I3 recipe with alignment weight 3.0
  I6 restore_verdict_parent  the rank-32 parent-weight delta merged permanently into NORA BEFORE the LoRA (`--restore-base`),
                             then the I2 recipe (no alignment term)

The I6 base. The restore factors of the single-model judges (G6, `single_model_judges.py`; factors_r128.safetensors)
are the SVD of W_parent - W_deployed with deployed = NORA libero-goal + the merged `gated` stop adapter of the
parent-gated reference (G6 load_deployed; its restore table: "NORA + merged adapter
results/raw/pod8/g1/adapters/gated"). Rounds 1 and 2 train on plain NORA, so those factors
applied here would give W_NORA + [W_parent - W_NORA - D_stop]_32: the stop adapter's delta D_stop stays inside the
truncated SVD and the result is not the rank-32 restore of this base. The restore stage below therefore recomputes the
factors exactly as G6's restore stage does (full SVD per matrix, kept to rank 32, bf16; seeded rank-32 randomized SVD
of the embedding delta over the Qwen rows; the parent's small tensors; tied head, no head rows), against plain NORA,
and G6's `apply_restore` merges them.

  restore  GPU. Plain NORA + parent tensors read one at a time (safe_open; never the parent model object). Factors of
           every language-model weight delta, applied once to check their direction against the parent and to record
           the fingerprint (sha1 over every tensor the restore writes) of the plain base and of the merged base.
                                         -> <out>/restore/{factors_r32_g15.safetensors, restore_base_g15.json, .txt}
  stage0   GPU. The first policy call of I4, I5, I6 on the round-1 first-call jobs (G4 pools + G11 + G13, init 10 and 11
           frames, 538 per model). Each adapter's train_meta.json must match the pre-registered recipe; I6 loads plain
           NORA, merges the restore (fingerprints of the restore stage reproduced), THEN merges its LoRA. Round 1's base /
           free300 / I0..I3 rows are reused, checked and summarised alongside (never eligible here).
                                         -> <out>/stage0/{stage0_g15.jsonl, stage0_g15.json, stage0_table_g15.txt, g15_go.json}

  python3 experiments/4_parent_taught_verdict/train_verdict_alignment.py --self-test                                          # CPU, no GPU
  python3 experiments/4_parent_taught_verdict/train_verdict_alignment.py --stage restore --out results/g15                    # GPU
  python3 experiments/4_parent_taught_verdict/train_verdict_alignment.py --stage stage0 --out results/g15 [--smoke]           # GPU: 0 = eligible arm, 3 = NO-GO
  python3 experiments/4_parent_taught_verdict/train_verdict_alignment.py --self-check-cpu results/g15                         # CPU, on the saved rows

RULES, CONFIRMATION, TARGET and the simulator condition are round 1's (first_call_eval). Simulator priority: I4, I5, I6.
No log line printed here contains command text: sentences are identified by set, pool, index and hash only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from refusal.paths import DATA_DIR, RESULTS_DIR, add_experiment_dirs  # noqa: E402
add_experiment_dirs()
from build_training_rows import (ARM_NAME as G14_ARM_NAME, GO, STOP, VERDICT_IDS_REF,  # noqa: E402
                                        cpu_shim, sha1_file, verdict_token_ids)
from first_call_eval import MODELS as G14_MODELS  # noqa: E402
from first_call_eval import (build_items, check_rows, generate_all, jobs_of, load_json, model_block,  # noqa: E402
                                    summarize)
from probe_gate_first_call import Report, file_line, sha  # noqa: E402
from single_model_judges import (EMB_ROWS, ParentTensors, addon_mb, apply_restore,  # noqa: E402
                                                check_names, checksums, compute_embedding_factors, compute_factors,
                                                forbid_parent, parent_key_map, parent_small, restore_targets,
                                                write_report)

EXP = "g15_install"
ARMS = ("I4", "I5", "I6")                    # also the simulator priority (pre-registered)
ARM_NAME = {"I4": "verdict_parent_align", "I5": "free_align_w3", "I6": "restore_verdict_parent"}
VERDICT_ARMS = ("I4", "I6")
RECIPE = {"I4": {"rows": "train_verdict_parent.jsonl", "align_weight": 1.0, "align_ref": "train_free_align.jsonl",
                 "restore": None, "text": "I2 rows + I3 alignment term, weight 1.0"},
          "I5": {"rows": "train_free_align.jsonl", "align_weight": 3.0, "align_ref": None, "restore": None,
                 "text": "I3 recipe, alignment weight 3.0"},
          "I6": {"rows": "train_verdict_parent.jsonl", "align_weight": None, "align_ref": None, "restore": 32,
                 "text": "rank-32 parent restore merged into NORA, then the I2 recipe"}}
LORA_ARGS = {"lr": 1e-4, "rank": 16, "alpha": 32, "batch": 8, "seed": 0}      # the G14 recipe, asserted from train_meta
G14_TEXT = {"base": "NORA libero-goal, no adapter", "free300": "pod-8b free300",
            **{arm: f"G14 {name}" for arm, name in G14_ARM_NAME.items()}}
RANK = 32
RESTORE_RANKS = (32, 128)          # round 2: rank 32; round 3 (G16) adds rank 128 (same stage, same checks)
RESIDUAL_TOL = 0.02               # |measured - expected| of ||W_restored - W_parent|| / ||W_parent - W_base||
COMPARE = (("eval", "HARM_HELDOUT"), ("eval", "TWIN_BENIGN"), ("eval", "TRAINED"), ("g11", "HARM_NEW"),
           ("g11", "TWIN_BENIGN_NEW"), ("g13", "BENIGN_BLIND"))
NO_GO_EXIT = 3
G6_NOTE = ("G6's factors are the SVD of W_parent - (NORA + merged pod-8b stop adapter); on plain NORA they would "
           "leave the stop adapter's delta inside the truncated SVD, so G15 recomputes the same factors against "
           "plain NORA")
SHIPPED_G14_STAGE0 = RESULTS_DIR / "raw/pod14/g14_v4/stage0"
SHIPPED_EVAL_FRAMES = RESULTS_DIR / "raw/pod8/g1/home_frames_eval"


def paths(out) -> dict:
    o = Path(out)
    return {"restore_dir": o / "restore", "factors": o / f"restore/factors_r{RANK}_g15.safetensors",
            "restore": o / "restore/restore_base_g15.json", "restore_table": o / "restore/restore_table_g15.txt",
            "adapters": o / "adapters", "dir": o / "stage0", "rows": o / "stage0/stage0_g15.jsonl",
            "summary": o / "stage0/stage0_g15.json", "table": o / "stage0/stage0_table_g15.txt",
            "go": o / "stage0/g15_go.json"}


# --------------------------------------------------------------------------------- the I6 base: restore

def fingerprint(tg: dict, n_rows: int) -> dict:
    """sha1 over every tensor the restore writes (the decoder Linear weights, the small tensors, the embedding rows
    [0, n_rows)), in a fixed order, plus G6's three per-matrix checksums for the log."""
    import torch
    h = hashlib.sha1()
    tensors = ([tg["lin"][n].weight for n in sorted(tg["lin"])] + [tg["small"][n] for n in sorted(tg["small"])]
               + [tg["emb"][:n_rows]])
    for t in tensors:
        h.update(t.detach().cpu().contiguous().flatten().view(torch.uint8).numpy())
    return {"all": h.hexdigest(), **checksums(tg)}


def build_factors(model, layers, pt: ParentTensors, device, rep: Report, n_rows: int,
                  rank: int = RANK) -> tuple[dict, dict]:
    """G6's restore factors at `rank` against `model` (the plain base), merged into `model` once to check them: for
    G6's three check matrices, ||W_restored - W_parent|| / ||W_parent - W_base|| must equal sqrt(1 - the fraction of
    ||D||^2 the rank-r SVD captures) within RESIDUAL_TOL. Returns (factors, record)."""
    tg = restore_targets(model, layers)
    if not tg["tied"]:
        raise SystemExit("the policy's lm_head is untied; G6 found NORA tied and copied no head rows, and this restore "
                         "has no rule for an untied head")
    pkeys = parent_key_map(pt.keys(), tg["n_layers"])
    idx = list(range(tg["n_layers"]))
    fp_base = fingerprint(tg, n_rows)
    before = {n: tg["lin"][n].weight.detach().float().cpu() for n in check_names(tg)}
    fac, fracs, rel = compute_factors(tg, pt, pkeys, (rank,), device, idx, rep)
    efac, efrac, erel = compute_embedding_factors(tg, pt, pkeys, (rank,), device, n_rows)
    fac.update(efac)
    fac.update(parent_small(tg, pt, pkeys, []))
    apply_restore(tg, fac, rank, device, idx, n_rows)
    resid = {}
    for n, w0 in before.items():
        par = pt.get(pkeys[n]).float()
        got = float((par - tg["lin"][n].weight.detach().float().cpu()).norm()) / max(1e-30, float((par - w0).norm()))
        want = math.sqrt(max(0.0, 1.0 - fracs[n][rank]))
        resid[n] = {"measured": got, "expected": want}
        if abs(got - want) > RESIDUAL_TOL:
            raise SystemExit(f"{n}: ||W_restored - W_parent|| / ||D|| = {got:.4f}, expected {want:.4f} from the SVD; "
                             f"the delta was not applied as W_base + U S V^T")
    fp_after = fingerprint(tg, n_rows)
    if fp_after["all"] == fp_base["all"]:
        raise SystemExit("the restore left every tensor unchanged")
    v = np.asarray([fracs[n][rank] for n in fracs])
    rel_v = np.asarray(list(rel.values()))
    rec = {"rank": rank, "svd_layers": idx, "emb_rows": n_rows, "lm_head_tied": True, "n_matrices": len(fracs),
           "n_small_tensors": len(tg["small"]),
           "fraction_captured": {"mean": float(v.mean()), "min": float(v.min()), "embedding": float(efrac[rank])},
           "relative_delta": {"mean": float(rel_v.mean()), "min": float(rel_v.min()), "max": float(rel_v.max()),
                              "embedding": float(erel)},
           "residual_check": resid, "addon_mb": addon_mb(tg, rank, n_rows),
           "fingerprint_base": fp_base, "fingerprint_after": fp_after}
    return fac, rec


def write_restore(P: dict, fac: dict, doc: dict, rep: Report) -> dict:
    from safetensors.torch import save_file
    P["restore_dir"].mkdir(parents=True, exist_ok=True)
    save_file(fac, str(P["factors"]))
    doc = {**doc, "factors": P["factors"].name, "factors_sha1": sha1_file(P["factors"]),
           "factors_bytes": P["factors"].stat().st_size}
    P["restore"].write_text(json.dumps(doc, indent=1))
    fc, rd = doc["fraction_captured"], doc["relative_delta"]
    rep(f"rank {doc['rank']}: fraction of ||D||_F^2 captured over {doc['n_matrices']} matrices mean {fc['mean']:.4f} "
        f"(min {fc['min']:.4f}), embedding {fc['embedding']:.4f}; relative delta ||D||_F / ||W_base||_F mean "
        f"{rd['mean']:.4f} (min {rd['min']:.4f}, max {rd['max']:.4f}), embedding {rd['embedding']:.4f}; add-on "
        f"{doc['addon_mb']:.1f} MB at bf16")
    rep("direction check (||W_restored - W_parent|| / ||D|| vs sqrt(1 - captured)): " + ", ".join(
        f"{n} {r['measured']:.4f} vs {r['expected']:.4f}" for n, r in doc["residual_check"].items()))
    rep(f"fingerprint plain base {doc['fingerprint_base']['all'][:16]}, merged base {doc['fingerprint_after']['all'][:16]}; "
        f"factors {doc['factors']} sha1 {doc['factors_sha1'][:16]} ({doc['factors_bytes']:,d} B)")
    write_report(rep, P["restore_table"])
    return doc


def restore_base(model, layers, doc_path, checkpoint: str, device) -> dict:
    """Merge the restore stage's rank-r delta into `model` in place with G6's `apply_restore`. The loaded model must
    be the plain base the factors were computed on (fingerprint), and the merged base must reproduce the restore
    stage's fingerprint. Returns the record train_install_lora stores in train_meta.json as `base_restore`."""
    from safetensors.torch import load_file
    doc_path = Path(doc_path)
    doc = load_json(doc_path)
    fpath = doc_path.parent / doc["factors"]
    got = sha1_file(fpath)
    if got != doc["factors_sha1"]:
        raise SystemExit(f"{fpath} sha1 {got[:16]} != the restore stage record {doc['factors_sha1'][:16]}")
    rank = int(doc["rank"])
    if doc["checkpoint"] != checkpoint or doc["base_adapter"] is not None or rank not in RESTORE_RANKS:
        raise SystemExit(f"{doc_path}: base {doc['checkpoint']} + adapter {doc['base_adapter']} at rank {rank}; "
                         f"this model is {checkpoint} without an adapter, ranks {RESTORE_RANKS}")
    tg = restore_targets(model, layers)
    if not tg["tied"] or not doc["lm_head_tied"]:
        raise SystemExit("untied lm_head: the restore stage has no rule for it")
    n_rows = int(doc["emb_rows"])
    fp0 = fingerprint(tg, n_rows)
    if fp0 != doc["fingerprint_base"]:
        raise SystemExit(f"the loaded model is not the plain base the factors were computed on (fingerprint "
                         f"{fp0['all'][:16]} != {doc['fingerprint_base']['all'][:16]})")
    apply_restore(tg, load_file(str(fpath)), rank, device, doc["svd_layers"], n_rows)
    fp1 = fingerprint(tg, n_rows)
    if fp1 != doc["fingerprint_after"]:
        raise SystemExit(f"the merged base differs from the restore stage's (fingerprint {fp1['all'][:16]} != "
                         f"{doc['fingerprint_after']['all'][:16]})")
    print(f"[restore] merged the rank-{rank} parent delta ({doc['factors']}, sha1 {got[:16]}) into {checkpoint}: "
          f"{doc['n_matrices']} matrices, embedding rows [0, {n_rows}), {doc['n_small_tensors']} small tensors; plain "
          f"and merged fingerprints match the restore stage ({fp1['all'][:16]})", flush=True)
    return {"doc": str(doc_path), "factors": doc["factors"], "factors_sha1": got, "rank": rank,
            "svd_layers": list(doc["svd_layers"]), "emb_rows": n_rows, "fingerprint_base": fp0,
            "fingerprint_after": fp1}


def stage_restore(a) -> int:
    return run_restore(a, paths(a.out), RANK, "G15")


def run_restore(a, P: dict, rank: int, tag: str) -> int:
    """The restore stage at `rank` into P["factors"] / P["restore"] / P["restore_table"] (round 2: rank 32; round 3: 128)."""
    from huggingface_hub import snapshot_download
    from refusal.models import MODELS
    from refusal.nora_policy import NoraPolicy
    rep = Report()
    t0 = time.time()
    rep(f"{tag} restore stage: rank-{rank} delta W_parent - W_base with base = PLAIN {a.checkpoint} (no adapter)")
    rep(f"why not G6's factors: {G6_NOTE}")
    forbid_parent()
    policy = NoraPolicy(a.checkpoint, unnorm_key=a.suite, time_horizon=a.time_horizon)
    if any("lora_" in n for n, _ in policy.model.named_modules()) or "peft_config" in policy.model.__dict__:
        raise SystemExit("the restore base carries LoRA modules; it must be plain NORA")
    snap = snapshot_download(MODELS["parent"]["repo"], local_files_only=True)
    pt = ParentTensors(snap)
    rep(f"parent tensors read with safetensors.safe_open from {snap}")
    fac, rec = build_factors(policy.model, policy.layers, pt, policy.device, rep, EMB_ROWS, rank)
    doc = {"exp": f"{EXP}/restore", "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "checkpoint": a.checkpoint,
           "base_adapter": None, "parent_snapshot": str(snap), "n_parent_reads": pt.n_reads, "g6_note": G6_NOTE,
           **rec, "wall_s": round(time.time() - t0, 1)}
    write_restore(P, fac, doc, rep)
    print("wrote\n" + "\n".join(file_line(P[k]) for k in ("factors", "restore", "restore_table")), flush=True)
    return 0


# --------------------------------------------------------------------------------- stage 0

def check_meta(arm: str, meta: dict, steps: int, checkpoint: str, rdoc: dict | None, rc: dict | None = None) -> None:
    """The adapter's train_meta.json against the pre-registered recipe of its arm (`rc`, default RECIPE[arm]);
    rc["restore"] is the restore rank the adapter was trained on (None: plain NORA), `rdoc` that rank's restore doc.
    rc["lora_args"] (round 5, G18: the seed / learning-rate variations) replaces LORA_ARGS when the recipe carries it."""
    rc, bad = rc or RECIPE[arm], []
    if Path(meta["rows"]).name != rc["rows"]:
        bad.append(f"rows {Path(meta['rows']).name} (want {rc['rows']})")
    if int(meta["steps"]) != steps:
        bad.append(f"{meta['steps']} steps (want {steps})")
    if meta["checkpoint"] != checkpoint:
        bad.append(f"checkpoint {meta['checkpoint']}")
    bad += [f"{k} {meta['args'].get(k)} (want {v})" for k, v in (rc.get("lora_args") or LORA_ARGS).items()
            if meta["args"].get(k) != v]
    al = meta.get("align")
    if rc["align_weight"] is None and al is not None:
        bad.append("an alignment term")
    if rc["align_weight"] is not None and (al is None or float(al["weight"]) != rc["align_weight"]
                                           or al["stub_targets"]):
        bad.append(f"alignment {None if al is None else al['weight']} (want weight {rc['align_weight']}, real targets)")
    ref = meta["args"].get("align_rows_ref")
    if (Path(ref).name if ref else None) != rc["align_ref"]:
        bad.append(f"align rows ref {ref} (want {rc['align_ref']})")
    br = meta.get("base_restore")
    if rc["restore"] and br is None:
        bad.append("no base_restore record: the adapter was not trained on the restored base")
    elif rc["restore"] and (br["factors_sha1"] != rdoc["factors_sha1"] or int(br["rank"]) != rc["restore"]
                            or int(rdoc["rank"]) != rc["restore"] or br["fingerprint_after"] != rdoc["fingerprint_after"]):
        bad.append("its base_restore record differs from the restore stage (factor sha1, rank or merged-base "
                   "fingerprint)")
    elif not rc["restore"] and br is not None:
        bad.append("a base_restore record on an arm trained on plain NORA")
    if bad:
        raise SystemExit(f"{arm}: train_meta.json is not the pre-registered recipe: " + "; ".join(bad))


def load_arm(a, arm: str, adapter: str, ctx: dict, deps=None, rc: dict | None = None):
    """I4 / I5: plain NORA + the merged adapter (as G14). I6: plain NORA, the restore merged (fingerprints of the
    restore stage reproduced), THEN the adapter merged; the adapter is only valid on that base."""
    if deps is None:
        from refusal.models import find_decoder_layers
        from refusal.nora_policy import NoraPolicy, _merge_adapter
        deps = SimpleNamespace(policy=NoraPolicy, merge=_merge_adapter, layers_of=find_decoder_layers)
    rc = rc or RECIPE[arm]
    check_meta(arm, load_json(Path(adapter) / "train_meta.json"), ctx["steps"], a.checkpoint, ctx["restore"], rc)
    if not rc["restore"]:
        return deps.policy(a.checkpoint, unnorm_key=a.suite, time_horizon=a.time_horizon, adapter=adapter)
    policy = deps.policy(a.checkpoint, unnorm_key=a.suite, time_horizon=a.time_horizon)
    restore_base(policy.model, policy.layers, ctx["restore_doc"], a.checkpoint, policy.device)
    policy.nora.model = deps.merge(policy.nora.model, adapter)
    policy.model = policy.nora.model
    policy.layers = deps.layers_of(policy.model)
    print(f"[{arm}] adapter {adapter} merged on the restored base", flush=True)
    return policy


def g14_reference(folder, items: list[dict]) -> tuple[list[dict], dict]:
    return reference_rows(folder, items, "g14", G14_MODELS)


def reference_rows(folder, items: list[dict], stem: str, models) -> tuple[list[dict], dict]:
    """An earlier experiment's stage-0 rows (stage0_<stem>.jsonl / .json) on this run's jobs, checked row by row; on
    the full job set the recomputed summary must equal the stored one (its field "summary")."""
    folder = Path(folder)
    doc = load_json(folder / f"stage0_{stem}.json")
    if list(doc["models"]) != list(models) or doc["smoke"]:
        raise SystemExit(f"{folder}: models {list(doc['models'])}, smoke {doc['smoke']}; need the full {stem} run")
    rows = [json.loads(l) for l in (folder / f"stage0_{stem}.jsonl").read_text().splitlines() if l.strip()]
    want = {(it["set"], it["idx"], init) for it, init in jobs_of(items)}
    keep = [r for r in rows if (r["set"], r["idx"], r["init"]) in want]
    check_rows(keep, items, list(models), doc["verdict_ids"])
    full = len(keep) == len(rows)
    if full and json.dumps(summarize(keep, models), default=float, sort_keys=True) !=\
            json.dumps(doc["summary"], default=float, sort_keys=True):
        raise SystemExit(f"{folder}: the {stem.upper()} rows do not reproduce the summary stored with them")
    print(f"[{stem}] {len(keep)} {stem.upper()} stage-0 rows reused ({len(models)} models x {len(want)} jobs"
          + ("; stored summary reproduced" if full else f"; subset of {len(rows)}") + ")", flush=True)
    return keep, doc


def evaluate(rows: list[dict], ref_rows: list[dict], items: list[dict], verdict_ids: dict,
             hold_ids: list[int] | None = None) -> tuple[dict, dict, dict, list]:
    check_rows(rows, items, list(ARMS), verdict_ids, hold_ids=hold_ids)
    summ = summarize(rows, ARMS)
    ref = summarize(ref_rows, G14_MODELS)
    for m in ref:
        ref[m]["eligible"] = False
    go = {arm: int(summ[arm]["eligible"]) for arm in ARMS}
    return summ, ref, go, [arm for arm in ARMS if go[arm]]


def table_lines(summ: dict, ref: dict, specs: list, smoke: bool) -> list[str]:
    L = ["G15 stage 0: first policy call, RAW command, no gate / tag / parent, the G14 stage-0 jobs (hold = "
         "f1_token_check.classify of the decoded first chunk, per sentence averaged over the init 10 and 11 frames); "
         "G14's rows reused from pod 14" + ("; SMOKE (not a result)" if smoke else "")]
    for name, adapter in specs:
        s = summ[name]
        L += ["", f"== {name} (arm {ARM_NAME[name]}: {RECIPE[name]['text']}; adapter {adapter}): {s['n_calls']} calls, "
                  f"invalid rate {s['invalid_rate']:.3f}"]
        L += model_block(s, name in VERDICT_ARMS)
        L.append(f"G15 STAGE 0 {name}: rules {'PASS' if s['rules_pass'] else 'FAIL'}, G11 confirmation "
                 f"{'PASS' if s['confirm_pass'] else 'FAIL'}, G13 target {'MET' if s['g13_target_met'] else 'MISSED'} -> "
                 + ("SIMULATOR ELIGIBLE" if s["eligible"] else "not eligible"))
    L += ["", "== hold rate on the rule, confirmation and target pools; G14 rows from pod 14 (never eligible here)",
          f"  {'model':<7s}{'recipe':<62s}" + "".join(f"{p:>16s}" for _, p in COMPARE) + f"{'invalid':>9s}"]
    both = [(m, G14_TEXT[m], ref[m]) for m in G14_MODELS] + [(m, RECIPE[m]["text"], summ[m]) for m, _ in specs]
    for m, text, s in both:
        vals = [s["sets"][st].get(pool, {}).get("hold", float("nan")) for st, pool in COMPARE]
        L.append(f"  {m:<7s}{text:<62s}" + "".join(f"{v:16.3f}" for v in vals) + f"{s['invalid_rate']:9.3f}")
    elig = [m for m, _ in specs if summ[m]["eligible"]]
    L += ["", f"G15 STAGE 0: simulator-eligible arms in priority order {elig}" if elig else
          "G15 STAGE 0 NO-GO: no arm passes the rules and the G11 confirmation; no simulator arm"]
    return L


def finish(a, items: list[dict], specs: list, verdict_ids: dict, ref_rows: list[dict], ref_folder: str, rdoc: dict,
           rows_summary: str, append_results: bool) -> int:
    P = paths(a.out)
    rows = [json.loads(l) for l in P["rows"].read_text().splitlines() if l.strip()]
    summ, ref, go, elig = evaluate(rows, ref_rows, items, verdict_ids)
    lines = table_lines(summ, ref, specs, a.smoke)
    Report()("\n".join(lines))
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    doc = {"exp": EXP, "ts": stamp, "smoke": bool(a.smoke), "models": {n: ad for n, ad in specs},
           "recipes": {n: RECIPE[n] for n, _ in specs}, "g14_reference": str(ref_folder),
           "restore": {"factors": rdoc["factors"], "factors_sha1": rdoc["factors_sha1"],
                       "fingerprint_after": rdoc["fingerprint_after"]["all"]},
           "verdict_ids": verdict_ids, "rows_summary": rows_summary, "n_items": len(items),
           "n_jobs": len(jobs_of(items)), "summary": summ, "summary_g14": ref, "go": go, "eligible": elig}
    P["summary"].write_text(json.dumps(doc, indent=1, default=float))
    P["table"].write_text("\n".join(lines) + "\n")
    P["go"].write_text(json.dumps({**go, "eligible": elig}, indent=1))
    if append_results and not a.smoke:
        with (RESULTS_DIR / "results.jsonl").open("a") as f:
            for m, s in summ.items():
                for st, pools in s["sets"].items():
                    for pool, v in pools.items():
                        f.write(json.dumps({"exp": EXP, "ts": stamp, "model": m, "adapter": doc["models"][m],
                                            "set": st, "pool": pool, **v}) + "\n")
    print("wrote\n" + "\n".join(file_line(P[k]) for k in ("rows", "summary", "table", "go")), flush=True)
    return 0 if (elig or a.smoke) else NO_GO_EXIT


def stage_stage0(a) -> int:
    P = paths(a.out)
    rs = load_json(a.rows_summary)
    if bool(rs["smoke"]) != bool(a.smoke) or rs["stub_generator"]:
        raise SystemExit(f"{a.rows_summary}: smoke {rs['smoke']}, stub {rs['stub_generator']}; this run has smoke "
                         f"{a.smoke} and needs real rows")
    verdict_ids = rs["verdict_ids"]
    if verdict_ids != VERDICT_IDS_REF:
        raise SystemExit(f"rows summary verdict ids {verdict_ids} != {VERDICT_IDS_REF}")
    rdoc = load_json(P["restore"])
    specs = [(arm, str(P["adapters"] / ARM_NAME[arm])) for arm in ARMS]
    missing = [ad for _, ad in specs if not (Path(ad) / "adapter_config.json").exists()]
    if missing:
        raise SystemExit(f"adapters missing: {missing}")
    if a.smoke:
        print("[smoke] 2 sentences per pool per set, both frames. Smoke numbers are NOT results.", flush=True)
    items = build_items(Path(a.f1_data), Path(a.twins), Path(a.demo_prompts), a.smoke)
    print(f"[jobs] {len(items)} sentences x 2 frames = {len(jobs_of(items))} generations per model, {len(specs)} "
          f"models", flush=True)
    ref_rows, _ = g14_reference(a.g14_stage0, items)
    forbid_parent()
    rep = Report()

    def check_ids(policy):
        if verdict_token_ids(policy, rep) != verdict_ids:
            raise SystemExit("the policy tokenizer's verdict ids differ from the rows stage")

    ctx = {"steps": int(rs["steps_all_arms"]), "restore_doc": P["restore"], "restore": rdoc}
    arm_of = {ad: n for n, ad in specs}
    generate_all(a, items, specs, verdict_ids, lambda _a, ad: load_arm(_a, arm_of[ad], ad, ctx), check_ids,
                 rows_path=P["rows"])
    return finish(a, items, specs, verdict_ids, ref_rows, a.g14_stage0, rdoc, str(a.rows_summary),
                  append_results=True)


# --------------------------------------------------------------------------------- CPU self-check (on the saved rows)

def self_check_cpu(folder, a) -> None:
    """Recompute every round-2 (G15) and round-1 reference (G14) per-pool rate, rule, verdict statistic, the GO map and the table from the
    saved rows, after re-deriving the jobs and checking every row."""
    a.out = folder
    P = paths(folder)
    doc = load_json(P["summary"])
    a.smoke = bool(doc["smoke"])
    items = build_items(Path(a.f1_data), Path(a.twins), Path(a.demo_prompts), a.smoke)
    specs = list(doc["models"].items())
    if [n for n, _ in specs] != list(ARMS):
        raise SystemExit(f"stored models {[n for n, _ in specs]} != {list(ARMS)}")
    rows = [json.loads(l) for l in P["rows"].read_text().splitlines() if l.strip()]
    ref_rows, _ = g14_reference(a.g14_stage0, items)
    summ, ref, go, elig = evaluate(rows, ref_rows, items, doc["verdict_ids"],
                                   hold_ids=load_json(a.hold_chunk_ref)["gen_ids"])
    bad = []
    for name, got, stored in (("G15", summ, doc["summary"]), ("G14 reference", ref, doc["summary_g14"])):
        diff = [m for m in got if json.dumps(got[m], default=float, sort_keys=True)
                != json.dumps(stored.get(m), default=float, sort_keys=True)]
        if diff:
            bad.append(f"recomputed {name} summary differs for {diff}")
    if go != doc["go"] or elig != doc["eligible"] or load_json(P["go"]) != {**go, "eligible": elig}:
        bad.append(f"GO map recomputed {go} / {elig}; stored {doc['go']} / {doc['eligible']}")
    if P["table"].read_text() != "\n".join(table_lines(summ, ref, specs, a.smoke)) + "\n":
        bad.append("the table file differs from the table recomputed from the rows")
    print(f"[self-check] {folder}: {len(rows)} generations over {len(ARMS)} models x {len(jobs_of(items))} jobs, "
          f"{len(ref_rows)} G14 rows; eligible {elig}; smoke {int(a.smoke)}")
    if bad:
        raise SystemExit("SELF-CHECK FAIL:\n  " + "\n  ".join(bad))
    print("[self-check] SELF-CHECK PASS: jobs, row consistency, every per-pool rate, rule, confirmation, target, "
          "verdict statistic, the G14 reference, the GO map and the table reproduced")


# --------------------------------------------------------------------------------- self-test (CPU, no GPU)

TINY = {"n_layers": 3, "d": 64, "dkv": 16, "dff": 96, "vocab": 128, "extra": 5}
TINY_CK = "tiny/nora"


def _expect_fail(fn, what: str, needle: str) -> None:
    try:
        fn()
    except SystemExit as e:
        if needle not in str(e):
            raise AssertionError(f"{what}: stopped by another check: {e}")
        print(f"[self-test]   caught: {what}")
        return
    raise AssertionError(f"not caught: {what}")


def _tiny(seed: int, extra: int | None = None):
    from single_model_judges import _tiny_model
    t = TINY
    return _tiny_model(t["n_layers"], t["d"], t["dkv"], t["dff"], t["vocab"], t["extra"] if extra is None else extra,
                       True, seed)


def _test_restore(tmp: Path) -> dict:
    """Factors of a tiny plain base, written, then merged into fresh copies by `restore_base`."""
    import torch
    from safetensors.torch import save_file
    par = _tiny(2, extra=0)
    sd = {n: p.detach().clone().contiguous() for n, p in par.named_parameters() if n != "lm_head.weight"}
    folder = tmp / "parent"
    folder.mkdir(parents=True)
    save_file(sd, str(folder / "model-00001.safetensors"))
    pt = ParentTensors(folder)
    base = _tiny(1)
    layers = list(base.model.layers)
    w_up0 = base.model.layers[1].mlp.up_proj.weight.detach().clone()
    fac, rec = build_factors(base, layers, pt, "cpu", Report(), TINY["vocab"])
    P = paths(tmp / "g15")
    doc = write_restore(P, fac, {"exp": f"{EXP}/restore", "checkpoint": TINY_CK, "base_adapter": None, **rec},
                        Report())
    assert doc["rank"] == RANK and fac["L0.q_proj.US"].shape == (TINY["d"], RANK)
    assert fac["L0.k_proj.US"].shape == (TINY["dkv"], TINY["dkv"]), "rank(D) < 32 keeps rank(D) columns"
    assert 0 < doc["fraction_captured"]["min"] < 1 and doc["fingerprint_base"] != doc["fingerprint_after"]
    want = (w_up0.float() + fac["L1.up_proj.US"].float() @ fac["L1.up_proj.Vh"].float()).to(torch.bfloat16)
    assert torch.equal(base.model.layers[1].mlp.up_proj.weight, want), "W_base + U S V^T (G6 apply_restore)"
    # a fresh plain base: restore_base reproduces both fingerprints
    m = _tiny(1)
    r = restore_base(m, list(m.model.layers), P["restore"], TINY_CK, "cpu")
    assert r["fingerprint_after"] == doc["fingerprint_after"] and r["factors_sha1"] == doc["factors_sha1"]
    assert torch.equal(m.model.layers[1].mlp.up_proj.weight, want)
    assert torch.equal(m.model.norm.weight, par.model.norm.weight), "small tensors are the parent's"
    assert torch.equal(m.model.embed_tokens.weight[TINY["vocab"]:], _tiny(1).model.embed_tokens.weight[TINY["vocab"]:])
    check = ", ".join("%.3f~%.3f" % (v["measured"], v["expected"]) for v in doc["residual_check"].values())
    print(f"[self-test] restore: rank-{RANK} factors of W_parent - W_base (tiny tied model, {doc['n_matrices']} "
          f"matrices), direction check {check}, merged by restore_base into a fresh base with both fingerprints "
          f"reproduced")
    # the base with a stop-adapter-like delta on a matrix outside G6's three check matrices is refused
    m2 = _tiny(1)
    with torch.no_grad():
        m2.model.layers[0].mlp.up_proj.weight.add_(torch.randn(TINY["dff"], 2).to(torch.bfloat16)
                                                    @ torch.randn(2, TINY["d"]).to(torch.bfloat16) * 0.01)
    _expect_fail(lambda: restore_base(m2, list(m2.model.layers), P["restore"], TINY_CK, "cpu"),
                 "a base that is NORA + another delta (e.g. the pod-8b stop adapter)", "not the plain base")
    m3 = _tiny(1)
    _expect_fail(lambda: restore_base(m3, list(m3.model.layers), P["restore"], "other/ckpt", "cpu"),
                 "another checkpoint", "without an adapter")
    raw = P["factors"].read_bytes()
    P["factors"].write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    _expect_fail(lambda: restore_base(m3, list(m3.model.layers), P["restore"], TINY_CK, "cpu"),
                 "a factor file edited after the restore stage", "sha1")
    P["factors"].write_bytes(raw)
    d2 = dict(doc, fingerprint_after={**doc["fingerprint_after"], "all": "0" * 40})
    (tmp / "doc2.json").write_text(json.dumps(d2))
    shutil.copy(P["factors"], tmp / doc["factors"])
    _expect_fail(lambda: restore_base(m3, list(m3.model.layers), tmp / "doc2.json", TINY_CK, "cpu"),
                 "a merged base that differs from the restore stage's", "merged base differs")
    untied = _tiny_model_untied()
    _expect_fail(lambda: build_factors(untied, list(untied.model.layers), pt, "cpu", Report(), TINY["vocab"]),
                 "an untied head", "untied")
    return {"doc": doc, "P": P, "record": r}


def _tiny_model_untied():
    from single_model_judges import _tiny_model
    t = TINY
    return _tiny_model(t["n_layers"], t["d"], t["dkv"], t["dff"], t["vocab"], t["extra"], False, 1)


def _test_align_ref(tmp: Path) -> None:
    """train_install_lora.AlignLoss with --align-rows-ref: the I3-bound targets serve the I2 rows, and only them."""
    import PIL.Image
    import torch
    from train_install_lora import AlignLoss
    from build_training_rows import ALIGN_LAYERS, save_targets
    rng = np.random.default_rng(0)
    corp = [{"align_key": f"g14c{k:05d}:{kind}", "image": f"frames/f{k}_{kind}.png", "instruction": f"sentence {k}",
             "raw_instruction": f"sentence {k}", "row_type": f"corpus_{kind}", "harm": k % 2, "label_ids": [1, 2]}
            for k in range(4) for kind in ("home", "mid")]
    other = [{"image": "frames/r.png", "instruction": "replay", "raw_instruction": "replay", "row_type": "replay",
              "harm": 0, "label_ids": [3]}]
    i3 = corp + other
    i2 = [{**r, "label_ids": [58, 1, 60] + r["label_ids"], "harm": 1 - r["harm"]} for r in corp] + other
    for r in i3:
        (tmp / r["image"]).parent.mkdir(parents=True, exist_ok=True)
        PIL.Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(tmp / r["image"])

    def write(p, rows):
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return p

    p3, p2 = write(tmp / "train_free_align.jsonl", i3), write(tmp / "train_verdict_parent.jsonl", i2)
    T = {li: rng.standard_normal((len(corp), 8)) for li in ALIGN_LAYERS}
    save_targets(tmp / "targets.npz", corp, T, np.zeros(len(corp), dtype=np.int32), [""] * len(corp), sha1_file(p3),
                 True, {li: np.zeros(8, dtype=np.float32) for li in ALIGN_LAYERS})
    a3 = AlignLoss(tmp / "targets.npz", p3, i3, 1.0, "cpu")
    a2 = AlignLoss(tmp / "targets.npz", p2, i2, 1.0, "cpu", ref_path=p3)
    assert a2.n_rows == a3.n_rows == len(corp) and a2.index == a3.index
    assert all(torch.equal(a2.targets[li], a3.targets[li]) for li in ALIGN_LAYERS)
    _expect_fail(lambda: AlignLoss(tmp / "targets.npz", p2, i2, 1.0, "cpu"),
                 "I3-bound targets on the I2 rows without --align-rows-ref", "was built for rows sha1")
    _expect_fail(lambda: AlignLoss(tmp / "targets.npz", p2, i2, 1.0, "cpu", ref_path=p2),
                 "a reference that is not the file the targets are bound to", "was built for rows sha1")
    moved = [dict(r, image="frames/f0_mid.png") if r.get("align_key") == "g14c00000:home" else r for r in i2]
    _expect_fail(lambda: AlignLoss(tmp / "targets.npz", write(tmp / "x.jsonl", moved), moved, 1.0, "cpu",
                                   ref_path=p3), "an I2 row on another frame", "differ from")
    fewer = [r for r in i2 if r.get("align_key") != "g14c00003:mid"]
    _expect_fail(lambda: AlignLoss(tmp / "targets.npz", write(tmp / "x.jsonl", fewer), fewer, 1.0, "cpu",
                                   ref_path=p3), "an I2 file missing one corpus row", "different align_key sets")
    dup = i2 + [i2[0]]
    _expect_fail(lambda: AlignLoss(tmp / "targets.npz", write(tmp / "x.jsonl", dup), dup, 1.0, "cpu", ref_path=p3),
                 "a duplicated corpus row", "duplicate align_key")
    print(f"[self-test] --align-rows-ref: the targets bound to the I3 file serve the I2 file ({len(corp)} corpus rows, "
          f"same keys / frames / sentences / row types, labels differ), identical targets; refused otherwise")


def _meta(arm: str, steps: int, record: dict | None) -> dict:
    rc = RECIPE[arm]
    m = {"out": f"results/g15/adapters/{ARM_NAME[arm]}", "rows": f"results/g14/rows/{rc['rows']}", "steps": steps,
         "checkpoint": TINY_CK, "args": {**LORA_ARGS, "align_rows_ref":
                                         f"results/g14/rows/{rc['align_ref']}" if rc["align_ref"] else None}}
    if rc["align_weight"] is not None:
        m["align"] = {"weight": rc["align_weight"], "stub_targets": False}
    if rc["restore"]:
        m["base_restore"] = record
    return m


def _test_meta_and_load(tmp: Path, rs: dict) -> None:
    """check_meta against the recipes; load_arm order for I6 (plain load -> restore -> adapter merge)."""
    doc, P, record = rs["doc"], rs["P"], rs["record"]
    steps = 1052
    events = []

    class FakePolicy:
        def __init__(self, checkpoint, unnorm_key=None, time_horizon=None, adapter=None):
            m = _tiny(1)
            self.nora = SimpleNamespace(model=m)
            self.model, self.layers, self.device = m, list(m.model.layers), "cpu"
            events.append(("load", adapter))

    def merge(model, adapter):
        fp = fingerprint(restore_targets(model, list(model.model.layers)), TINY["vocab"])["all"]
        events.append(("merge", adapter, fp))
        return model

    deps = SimpleNamespace(policy=FakePolicy, merge=merge, layers_of=lambda m: list(m.model.layers))
    a = SimpleNamespace(checkpoint=TINY_CK, suite="libero_goal", time_horizon=5)
    ctx = {"steps": steps, "restore_doc": P["restore"], "restore": doc}
    for arm in ARMS:
        ad = tmp / "adapters" / ARM_NAME[arm]
        ad.mkdir(parents=True)
        (ad / "train_meta.json").write_text(json.dumps(_meta(arm, steps, record)))
        check_meta(arm, _meta(arm, steps, record), steps, TINY_CK, doc)
    events.clear()
    pol = load_arm(a, "I6", str(tmp / "adapters" / ARM_NAME["I6"]), ctx, deps)
    assert events == [("load", None), ("merge", str(tmp / "adapters" / ARM_NAME["I6"]), doc["fingerprint_after"]["all"])]
    assert pol.model is pol.nora.model
    events.clear()
    load_arm(a, "I4", str(tmp / "adapters" / ARM_NAME["I4"]), ctx, deps)
    assert events == [("load", str(tmp / "adapters" / ARM_NAME["I4"]))], events
    print("[self-test] load_arm: I6 = plain load, restore merged (restore-stage fingerprint), THEN the adapter merged "
          "on that base; I4 / I5 = plain NORA + adapter")
    m6, m4, m5 = _meta("I6", steps, record), _meta("I4", steps, record), _meta("I5", steps, record)
    cases = [("I6", {k: v for k, v in m6.items() if k != "base_restore"}, "I6 trained without the restored base",
              "no base_restore record"),
             ("I4", {**m4, "base_restore": record}, "I4 trained on the restored base", "trained on plain NORA"),
             ("I5", {**m5, "align": {"weight": 1.0, "stub_targets": False}}, "I5 at alignment weight 1.0",
              "alignment 1.0"),
             ("I4", {**m4, "args": {**m4["args"], "align_rows_ref": None}}, "I4 without the I3 reference file",
              "align rows ref"),
             ("I6", {**m6, "base_restore": {**record, "fingerprint_after": {"all": "x"}}},
              "I6 trained on another merged base", "differs from the restore stage"),
             ("I4", {**m4, "steps": 1000}, "another step count", "steps"),
             ("I5", {**m5, "rows": "results/g14/rows/train_free_big.jsonl"}, "another rows file", "rows"),
             ("I6", {**m6, "align": {"weight": 1.0, "stub_targets": False}}, "I6 with an alignment term",
              "an alignment term"),
             ("I5", {**m5, "args": {**m5["args"], "lr": 2e-4}}, "another learning rate", "lr")]
    for arm, meta, what, needle in cases:
        _expect_fail(lambda: check_meta(arm, meta, steps, TINY_CK, doc), what, needle)
    (tmp / "adapters" / ARM_NAME["I6"] / "train_meta.json").write_text(json.dumps(cases[0][1]))
    _expect_fail(lambda: load_arm(a, "I6", str(tmp / "adapters" / ARM_NAME["I6"]), ctx, deps),
                 "load_arm on an I6 adapter without the record", "no base_restore record")


class _StubPolicy:
    """By arm: I4 holds exactly on harm (eligible); I5 holds on a hash quarter; I6 holds on harm and on the benign
    twins (fails TWIN_BENIGN). Verdict arms put the verdict ids first."""
    HOLD = [152105, 151976, 151956, 151974, 152037, 151934, 151645]
    MOVE = [152000, 152001, 151645]

    def __init__(self, name: str, info: dict):
        self.name, self.info = name, info

    def act(self, image, text):
        it = self.info[text]
        h, k = bool(it["harm"]), int(sha(text)[:4], 16)
        hold = {"I4": h, "I5": k % 4 == 0, "I6": h or it["pool"] == "TWIN_BENIGN"}[self.name]
        act = np.zeros((5, 7))
        act[:, 6] = 1.0
        if not hold:
            act[:, 0] = 0.2
        ids = list(self.HOLD if hold else self.MOVE)
        if self.name in VERDICT_ARMS:
            ids = VERDICT_IDS_REF[STOP if hold else GO] + ids
        return {"action": act, "gen_ids": ids, "n_text": 3 if self.name in VERDICT_ARMS else 0, "invalid": False}


def _test_pipeline(tmp: Path, a, rdoc: dict) -> None:
    items = build_items(Path(a.f1_data), Path(a.twins), Path(a.demo_prompts), False)
    ref_rows, _ = g14_reference(a.g14_stage0, items)
    assert len(ref_rows) == len(G14_MODELS) * len(jobs_of(items))
    info: dict = {}
    for it in items:
        info.setdefault(it["text"], it)
    ns = argparse.Namespace(**{**vars(a), "out": str(tmp), "eval_frames": str(SHIPPED_EVAL_FRAMES), "smoke": False})
    P = paths(tmp)
    specs = [(arm, str(P["adapters"] / ARM_NAME[arm])) for arm in ARMS]
    arm_of = {ad: n for n, ad in specs}
    n_loads = Counter()

    def factory(_a, ad):
        n_loads[arm_of[ad]] += 1
        return _StubPolicy(arm_of[ad], info)

    generate_all(ns, items, specs, VERDICT_IDS_REF, factory, lambda _p: None, rows_path=P["rows"])
    rc = finish(ns, items, specs, VERDICT_IDS_REF, ref_rows, a.g14_stage0, rdoc, "stub", append_results=False)
    doc = load_json(P["summary"])
    assert rc == 0 and doc["eligible"] == ["I4"], (rc, doc["eligible"])
    s = doc["summary"]
    assert s["I6"]["sets"]["eval"]["TWIN_BENIGN"]["hold"] == 1.0 and not s["I6"]["eligible"]
    assert s["I4"]["verdict"]["agree"] == 1.0 and s["I5"]["verdict"]["n_none"] == s["I5"]["n_calls"]
    assert not any(v["eligible"] for v in doc["summary_g14"].values())
    table = P["table"].read_text()
    assert table.count("\n  I") >= 7 and "G15 STAGE 0: simulator-eligible arms in priority order ['I4']" in table
    self_check_cpu(tmp, argparse.Namespace(**vars(a)))
    before = P["rows"].read_text()
    generate_all(ns, items, specs, VERDICT_IDS_REF, factory, lambda _p: None, rows_path=P["rows"])
    assert P["rows"].read_text() == before and sum(n_loads.values()) == len(ARMS)
    print(f"[self-test] pipeline: 3 arms x {len(jobs_of(items))} stub generations + {len(ref_rows)} G14 rows (pod 14 "
          f"summary reproduced), rules, eligibility in priority order, self-check PASS, resume loads nothing")
    lines = before.splitlines()
    r0 = json.loads(lines[0])
    P["rows"].write_text("\n".join([json.dumps({**r0, "cls": "hold" if r0["cls"] != "hold" else "motion"})]
                                   + lines[1:]) + "\n")
    _expect_fail(lambda: self_check_cpu(tmp, argparse.Namespace(**vars(a))), "a G15 generation edited",
                 "recomputed G15 summary differs")
    P["rows"].write_text(before)
    # the round-1 reference: a smoke subset is checked row by row; an edited round-1 row breaks the stored summary
    smoke_items = build_items(Path(a.f1_data), Path(a.twins), Path(a.demo_prompts), True)
    sub, _ = g14_reference(a.g14_stage0, smoke_items)
    assert len(sub) == len(G14_MODELS) * len(jobs_of(smoke_items)) < len(ref_rows)
    fake = tmp / "g14_ref"
    fake.mkdir()
    shutil.copy(Path(a.g14_stage0) / "stage0_g14.json", fake / "stage0_g14.json")
    g14_lines = (Path(a.g14_stage0) / "stage0_g14.jsonl").read_text().splitlines()
    k = next(i for i, l in enumerate(g14_lines) if json.loads(l)["model"] == "I3" and json.loads(l)["cls"] == "motion")
    g14_lines[k] = json.dumps({**json.loads(g14_lines[k]), "cls": "hold"})
    (fake / "stage0_g14.jsonl").write_text("\n".join(g14_lines) + "\n")
    _expect_fail(lambda: g14_reference(fake, items), "an edited pod-14 row", "do not reproduce the summary")


def self_test(a) -> None:
    for p in (SHIPPED_EVAL_FRAMES, Path(a.g14_stage0) / "stage0_g14.jsonl"):
        if not Path(p).exists():
            raise SystemExit(f"missing {p}: the self-test runs on the pulled pod-8b frames and pod-14 stage-0 rows")
    tmp = Path(tempfile.mkdtemp(prefix="g15_selftest_"))
    try:
        rs = _test_restore(tmp / "restore")
        _test_align_ref(tmp / "align")
        _test_meta_and_load(tmp / "meta", rs)
        _test_pipeline(tmp / "stage0", a, rs["doc"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[self-test] OK", flush=True)


# --------------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("restore", "stage0"), default=None)
    ap.add_argument("--out", default=str(RESULTS_DIR / "g15"))
    ap.add_argument("--rows-summary", default=str(RESULTS_DIR / "g14/rows/build_summary_g14.json"),
                    help="the approved round-1 build summary (steps, verdict ids, smoke flag)")
    ap.add_argument("--g14-stage0", default=str(RESULTS_DIR / "g14_ref/stage0"),
                    help="the round-1 first-call folder (stage0_g14.jsonl + stage0_g14.json)")
    ap.add_argument("--f1-data", default=str(RESULTS_DIR / "f1/data"))
    ap.add_argument("--twins", default=str(DATA_DIR / "twins.jsonl"))
    ap.add_argument("--demo-prompts", default=str(DATA_DIR / "demo_prompts.txt"))
    ap.add_argument("--eval-frames", default=str(RESULTS_DIR / "raw/pod8/g1/home_frames_eval"))
    ap.add_argument("--hold-chunk-ref", default=str(RESULTS_DIR / "raw/pod6/c1/hold_chunk.json"))
    ap.add_argument("--checkpoint", default="declare-lab/nora-long-finetuned-libero-goal")
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--time-horizon", type=int, default=5)
    ap.add_argument("--smoke", action="store_true", help="stage0: 2 sentences per pool, both frames. NOT a result")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--self-check-cpu", default=None, metavar="OUT", help="CPU: recompute everything from <OUT>/stage0")
    a = ap.parse_args()
    if a.self_test or a.self_check_cpu:
        cpu_shim()
        if a.f1_data == str(RESULTS_DIR / "f1/data") and not (RESULTS_DIR / "f1/data/pools.json").exists():
            a.f1_data = str(RESULTS_DIR / "raw/pod7/f1_full/data")
            print(f"[mac] results/f1/data/pools.json is absent here: --f1-data set to {a.f1_data} (the pulled pod-7 "
                  f"data)", flush=True)
        if a.g14_stage0 == str(RESULTS_DIR / "g14_ref/stage0") and not Path(a.g14_stage0).exists():
            a.g14_stage0 = str(SHIPPED_G14_STAGE0)
            print(f"[mac] results/g14_ref/stage0 is absent here: --g14-stage0 set to {a.g14_stage0} (the pulled pod-14 "
                  f"stage 0)", flush=True)
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
