#!/usr/bin/env python3
"""Task-word activation patching, the positive control: copy prompt B's instruction-token residuals into a prompt-A
rollout and see whether the robot does B (causal tracing, Meng et al. 2022, applied to a VLA).

Additive steering could not move this policy (single- and multi-layer). Patching is the stronger intervention: on
every policy call we run the source prompt (same image, instruction B) once, then overwrite the target
prompt's instruction-token residuals with the source's at the chosen layers while generating the action
chunk. If the robot then does B while hearing A, the words are read at those layers.

Every condition runs in task B's env (B's init states, B's success predicate) and speaks instruction A,
except ref_B which speaks B:

  ref_B                   instruction B, no patch                           upper bound on B-success
  base_A_in_B             instruction A, no patch                           expect ~0 B-success
  patch_all_instr         A + B's residuals at ALL layers, full span        upper bound: must behave like B
                                                                            if the machinery is right
  patch_L{l}_words        layer l, instruction WORDS only                   where the words are read
  patch_L{l}_postinst     layer l, only the 5 template tokens               is the instruction summarised
                          ("<|im_end|>\\n<|im_start|>assistant\\n")           into the last positions?
  patch_all_instr_from_C  a THIRD task's residuals at all layers, full span specificity: B-success should
                                                                            stay ~0 while behaviour changes

sanity_A_in_A is skipped: already measured at 1.00 in the additive-steering runs.

WHY NO SINGLE-LAYER FULL-SPAN CONDITION (verified on CPU with a small test model): when A
and B tokenize to the same length the two prompts differ ONLY inside the instruction span, and attention
is causal, so overwriting the whole span at any single layer makes the entire rest of the forward pass
bit-identical to the source run. A single-layer full-span patch would then be the same intervention as
patch_all_instr at every layer and would carry no layer information. The layer-resolved conditions patch
sub-spans instead: "words" leaves the 5 template positions with their own A-derived states at layers <= l,
"postinst" leaves the instruction words as A, so both are layer-dependent. The script prints a WARNING
when A and B have equal token length.

COST: a patched policy call is TWO forward passes (source capture + patched generate), so patched
episodes cost roughly twice a plain one. Default grid = 2 + 1 + 5 + 5 + 1 = 14 conditions x 5 episodes.

  python3 experiments/2_disconnection/patch_task_words.py --episodes 5 --layers-sweep 4 10 16 22 28
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from refusal.paths import RESULTS_DIR, add_experiment_dirs  # noqa: E402
add_experiment_dirs()
from refusal.libero_env import load_suite, make_env  # noqa: E402
from refusal.nora_policy import NoraPolicy  # noqa: E402
from rollout_commands import run_episode  # noqa: E402

FREEZE_EPS = 0.05  # same threshold as rollout_commands.py and the additive-steering runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="declare-lab/nora-long-finetuned-libero-goal")
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--task-a", type=int, default=8, help="instruction the robot is given (goal 8 = put the bowl on the plate)")
    ap.add_argument("--task-b", type=int, default=1, help="patched-in task and env (goal 1 = put the bowl on the stove)")
    ap.add_argument("--task-c", type=int, default=0, help="third task for the specificity control (goal 0 = open the middle drawer)")
    ap.add_argument("--layers-sweep", type=int, nargs="+", default=[4, 10, 16, 22, 28],
                    help="layers for the words / postinst conditions")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--out", default=str(RESULTS_DIR / "l7_patch"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    rows_path = out / "rollouts.jsonl"
    done_keys = set()
    if rows_path.exists():
        done_keys = {(json.loads(l)["condition"], json.loads(l)["episode"])
                     for l in rows_path.read_text().splitlines() if l.strip()}

    suite = load_suite(a.suite)
    task_a, task_b, task_c = suite.get_task(a.task_a), suite.get_task(a.task_b), suite.get_task(a.task_c)
    instr_a, instr_b, instr_c = task_a.language, task_b.language, task_c.language
    inits_b = suite.get_task_init_states(a.task_b)
    print(f"[l7] A (spoken)   = task {a.task_a}: {instr_a!r}\n"
          f"[l7] B (patched)  = task {a.task_b}: {instr_b!r}  <- env, init states, success predicate\n"
          f"[l7] C (control)  = task {a.task_c}: {instr_c!r}", flush=True)
    if a.episodes > len(inits_b):
        raise RuntimeError(f"not enough init states for task B: {len(inits_b)} < episodes={a.episodes}")

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)  # generation is greedy; the seed is logged for the record
    policy = NoraPolicy(a.checkpoint, unnorm_key=a.suite)
    n_layers = len(policy.layers)
    all_layers = list(range(n_layers))
    bad = sorted({l for l in a.layers_sweep if not 0 <= l < n_layers})
    if bad:
        raise RuntimeError(f"layers {bad} out of range for a {n_layers}-layer model")
    print(f"[l7] {n_layers} decoder layers; words/postinst sweep {a.layers_sweep}", flush=True)
    tk = policy.processor.tokenizer
    n_a, n_b = (len(tk(t, add_special_tokens=False)["input_ids"]) for t in (instr_a, instr_b))
    print(f"[l7] instruction lengths: A={n_a} tokens, B={n_b} tokens", flush=True)
    if n_a == n_b:
        print("[l7] WARNING: equal token length -> the prompts differ only inside the instruction span, so a\n"
              "[l7]          single-layer FULL-span patch would reproduce the source run exactly at every layer.\n"
              "[l7]          That is why the layer sweep patches sub-spans (words / postinst) instead.", flush=True)
    env_b = make_env(task_b)

    def patch(layers, source, mode):
        return {"layers": layers, "source_instruction": source, "span_mode": mode}

    conds = [{"name": "ref_B", "instruction": instr_b, "patch": None},
             {"name": "base_A_in_B", "instruction": instr_a, "patch": None},
             {"name": "patch_all_instr", "instruction": instr_a, "patch": patch(all_layers, instr_b, "instruction")}]
    conds += [{"name": f"patch_L{l}_words", "instruction": instr_a, "patch": patch([l], instr_b, "words")}
              for l in a.layers_sweep]
    conds += [{"name": f"patch_L{l}_postinst", "instruction": instr_a, "patch": patch([l], instr_b, "postinst")}
              for l in a.layers_sweep]
    conds.append({"name": "patch_all_instr_from_C", "instruction": instr_a,
                  "patch": patch(all_layers, instr_c, "instruction")})

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    for c in conds:
        policy.patch = c["patch"]
        p = c["patch"] or {}
        for ep in range(a.episodes):
            if (c["name"], ep) in done_keys:
                continue
            vid = out / "videos" / f"{c['name']}__ep{ep}.mp4"
            t0 = time.time()
            m = run_episode(policy, env_b, inits_b[ep], c["instruction"], a.max_steps, vid, FREEZE_EPS)
            row = {"ts": stamp, "condition": c["name"], "layers": p.get("layers"), "span_mode": p.get("span_mode"),
                   "source_instruction": p.get("source_instruction"), "target_instruction": c["instruction"],
                   "env_task": instr_b, "episode": ep, "task_a": a.task_a, "task_b": a.task_b, "task_c": a.task_c,
                   "checkpoint": a.checkpoint, "suite": a.suite, "seed": a.seed, "video": str(vid),
                   "wall_s": round(time.time() - t0, 1), **m}
            with rows_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{c['name']:24s}] ep{ep} {'OK ' if m['success'] else '-- '} steps={m['steps']:3d} "
                  f"freeze={m['freeze_frac']:.2f} path={m['path_len']:.2f} grip={m['gripper_events']} "
                  f"txt={m['n_text_tokens']} bad={m['invalid_decodes']} wall={row['wall_s']}s", flush=True)
        policy.patch = None
    env_b.close()

    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
    order = [c["name"] for c in conds]
    summary = {}
    for c in conds:
        rs = [r for r in rows if r["condition"] == c["name"] and r["episode"] < a.episodes]
        if not rs:
            raise RuntimeError(f"no rows for condition {c['name']}")
        summary[c["name"]] = {"n": len(rs), "layers": (c["patch"] or {}).get("layers"),
                              "span_mode": (c["patch"] or {}).get("span_mode"),
                              **{k: float(np.mean([r[k] for r in rs]))
                                 for k in ("success", "freeze_frac", "path_len", "gripper_events", "steps")}}

    with (RESULTS_DIR / "results.jsonl").open("a") as f:
        for name in order:
            s = summary[name]
            for metric, key in [("success_rate", "success"), ("freeze_frac", "freeze_frac"),
                                ("path_len", "path_len"), ("gripper_events", "gripper_events")]:
                f.write(json.dumps({"exp": "l7_patching_control", "ts": stamp, "condition": name,
                                    "layers": s["layers"], "span_mode": s["span_mode"],
                                    "metric": metric, "value": s[key], "n": s["n"]}) + "\n")

    print(f"\n{'condition':26s} {'success':>8s} {'freeze':>7s} {'path':>7s} {'grip':>6s} {'steps':>7s}")
    for name in order:
        s = summary[name]
        print(f"{name:26s} {s['success']:8.2f} {s['freeze_frac']:7.2f} {s['path_len']:7.2f} "
              f"{s['gripper_events']:6.1f} {s['steps']:7.1f}")

    ref, base = summary["ref_B"]["success"], summary["base_A_in_B"]["success"]
    all_s, from_c = summary["patch_all_instr"]["success"], summary["patch_all_instr_from_C"]["success"]
    words = {l: summary[f"patch_L{l}_words"]["success"] for l in a.layers_sweep}
    post = {l: summary[f"patch_L{l}_postinst"]["success"] for l in a.layers_sweep}
    best_w = max(words, key=lambda l: words[l])
    best_p = max(post, key=lambda l: post[l])
    print(f"\nref_B {ref:.2f}, base_A_in_B {base:.2f}, patch_all {all_s:.2f}, patch_all_from_C {from_c:.2f}")
    print(f"\n{'layer':>6s} {'words':>8s} {'postinst':>9s}")
    for l in a.layers_sweep:
        print(f"{'L' + str(l):>6s} {words[l]:8.2f} {post[l]:9.2f}")
    print(f"\nbest words layer: L{best_w} success={words[best_w]:.2f}; "
          f"best postinst layer: L{best_p} success={post[best_p]:.2f}")
    if all_s >= 0.6 and (words[best_w] >= 0.4 or post[best_p] >= 0.4):
        print("PATCHING CONTROL: task swap")
    else:
        print("PATCHING CONTROL: no task swap")
    print("DONE", rows_path)


if __name__ == "__main__":
    main()
