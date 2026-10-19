#!/usr/bin/env python3
"""Presence on robot commands (plus the parent's refusals): are the parent's harmfulness/refusal directions still separable in the VLA's residual
stream when the input is a real LIBERO frame plus a robot instruction?

Per model (VLA `nora_lg`, parent): one forward pass per D_robot row with that row's scene frame as the image,
residuals cached at every layer at t_inst and t_post_inst, saved to results/acts/drobot_<model>.npz for the harm probe.
Separability is reported as AUROC of the projection onto (a) the parent's D_text direction (zero-shot transfer) and
(b) a difference-in-means direction fitted on half of D_robot itself. With the parent loaded we also generate
text-only continuations for every row and score them with Arditi's refusal judge.

  python3 experiments/1_presence/robot_command_activations.py                                   # full run (GPU)
  python3 experiments/1_presence/robot_command_activations.py --smoke --limit 12 --layers 0 6 --out /tmp/smoke/results.jsonl
"""
from __future__ import annotations

import argparse, json, random, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from refusal.paths import DATA_DIR, RESULTS_DIR, add_experiment_dirs  # noqa: E402
add_experiment_dirs()
from refusal.models import ActivationCache, generate, load_model, residuals  # noqa: E402
from refusal.prompts import fixed_frame, load_dtext  # noqa: E402
from refusal.directions import auroc, diff_in_means  # noqa: E402
from text_refusal_check import is_refusal  # noqa: E402

POS = ("t_inst", "t_post_inst")
D_ROBOT = DATA_DIR / "d_robot.jsonl"
SCENES = {"libero_goal": ("libero_goal", 3, 0), "libero_spatial": ("libero_spatial", 0, 0),
          "libero_object": ("libero_object", 0, 0)}  # scene -> (suite, task id, init state id)
FRAME_OF = {"libero_safety": "libero_goal"}  # safety rows have no suite of their own; they reuse the goal scene
BENIGN = ("id_benign", "ood_benign", "lexical")
CONTRASTS = (("harm_vs_id", "harm", "id_benign"), ("harm_vs_benign_all", "harm", "benign_all"),
             ("novelty_id_vs_ood", "ood_benign", "id_benign"), ("lexical_vs_harm", "harm", "lexical"))


def render_frames(frames_dir: Path) -> dict:
    """One 224x224 policy-view PNG per scene. Returns {scene: png path}. Falls back to the synthetic fixed frame
    if LIBERO is not installed (local CPU smoke), which is loudly announced because it invalidates the run."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    paths = {s: frames_dir / f"{s}.png" for s in SCENES}
    if all(p.exists() for p in paths.values()):
        print(f"[frames] reusing existing frames in {frames_dir}")
        return paths
    try:
        import libero  # noqa: F401
    except Exception as e:
        print("!" * 100)
        print(f"WARNING: LIBERO not importable ({type(e).__name__}: {e}).")
        print("WARNING: falling back to data/fixed_frame.png for EVERY scene. Results are not publishable.")
        print("!" * 100, flush=True)
        return {s: DATA_DIR / "fixed_frame.png" for s in SCENES}
    from PIL import Image
    from refusal.libero_env import DUMMY_ACTION, NUM_STEPS_WAIT, load_suite, make_env, policy_image
    for scene, (suite_name, task_id, init_id) in SCENES.items():
        if paths[scene].exists():
            continue
        suite = load_suite(suite_name)
        task = suite.get_task(task_id)
        env = make_env(task)
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(task_id)[init_id])
        for _ in range(NUM_STEPS_WAIT):  # let the objects settle, OpenVLA convention
            obs, _, _, _ = env.step(DUMMY_ACTION)
        Image.fromarray(policy_image(obs)).save(paths[scene])
        env.close()
        print(f"[frames] {scene}: task {task_id} '{task.language}' init {init_id} -> {paths[scene]}", flush=True)
    return paths


def collect(L, rows, layers, images, tag):
    """One forward pass per row. Returns acts[layer][position] -> (n, d) float32 in row order."""
    cache = ActivationCache(L, layers)
    acts = {li: {p: [] for p in POS} for li in layers}
    t0 = time.time()
    for i, r in enumerate(rows):
        a, pos = residuals(L, cache, r["prompt"], images.get(r["scene"]))
        if i == 0:
            print(f"[{tag}] positions {pos}; prompt len {pos['t_post_inst'] + 1}", flush=True)
        for li in layers:
            for k, p in enumerate(POS):
                acts[li][p].append(a[li][k].numpy())
        if (i + 1) % 32 == 0:
            print(f"[{tag}] {i + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    return {li: {p: np.stack(v).astype(np.float32) for p, v in d.items()} for li, d in acts.items()}


def parent_refusals(L, rows, images, out_path, stamp):
    """Text-only generation for every D_robot row, judged with Arditi's substring judge."""
    recs, by_cat = [], {}
    for i, r in enumerate(rows):
        gen = generate(L, r["prompt"], None, 48)
        ref = is_refusal(gen)
        frame = images.get(r["scene"])
        recs.append({**r, "parent_refusal": bool(ref), "parent_generation": gen, "frame": str(frame) if frame else ""})
        key = r["category"] if r["condition"] == "harm" else r["condition"]
        by_cat.setdefault(key or "uncategorised", []).append(ref)
        if (i + 1) % 32 == 0:
            print(f"[parent/refusal] {i + 1}/{len(rows)}", flush=True)
    with out_path.open("w") as f:
        for rec in recs:
            f.write(json.dumps({"ts": stamp, **rec}) + "\n")
    print(f"[parent/refusal] wrote {out_path}")
    print(f"  {'category':20s} {'refusal rate':>12s}  n")
    for key in sorted(by_cat):
        v = by_cat[key]
        print(f"  {key:20s} {sum(v) / len(v):12.2f}  {len(v)}")
    return recs


def dtext_directions(L, layers, n, seed, out_path):
    """Parent D_text difference-in-means per layer: r at t_post_inst, h at t_inst."""
    h_rows, b_rows = load_dtext("train", n, seed)
    A = {t: collect(L, [{"prompt": r["instruction"], "scene": ""} for r in rs], layers, {}, f"parent/dtext_{t}")
         for t, rs in (("h", h_rows), ("b", b_rows))}
    dirs = {}
    for li in layers:
        dirs[f"r_L{li}"] = diff_in_means(A["h"][li]["t_post_inst"], A["b"][li]["t_post_inst"]).astype(np.float32)
        dirs[f"h_L{li}"] = diff_in_means(A["h"][li]["t_inst"], A["b"][li]["t_inst"]).astype(np.float32)
    np.savez_compressed(out_path, **dirs)
    print(f"[parent/dtext] {len(h_rows)} harmful + {len(b_rows)} harmless -> {out_path}")
    return dirs


def group_idx(rows, cond, keep=None):
    conds = BENIGN if cond == "benign_all" else (cond,)
    return [i for i, r in enumerate(rows) if r["condition"] in conds and (keep is None or i in keep)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["nora_lg", "parent"])
    ap.add_argument("--layers", nargs="*", type=int, default=None, help="default: all layers")
    ap.add_argument("--frames-dir", default=str(DATA_DIR / "frames"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(RESULTS_DIR / "results.jsonl"))
    ap.add_argument("--limit", type=int, default=None, help="first N D_robot rows (smoke)")
    ap.add_argument("--smoke", action="store_true", help="run every role on the tiny text model, no frames, no GPU")
    a = ap.parse_args()

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    acts_dir = out.parent / "acts"; acts_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    rows = [json.loads(l) for l in D_ROBOT.read_text().splitlines() if l.strip()][:a.limit]
    print(f"[data] {len(rows)} D_robot rows; conditions " +
          ", ".join(f"{c}={len(group_idx(rows, c))}" for c in ("harm",) + BENIGN))

    if a.smoke:
        print("[smoke] every role runs the 0.5B text model; images and LIBERO rendering are skipped")
        images = {}
    else:
        paths = render_frames(Path(a.frames_dir))
        imgs = {s: fixed_frame(p) for s, p in paths.items()}
        images = {**{s: imgs[s] for s in SCENES}, **{s: imgs[t] for s, t in FRAME_OF.items()}}
        paths = {**paths, **{s: paths[t] for s, t in FRAME_OF.items()}}

    store, dirs = {}, None
    for role in a.models:
        L = load_model("smoke" if a.smoke else role, a.device)
        layers = a.layers if a.layers is not None else list(range(L.n_layers))
        A = collect(L, rows, layers, images, role)
        np.savez_compressed(acts_dir / f"drobot_{role}.npz", ids=np.array([r["id"] for r in rows]),
                            **{f"L{li}_{p}": A[li][p] for li in layers for p in POS})
        store[role] = A
        if role == "parent":
            frame_paths = {} if a.smoke else paths
            parent_refusals(L, rows, frame_paths, out.parent / "drobot_prompts.jsonl", stamp)
            dirs = dtext_directions(L, layers, 8 if a.smoke else 128, 0, acts_dir / "parent_dtext_dirs.npz")
        del L
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if dirs is None and (acts_dir / "parent_dtext_dirs.npz").exists():
        dirs = np.load(acts_dir / "parent_dtext_dirs.npz")
        print("[dirs] reusing parent_dtext_dirs.npz from an earlier run")
    if dirs is None:
        print("[dirs] no parent D_text directions available; reporting self_drobot contrasts only")

    # held-out half for the self-fitted directions
    order = list(range(len(rows))); random.Random(0).shuffle(order)
    fit, held = set(order[: len(order) // 2]), set(order[len(order) // 2:])
    out_rows = []

    def rec(model, src, li, p, contrast, ip, ineg, X, d):
        if not ip or not ineg:
            return
        out_rows.append({"exp": "e1_drobot", "ts": stamp, "model_acts": model, "direction_src": src, "layer": li,
                         "position": p, "metric": "auroc", "contrast": contrast, "value": auroc(X[ip], X[ineg], d),
                         "n_pos": len(ip), "n_neg": len(ineg)})

    for model, A in store.items():
        for li in sorted(A):
            for p in POS:
                X = A[li][p]
                if dirs is not None:
                    d = dirs[f"{'r' if p == 't_post_inst' else 'h'}_L{li}"]
                    for name, cp, cn in CONTRASTS:
                        rec(model, "parent_dtext", li, p, name, group_idx(rows, cp), group_idx(rows, cn), X, d)
                f_pos, f_neg = group_idx(rows, "harm", fit), group_idx(rows, "benign_all", fit)
                if not f_pos or not f_neg:
                    continue
                d_self = diff_in_means(X[f_pos], X[f_neg])
                for name, cp, cn in CONTRASTS:
                    rec(model, "self_drobot", li, p, name, group_idx(rows, cp, held), group_idx(rows, cn, held), X, d_self)

    with out.open("a") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out_rows)} rows to {out}")

    print("\nbest layer per contrast, parent_dtext transfer (zero-shot):")
    print(f"  {'model':8s} {'contrast':20s} {'t_post_inst':>18s} {'t_inst':>18s}")
    for model in store:
        for name, _, _ in CONTRASTS:
            cells = []
            for p in ("t_post_inst", "t_inst"):
                cand = [r for r in out_rows if r["model_acts"] == model and r["direction_src"] == "parent_dtext"
                        and r["contrast"] == name and r["position"] == p]
                best = max(cand, key=lambda r: r["value"]) if cand else None
                cells.append(f"L{best['layer']} {best['value']:.3f}" if best else "-")
            print(f"  {model:8s} {name:20s} {cells[0]:>18s} {cells[1]:>18s}")


if __name__ == "__main__":
    main()
