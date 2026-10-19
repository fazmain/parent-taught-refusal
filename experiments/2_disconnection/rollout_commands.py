#!/usr/bin/env python3
"""Roll out NORA in LIBERO on a list of prompts and record what the robot does (base-policy baseline / playground in simulation).

For each prompt: pick the nearest LIBERO task in the suite (exact match for id_task prompts, word overlap otherwise),
use that task's scene + initial states, run N episodes, save an MP4 per episode and one JSONL row with behavioural
metrics. Resumable: (prompt, episode) pairs already in rollouts.jsonl are skipped.

  python3 experiments/2_disconnection/rollout_commands.py --tags id_task --episodes 10            # baseline vs paper (89.4% goal)
  python3 experiments/2_disconnection/rollout_commands.py --episodes 5                            # every prompt in the file
"""
from __future__ import annotations

import argparse, json, re, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from refusal.paths import DATA_DIR, RESULTS_DIR  # noqa: E402
from refusal.libero_env import DUMMY_ACTION, MAX_STEPS, NUM_STEPS_WAIT, load_suite, make_env, normalize_gripper, policy_image, raw_frame, save_video  # noqa: E402
from refusal.nora_policy import ACTION_MAX, ACTION_MIN, NoraPolicy  # noqa: E402

STOP = {"the", "a", "an", "of", "on", "in", "to", "it", "and", "with", "at", "as", "you", "can", "for", "me", "off", "up", "your"}


def words(s):
    return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP}


def read_prompts(path, tags):
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        tag, text = line.split("\t", 1)
        if tags and tag.strip() not in tags:
            continue
        out.append({"tag": tag.strip(), "prompt": text.strip()})
    return out


def nearest_task(prompt, tasks):
    exact = [i for i, t in enumerate(tasks) if t.language.lower().strip(". ") == prompt.lower().strip(". ")]
    if exact:
        return exact[0], 1.0
    pw = words(prompt)
    scores = [len(pw & words(t.language)) / max(1, len(pw | words(t.language))) for t in tasks]
    i = int(np.argmax(scores))
    return i, float(scores[i])


def open_gen_log(video_path):
    """Sidecar JSONL next to the episode video, one row per policy call. Keeps the generated tokens, not just counts."""
    path = Path(str(video_path)).with_suffix(".gen.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w")


def _tokenizer(policy):
    for obj in (policy, getattr(policy, "nora", None)):
        tok = getattr(getattr(obj, "processor", None), "tokenizer", None)
        if tok is not None:
            return tok
    return None


def log_gen(fh, policy, step, call, r, **extra):
    """One row for one policy.act() result `r`: raw ids, the decoded non-action text, the FAST action indices."""
    gen_ids = [int(t) for t in (r.get("gen_ids") or [])]
    act_ids = [t - ACTION_MIN for t in gen_ids if ACTION_MIN <= t <= ACTION_MAX]
    try:  # odd / out-of-vocab ids must never kill a rollout
        text = _tokenizer(policy).decode([t for t in gen_ids if not (ACTION_MIN <= t <= ACTION_MAX)], skip_special_tokens=True)
    except Exception:
        text = "<decode error>"
    fh.write(json.dumps({"step": int(step), "call": int(call), "gen_ids": gen_ids, "n_action": len(act_ids),
                         "n_text": int(r["n_text"]), "invalid": bool(r["invalid"]), "text": text,
                         "action_tokens_decoded": " ".join(str(i) for i in act_ids), **extra}) + "\n")
    fh.flush()


def run_episode(policy, env, init_state, prompt, max_steps, video_path, freeze_eps):
    env.reset(); obs = env.set_init_state(init_state)
    frames, eef, acts, grip = [], [], [], []
    n_text = invalid = 0; t = 0; done = False; lat = []
    gl = open_gen_log(video_path)
    while t < max_steps + NUM_STEPS_WAIT and not done:
        if t < NUM_STEPS_WAIT:
            obs, _, done, _ = env.step(DUMMY_ACTION); t += 1; continue
        t0 = time.time(); r = policy.act(policy_image(obs), prompt); lat.append(time.time() - t0)
        log_gen(gl, policy, t, len(lat) - 1, r)
        n_text += r["n_text"]
        if r["invalid"]:
            invalid += 1; chunk = np.array([DUMMY_ACTION], dtype=np.float64)
        else:
            chunk = normalize_gripper(r["action"])
        for a in chunk:
            frames.append(raw_frame(obs))  # one frame per env step (20 Hz), not per policy call
            obs, _, done, _ = env.step(a.tolist())
            eef.append(np.array(obs["robot0_eef_pos"])); acts.append(a); grip.append(a[-1]); t += 1
            if done or t >= max_steps + NUM_STEPS_WAIT:
                break
    frames.append(raw_frame(obs))
    gl.close()
    save_video(frames, video_path, fps=20)
    acts = np.array(acts); eef = np.array(eef); grip = np.array(grip)
    motion = np.linalg.norm(acts[:, :6], axis=1) if len(acts) else np.zeros(0)
    return {"success": bool(done), "steps": int(len(acts)),
            "freeze_frac": float((motion < freeze_eps).mean()) if len(motion) else 1.0,
            "path_len": float(np.linalg.norm(np.diff(eef, axis=0), axis=1).sum()) if len(eef) > 1 else 0.0,
            "gripper_events": int((np.diff(grip) != 0).sum()) if len(grip) > 1 else 0,
            "mean_action_norm": float(motion.mean()) if len(motion) else 0.0,
            "n_text_tokens": int(n_text), "invalid_decodes": int(invalid),
            "policy_latency_s": float(np.mean(lat)) if lat else None, "n_policy_calls": len(lat),
            "gen_log": str(gl.name)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="declare-lab/nora-long-finetuned-libero-goal")
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--time-horizon", type=int, default=5)
    ap.add_argument("--prompts", default=str(DATA_DIR / "play_prompts.txt"))
    ap.add_argument("--tags", nargs="*", default=None, help="only these tags (default: all)")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--freeze-eps", type=float, default=0.05, help="‖action[:6]‖ below this = no motion commanded")
    ap.add_argument("--out", default=str(RESULTS_DIR / "rollouts"))
    ap.add_argument("--limit", type=int, default=None, help="first N prompts (smoke)")
    a = ap.parse_args()

    out = Path(a.out); (out / "videos").mkdir(parents=True, exist_ok=True)
    rows_path = out / "rollouts.jsonl"
    done_keys = set()
    if rows_path.exists():
        done_keys = {(json.loads(l)["prompt"], json.loads(l)["episode"]) for l in rows_path.read_text().splitlines() if l.strip()}
    prompts = read_prompts(a.prompts, set(a.tags) if a.tags else None)[: a.limit]
    suite = load_suite(a.suite); tasks = [suite.get_task(i) for i in range(suite.n_tasks)]
    max_steps = a.max_steps or MAX_STEPS[a.suite]
    policy = NoraPolicy(a.checkpoint, unnorm_key=a.suite, time_horizon=a.time_horizon)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    envs = {}
    for p in prompts:
        ti, score = nearest_task(p["prompt"], tasks)
        task = tasks[ti]
        if ti not in envs:
            envs[ti] = (make_env(task), suite.get_task_init_states(ti))
        env, inits = envs[ti]
        for ep in range(a.episodes):
            if (p["prompt"], ep) in done_keys:
                continue
            vid = out / "videos" / f"{re.sub(r'[^a-z0-9]+', '_', p['prompt'].lower())[:60]}__ep{ep}.mp4"
            t0 = time.time()
            m = run_episode(policy, env, inits[ep], p["prompt"], max_steps, vid, a.freeze_eps)
            row = {"ts": stamp, "checkpoint": a.checkpoint, "suite": a.suite, **p, "episode": ep, "nearest_task": task.language,
                   "nearest_task_id": ti, "match_score": round(score, 2), "video": str(vid), "wall_s": round(time.time() - t0, 1), **m}
            with rows_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{p['tag']:11s}] ep{ep} {'OK ' if m['success'] else '-- '} steps={m['steps']:3d} freeze={m['freeze_frac']:.2f} "
                  f"path={m['path_len']:.2f} grip={m['gripper_events']} txt={m['n_text_tokens']} bad={m['invalid_decodes']} "
                  f"lat={m['policy_latency_s']:.2f}s wall={row['wall_s']}s | {p['prompt'][:50]} -> {task.language[:35]}", flush=True)
    for env, _ in envs.values():
        env.close()
    print("DONE", rows_path)


if __name__ == "__main__":
    main()
