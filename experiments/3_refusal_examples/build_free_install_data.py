#!/usr/bin/env python3
"""Training data for the free install (Section 4.1). Three stages, one script.

  A replay   GPU + simulator. Roll the BASE policy over the 10 libero_goal tasks x init states 0..9 and keep,
             for every policy call of a SUCCESSFUL episode, the 224x224 policy image and the exact ids NORA
             generated for it. Also one home-pose frame (after NUM_STEPS_WAIT) per (task, init state) pair.
  B pools    No GPU. The instruction pools and their train / held-out / never-trained split labels.
  C rows     No GPU except the FAST-label check. Assembles `train_install.jsonl` and `train_control.jsonl`:
             the same replay and paraphrase-execute rows for both models, and a halt set that differs only in
             WHICH sentences it contains (harm for the install model, a random benign concept for the control).

The label of a replay row is that call's own generated ids, so the replay rows are a behaviour-cloning copy of
the base policy and training on them alone is close to a no-op; the halt rows are the only new behaviour. The
halt label is the scripted hold chunk of the conditional-steering diagnostic
(`results/raw/pod6/c1/hold_chunk.json`), re-derived here with `build_hold_chunk`
and asserted equal to that file.

  python3 experiments/3_refusal_examples/build_free_install_data.py --stub-replay 1800 --out /tmp/f1data   # B + C only, no GPU, no sim
  python3 experiments/3_refusal_examples/build_free_install_data.py --self-test                            # paraphrase templates only
  python3 experiments/3_refusal_examples/build_free_install_data.py --smoke --out results/f1_smoke/data    # 1 task, 2 init states, 40 steps
  python3 experiments/3_refusal_examples/build_free_install_data.py --out results/f1/data                  # the real run (GPU)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from refusal.paths import DATA_DIR, RESULTS_DIR  # noqa: E402

# The libero_goal task instructions, in suite order (task_id -> language, read off the conditional-steering
# derive rollouts, which recorded both). Hard-coded so stages B and C run without LIBERO; `check_suite` asserts
# the live suite gives this list in this ORDER, because `data/twins.jsonl` addresses tasks by index.
LIBERO_GOAL_TASKS = [
    "open the middle drawer of the cabinet",
    "put the bowl on the stove",
    "put the wine bottle on top of the cabinet",
    "open the top drawer and put the bowl inside",
    "put the bowl on top of the cabinet",
    "push the plate to the front of the stove",
    "put the cream cheese in the bowl",
    "turn on the stove",
    "put the bowl on the plate",
    "put the wine bottle on the rack",
]
HARM_TRAIN_CATEGORIES = ("harm-to-person", "hazard-creation", "object-damage")
HARM_HELDOUT_CATEGORIES = ("excessive-force", "harm-to-self")
N_HARM_TRAIN_RAW, N_HARM_HELDOUT_RAW = 66, 26  # D_robot rows before scene-grounding, asserted
N_HARM_MIN = 20                                # a pool smaller than this means the grounding rule broke

# Scene-grounding of every pool built from D_robot. Many D_robot prompts name
# objects that do not exist in the LIBERO-Goal scene, so a halt trained on them could key on "unfamiliar noun"
# rather than on intent, while the lexical / OOD-benign controls are all scene-grounded. The map is whole-word
# and longest-match-first; what the map cannot rescue is dropped, and the result is de-duplicated.
NOUN_MAP = (("cookie box", "cream cheese"), ("orange juice", "wine bottle"), ("alphabet soup", "bowl"),
            ("tomato sauce", "cream cheese"), ("bbq sauce", "wine bottle"), ("soda can", "wine bottle"),
            ("moka pot", "wine bottle"), ("ramekin", "bowl"), ("ketchup", "cream cheese"),
            ("milk", "wine bottle"), ("butter", "cream cheese"), ("mug", "bowl"), ("basket", "rack"),
            ("cup", "bowl"), ("box", "cream cheese"))
SCENE_NOUNS = ("bowl", "plate", "cream cheese", "wine bottle", "stove", "cabinet", "drawer", "rack", "table",
               "person", "hand", "face", "foot", "gripper", "arm", "burner", "floor", "edge", "outlet")
# Nouns still present after the map that the LIBERO-Goal scene does not contain. Verbs that look like nouns
# ("jam the plate into the drawer", "for me to drink") are deliberately absent.
OFF_SCENE_NOUNS = ("apple", "book", "candle", "console", "dish", "egg", "frypan", "game", "glasses", "hammer",
                   "keyboard", "knife", "microwave", "notebook", "oven", "scissors", "towel", "vase", "water")

# Paraphrase templates. Deterministic, no randomness: prefix x single-swap body.
PREFIXES = ("", "please ", "now ", "go ahead and ", "robot, ", "I want you to ", "could you ", "next, ")
SWAPS = (
    (r"\bput\b", "place"),
    (r"\bput\b", "set"),
    (r"(?<!turn )\bon\b(?! top\b)", "onto"),   # not "turn on ...", not "on top of ..."
    (r"\bturn on\b", "switch on"),
    (r"\bopen\b", "pull open"),
    (r"\bthe bowl\b", "the black bowl"),
    (r"\bcabinet\b", "cupboard"),
    (r"\bwine bottle\b", "bottle of wine"),
    (r"\bcream cheese\b", "cream cheese box"),
    # Three swaps beyond the original list, so that every one of the 10 tasks reaches >= 8 DISTINCT paraphrases
    # without falling back on the bare instruction itself (see `paraphrases`).
    (r"\bpush\b", "slide"),
    (r"\bthe plate\b", "the white plate"),
    (r"\bthe stove\b", "the stovetop"),
)
HOLD_CHUNK_REF = RESULTS_DIR / "raw/pod6/c1/hold_chunk.json"
NOVEL_ROW_TYPE = "novel_benign_execute"
ROW_TYPES = ("replay", "paraphrase_execute", "halt", NOVEL_ROW_TYPE)


# --------------------------------------------------------------------------------- stage B: instruction pools

def swap_bodies(instruction: str) -> list[str]:
    """The instruction plus one variant per applicable single swap, in template order, deduplicated."""
    out = [instruction]
    for pat, rep in SWAPS:
        s = re.sub(pat, rep, instruction, count=1)
        if s != instruction and s not in out:
            out.append(s)
    return out


def paraphrases(instruction: str, n: int) -> list[str]:
    """`n` deterministic paraphrases of one instruction, prefix x body, never the bare instruction itself.

    The (body, prefix) pairs are walked diagonally, so the first few paraphrases vary both axes instead of
    giving eight prefixes of the same body. Fails loudly if the templates cannot produce `n` distinct ones."""
    bodies = swap_bodies(instruction)
    pairs = sorted(((bi, pi) for bi in range(len(bodies)) for pi in range(len(PREFIXES))),
                   key=lambda t: (t[0] + t[1], t[1], t[0]))
    out: list[str] = []
    for bi, pi in pairs:
        s = PREFIXES[pi] + bodies[bi]
        if s == instruction or s in out:
            continue
        out.append(s)
        if len(out) == n:
            return out
    raise SystemExit(f"paraphrase templates give only {len(out)} distinct variants of {instruction!r}, need {n}; "
                     f"bodies={bodies}")


def strip_prefix(sentence: str) -> tuple[str, str]:
    """(prefix, body) for a templated sentence; ("", sentence) when it carries none of the known prefixes."""
    for p in sorted((p for p in PREFIXES if p), key=len, reverse=True):
        if sentence.startswith(p):
            return p, sentence[len(p):]
    return "", sentence


def reprefix(sentence: str, rng, k: int = 2) -> list[str]:
    """`k` distinct re-prefixed variants of a sentence, never repeating its own prefix or the bare body."""
    prefix, body = strip_prefix(sentence)
    choices = [p for p in PREFIXES if p and p != prefix]
    idx = rng.choice(len(choices), size=k, replace=False)
    return [choices[int(i)] + body for i in idx]


def ground(prompt: str) -> tuple[str, list[str]]:
    """(scene-grounded prompt, off-scene nouns that survived the map). Deterministic, whole-word."""
    s = prompt
    for src, dst in NOUN_MAP:
        # "box" -> "cream cheese" only where it is not already part of "cream cheese box"
        pat = r"(?<!cream cheese )\bbox\b" if src == "box" else rf"\b{re.escape(src)}\b"
        s = re.sub(pat, dst, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s, sorted({w for w in OFF_SCENE_NOUNS if re.search(rf"\b{w}\b", s)})


def ground_pool(name: str, prompts: list[tuple[str, str]]) -> list[dict]:
    """Apply the scene-grounding rule to one D_robot pool. `prompts` is [(prompt, category)]; every dropped
    prompt and every post-mapping duplicate is printed, and the surviving N is printed."""
    kept, dropped, dups, seen = [], [], [], set()
    for p, cat in prompts:
        s, bad = ground(p)
        if bad:
            dropped.append((p, s, bad))
            continue
        if s in seen:
            dups.append((p, s))
            continue
        seen.add(s)
        kept.append({"text": s, "category": cat, "orig": p})
    print(f"[ground] {name}: {len(prompts)} raw rows -> N = {len(kept)} scene-grounded "
          f"({len(dropped)} dropped, {len(dups)} duplicates after mapping)")
    for p, s, bad in dropped:
        print(f"[ground]   DROP {','.join(bad):<22s} | {p}")
    for p, s in dups:
        print(f"[ground]   DUP  {'':<22s} | {p}" + ("" if p == s else f"  ->  {s}"))
    print(flush=True)
    return kept


def load_drobot(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"{path} is empty")
    return rows


def build_pools(a, tasks: list[str]) -> dict:
    """Every instruction pool with its split label. Pure python, no GPU, no simulator."""
    rows = load_drobot(a.drobot)
    by_cat = defaultdict(list)
    for r in rows:
        if r["condition"] == "harm":
            by_cat[r["category"]].append(r["prompt"])
    missing = [c for c in HARM_TRAIN_CATEGORIES + HARM_HELDOUT_CATEGORIES if c not in by_cat]
    if missing:
        raise SystemExit(f"D_robot {a.drobot} has no rows for harm categories {missing}; have {sorted(by_cat)}")
    raw_train = [(p, c) for c in HARM_TRAIN_CATEGORIES for p in by_cat[c]]
    raw_heldout = [(p, c) for c in HARM_HELDOUT_CATEGORIES for p in by_cat[c]]
    if len(raw_train) != N_HARM_TRAIN_RAW or len(raw_heldout) != N_HARM_HELDOUT_RAW:
        raise SystemExit(f"expected {N_HARM_TRAIN_RAW} HARM_TRAIN / {N_HARM_HELDOUT_RAW} HARM_HELDOUT D_robot "
                         f"rows, got {len(raw_train)} / {len(raw_heldout)}")
    harm_train_rows = ground_pool("HARM_TRAIN", raw_train)
    harm_heldout_rows = ground_pool("HARM_HELDOUT", raw_heldout)
    for name, rows_ in (("HARM_TRAIN", harm_train_rows), ("HARM_HELDOUT", harm_heldout_rows)):
        if len(rows_) < N_HARM_MIN:
            raise SystemExit(f"{name} is down to {len(rows_)} prompts after scene-grounding (< {N_HARM_MIN}); "
                             f"the noun map or the drop list is wrong")
    harm_train = [r["text"] for r in harm_train_rows]
    harm_heldout = [r["text"] for r in harm_heldout_rows]
    overlap = set(harm_train) & set(harm_heldout)
    if overlap:
        raise SystemExit(f"HARM_TRAIN and HARM_HELDOUT overlap after mapping: {sorted(overlap)}")
    cat_of = {r["text"]: r["category"] for r in harm_train_rows + harm_heldout_rows}
    if a.control_halt_n is None:                 # matched N: the control halts on as many sentences as the install
        a.control_halt_n = len(harm_train)

    rng = np.random.default_rng(a.seed)
    # BENIGN_PARA: per task, `--para-per-task` deterministic paraphrases, split 70/30 per task so that the
    # held-out over-refusal set covers all 10 scenes.
    para_train, para_heldout = [], []
    for ti, task in enumerate(tasks):
        ps = paraphrases(task, a.para_per_task)
        if len(set(ps)) != len(ps):
            raise SystemExit(f"duplicate paraphrases for task {ti} {task!r}")
        order = rng.permutation(len(ps))
        n_train = int(round(a.para_train_frac * len(ps)))
        tr = [ps[int(i)] for i in order[:n_train]]
        ho = [ps[int(i)] for i in order[n_train:]]
        if not tr or not ho:
            raise SystemExit(f"task {ti}: empty paraphrase split (train {len(tr)}, held-out {len(ho)})")
        para_train += [(ti, s) for s in tr]
        para_heldout += [(ti, s) for s in ho]
    dupes = [s for s, n in Counter(s for _, s in para_train + para_heldout).items() if n > 1]
    if dupes:
        raise SystemExit(f"the same paraphrase is produced by two tasks: {dupes[:5]}")

    # HARM_PARA: two prefixed variants per HARM_TRAIN prompt. Augmentation only: N stays |HARM_TRAIN|.
    harm_para = {p: reprefix(p, rng, k=2) for p in harm_train}

    # CONTROL_HALT: `--control-halt-n` sentences drawn from BENIGN_PARA_TRAIN. A halt sentence's two extra
    # phrasings are the same BODY under other prefixes, so a whole body is the unit that gets consumed: to
    # guarantee the control model keeps an execute set on every scene, `--control-reserve` bodies per task
    # (the smallest ones, so the drawable pool loses as little as possible) are held back entirely.
    drawable = []
    for ti, task in enumerate(tasks):
        by_body = defaultdict(list)
        for t, s in para_train:
            if t == ti:
                by_body[strip_prefix(s)[1]].append(s)
        if len(by_body) <= a.control_reserve:
            raise SystemExit(f"task {ti}: only {len(by_body)} paraphrase bodies, cannot reserve "
                             f"{a.control_reserve}; raise --para-per-task or lower --control-reserve")
        order = sorted(by_body, key=lambda b: (len(by_body[b]), b))
        for body in order[a.control_reserve:]:
            drawable += [(ti, s) for s in by_body[body]]
    if len(drawable) < a.control_halt_n:
        raise SystemExit(f"only {len(drawable)} drawable benign paraphrases, need {a.control_halt_n} for "
                         f"CONTROL_HALT; raise --para-per-task or lower --control-halt-n")
    # At most `--body-reuse` halt originals may share a body, because each original claims three of the seven
    # non-empty prefixes of its body for its three phrasings; without that cap two originals of the same body
    # would generate the same sentence and the halt set would silently shrink.
    body_used, control_halt = Counter(), []
    for i in rng.permutation(len(drawable)):
        ti, s = drawable[int(i)]
        body = strip_prefix(s)[1]
        if body_used[body] >= a.body_reuse:
            continue
        body_used[body] += 1
        control_halt.append((ti, s))
        if len(control_halt) == a.control_halt_n:
            break
    if len(control_halt) != a.control_halt_n:
        raise SystemExit(f"only {len(control_halt)} halt originals with at most {a.body_reuse} per body "
                         f"({len(body_used)} distinct bodies in {len(drawable)} drawable paraphrases); "
                         f"raise --para-per-task")
    control_halt.sort(key=lambda t: (t[0], t[1]))
    claimed = defaultdict(set)                      # body -> prefixes already used by some halt phrasing
    for _, s in control_halt:
        p, body = strip_prefix(s)
        claimed[body].add(p)
    control_phrasings = {}
    for _, s in control_halt:
        p, body = strip_prefix(s)
        free = [q for q in PREFIXES if q and q not in claimed[body]]
        if len(free) < 2:
            raise SystemExit(f"no free prefix left for body {body!r} (claimed {sorted(claimed[body])})")
        idx = rng.choice(len(free), size=2, replace=False)
        extra = [free[int(j)] for j in idx]
        claimed[body].update(extra)
        control_phrasings[s] = [s] + [q + body for q in extra]
        if len(set(control_phrasings[s])) != 3:
            raise SystemExit(f"control halt phrasings collide for {s!r}: {control_phrasings[s]}")
    halt_sentences = {s for v in control_phrasings.values() for s in v}
    if len(halt_sentences) != 3 * a.control_halt_n:
        raise SystemExit(f"{len(halt_sentences)} distinct control halt phrasings, expected "
                         f"{3 * a.control_halt_n}: the prefix claims collided")
    if halt_sentences & set(tasks):
        raise SystemExit(f"a control halt phrasing is a trained instruction: {sorted(halt_sentences & set(tasks))}")
    # The control's execute pool = every training paraphrase that is not one of its halt phrasings.
    control_execute = [(ti, s) for ti, s in para_train if s not in halt_sentences]
    per_task = Counter(ti for ti, _ in control_execute)
    empty = [ti for ti in range(len(tasks)) if not per_task[ti]]
    if empty:
        raise SystemExit(f"control execute pool is empty for tasks {empty}")

    train_texts = {s for _, s in para_train}
    # The same scene-grounding rule on the two never-trained D_robot pools, so every evaluation sentence is
    # grounded too (the simulator evaluation may re-apply NOUN_MAP from pools.json; the map is idempotent).
    ood_rows = ground_pool("OOD_BENIGN", [(r["prompt"], "") for r in rows if r["condition"] == "ood_benign"])
    lex_rows = ground_pool("LEXICAL", [(r["prompt"], "") for r in rows if r["condition"] == "lexical"])
    never = [{"text": l.split("\t", 1)[1].strip(), "orig": l.split("\t", 1)[1].strip(),
              "pool": "demo_prompts", "tag": l.split("\t", 1)[0].strip(), "split": "never"}
             for l in Path(a.demo_prompts).read_text().splitlines() if l.strip() and not l.startswith("#")]
    never += [{**r, "pool": "ood_benign", "split": "never"} for r in ood_rows]
    never += [{**r, "pool": "lexical", "split": "never"} for r in lex_rows]
    never += [{**r, "pool": "harm_heldout", "split": "never"} for r in harm_heldout_rows]
    pools = {
        "meta": {"seed": int(a.seed), "para_per_task": int(a.para_per_task),
                 "para_train_frac": float(a.para_train_frac), "control_halt_n": int(a.control_halt_n),
                 "control_reserve": int(a.control_reserve), "body_reuse": int(a.body_reuse),
                 "drobot": str(a.drobot), "tasks": tasks,
                 "n_harm_train": len(harm_train), "n_harm_heldout": len(harm_heldout),
                 "scene_grounding": "spec addendum 2026-09-18 01:30"},
        "NOUN_MAP": [list(kv) for kv in NOUN_MAP],
        "SCENE_NOUNS": list(SCENE_NOUNS),
        "OFF_SCENE_NOUNS": list(OFF_SCENE_NOUNS),
        "harm_train": [{**r, "split": "train"} for r in harm_train_rows],
        "harm_heldout": [{**r, "split": "never"} for r in harm_heldout_rows],
        "harm_para": [{"text": v, "origin": p, "category": cat_of[p], "split": "train"}
                      for p, vs in harm_para.items() for v in vs],
        "benign_para": [{"text": s, "task_id": ti, "task": tasks[ti],
                         "split": "train" if s in train_texts else "heldout"}
                        for ti, s in para_train + para_heldout],
        "control_halt": [{"text": v, "origin": s, "task_id": ti, "split": "train"}
                         for ti, s in control_halt for v in control_phrasings[s]],
        "control_execute": [{"text": s, "task_id": ti, "split": "train"} for ti, s in control_execute],
        "never_trained": never,
    }
    pools["_lists"] = {  # convenience for stage C, dropped before pools.json is written
        "harm_train": harm_train, "harm_para": harm_para, "para_train": para_train,
        "para_heldout": para_heldout, "control_halt": control_halt,
        "control_phrasings": control_phrasings, "control_execute": control_execute}
    return pools


def print_pools(pools) -> None:
    print(f"\n{'pool':<22s}{'n':>6s}  {'split':<10s}note")
    rows = [("HARM_TRAIN", len(pools["harm_train"]), "train", "D_robot " + ",".join(HARM_TRAIN_CATEGORIES)),
            ("HARM_PARA", len(pools["harm_para"]), "train", "2 prefixed variants per HARM_TRAIN prompt"),
            ("HARM_HELDOUT", len(pools["harm_heldout"]), "never", "D_robot " + ",".join(HARM_HELDOUT_CATEGORIES)),
            ("BENIGN_PARA_TRAIN", len(pools["_lists"]["para_train"]), "train", "paraphrase-execute sentences"),
            ("BENIGN_PARA_HELDOUT", len(pools["_lists"]["para_heldout"]), "heldout", "over-refusal test"),
            ("CONTROL_HALT", len(pools["control_halt"]), "train",
             f"{len(pools['_lists']['control_halt'])} originals x 3 phrasings"),
            ("CONTROL_EXECUTE", len(pools["control_execute"]), "train", "control's paraphrase-execute pool"),
            ("NEVER_TRAINED", len(pools["never_trained"]), "never", "demo + ood_benign + lexical + HARM_HELDOUT")]
    for name, n, split, note in rows:
        print(f"{name:<22s}{n:6d}  {split:<10s}{note}")
    print(flush=True)


# --------------------------------------------------------------------------------- stage A: replay collection

def check_suite(tasks_live: list[str]) -> None:
    """Order matters: `data/twins.jsonl` and the frame file names address tasks by suite index."""
    if list(tasks_live) != LIBERO_GOAL_TASKS:
        diff = [(i, a_, b) for i, (a_, b) in enumerate(zip(tasks_live, LIBERO_GOAL_TASKS)) if a_ != b]
        raise SystemExit(f"libero_goal task order changed: {len(tasks_live)} live tasks, first mismatches "
                         f"{diff[:3]}; live list = {list(tasks_live)}")


def save_png(img: np.ndarray, path: Path) -> None:
    import PIL.Image
    path.parent.mkdir(parents=True, exist_ok=True)
    PIL.Image.fromarray(np.asarray(img, dtype=np.uint8)).save(path)


def load_png(path: Path) -> np.ndarray:
    import PIL.Image
    if not Path(path).exists():
        raise SystemExit(f"missing frame {path}")
    return np.asarray(PIL.Image.open(path).convert("RGB"), dtype=np.uint8)


def keep_call(gen_ids: list[int], invalid: bool, n_text: int) -> bool:
    """A replay label must be exactly what NORA emitted: action tokens then <|im_end|>, nothing else."""
    from refusal.models import IM_END
    from refusal.nora_policy import ACTION_MAX, ACTION_MIN
    if invalid or n_text or len(gen_ids) < 2:
        return False
    return gen_ids[-1] == IM_END and all(ACTION_MIN <= t <= ACTION_MAX for t in gen_ids[:-1])


def stage_a(a, out: Path, policy, suite, tasks_live: list) -> None:
    """Base-policy rollouts; appends to replay.jsonl / episodes.jsonl and writes the PNGs. Resumable per
    (task, init state) pair."""
    from refusal.libero_env import DUMMY_ACTION, NUM_STEPS_WAIT, make_env, normalize_gripper, policy_image

    replay_path, ep_path = out / "replay.jsonl", out / "episodes.jsonl"
    done = {(r["task_id"], r["init_state"]) for r in read_jsonl(ep_path)}
    task_ids = list(range(min(a.tasks, len(tasks_live))))
    todo = [(ti, i) for ti in task_ids for i in range(a.init_states) if (ti, i) not in done]
    print(f"[stageA] {len(todo)} episodes to run ({len(done)} already done), {a.max_steps} steps, "
          f"checkpoint {a.checkpoint}", flush=True)
    n_kept = n_dropped = 0
    for ti in task_ids:
        pairs = [(t, i) for t, i in todo if t == ti]
        if not pairs:
            continue
        task = tasks_live[ti]
        env = make_env(task)
        inits = suite.get_task_init_states(ti)
        try:
            for _, init in pairs:
                t0 = time.time()
                env.reset()
                obs = env.set_init_state(inits[init])
                t, done_flag = 0, False
                while t < NUM_STEPS_WAIT:
                    obs, _, d, _ = env.step(DUMMY_ACTION)
                    done_flag = done_flag or bool(d)
                    t += 1
                save_png(policy_image(obs), out / "home_frames" / f"t{ti}_i{init}.png")
                calls, limit = [], a.max_steps + NUM_STEPS_WAIT
                while t < limit and not done_flag:
                    img = policy_image(obs)
                    r = policy.act(img, task.language)
                    calls.append({"image": img, "step": int(t), "call": len(calls),
                                  "gen_ids": [int(x) for x in r["gen_ids"]],
                                  "invalid": bool(r["invalid"]), "n_text": int(r["n_text"])})
                    chunk = np.array([DUMMY_ACTION], dtype=np.float64) if r["invalid"] else normalize_gripper(r["action"])
                    for a_row in np.asarray(chunk, dtype=np.float64):
                        obs, _, d, _ = env.step(a_row.tolist())
                        done_flag = done_flag or bool(d)
                        t += 1
                        if done_flag or t >= limit:
                            break
                keep = bool(done_flag) or a.keep_failed
                n_calls_kept = 0
                if keep:
                    for c in calls:
                        if not keep_call(c["gen_ids"], c["invalid"], c["n_text"]):
                            n_dropped += 1
                            continue
                        rel = f"frames/t{ti}_i{init}_c{c['call']:04d}.png"
                        save_png(c["image"], out / rel)
                        append_jsonl(replay_path, {"image": rel, "instruction": task.language, "task_id": ti,
                                                   "task": task.language, "init_state": int(init),
                                                   "step": c["step"], "call": c["call"], "label_ids": c["gen_ids"]})
                        n_calls_kept += 1
                        n_kept += 1
                else:
                    n_dropped += len(calls)
                append_jsonl(ep_path, {"task_id": ti, "task": task.language, "init_state": int(init),
                                       "success": bool(done_flag), "n_calls": len(calls),
                                       "n_calls_kept": n_calls_kept, "steps": int(t),
                                       "home_frame": f"home_frames/t{ti}_i{init}.png",
                                       "wall_s": round(time.time() - t0, 1)})
                print(f"[stageA] t{ti} i{init} {'OK ' if done_flag else '-- '}calls={len(calls):3d} "
                      f"kept={n_calls_kept:3d} steps={t:3d} wall={time.time() - t0:5.1f}s | {task.language[:40]}",
                      flush=True)
        finally:
            env.close()
    eps = read_jsonl(ep_path)
    n_succ = sum(1 for r in eps if r["success"])
    print(f"[stageA] {len(eps)} episodes, {n_succ} successful ({n_succ / max(1, len(eps)):.2f}), "
          f"{n_kept} calls kept this run, {n_dropped} dropped (failed episode / invalid / text token)", flush=True)


def stub_replay(n: int, tasks: list[str], init_states: int, label_ids: list[int], seed: int) -> list[dict]:
    """A fake replay manifest for local testing: real row shape, image paths that do not exist, hold-chunk ids
    as labels. Never produced on the GPU machine."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        ti = int(rng.integers(0, len(tasks)))
        init = int(rng.integers(0, init_states))
        call = k % 55
        rows.append({"image": f"frames/t{ti}_i{init}_c{call:04d}.png", "instruction": tasks[ti], "task_id": ti,
                     "task": tasks[ti], "init_state": init, "step": 10 + 5 * call, "call": call,
                     "label_ids": list(label_ids)})
    return rows


# --------------------------------------------------------------------------------- stage C: training rows

def read_jsonl(path: Path) -> list[dict]:
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def append_jsonl(path: Path, row: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as f:
        f.write(json.dumps(row) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def halt_frames(a, rng, home: list[dict], mid: list[dict], n: int) -> list[dict]:
    """`n` distinct frames for one halt sentence: home pose with probability `--home-frac`, else a replay frame
    from mid-episode (step >= --mid-step)."""
    picked, guard = [], 0
    while len(picked) < n and guard < 200:
        guard += 1
        src = home if (rng.random() < a.home_frac or not mid) else mid
        f = src[int(rng.integers(0, len(src)))]
        if f["image"] not in {p["image"] for p in picked}:
            picked.append(f)
    if len(picked) != n:
        raise SystemExit(f"could not draw {n} distinct halt frames (home {len(home)}, mid {len(mid)})")
    return picked


def build_rows(a, pools: dict, replay: list[dict], home: list[dict], hold_gen_ids: list[int]) -> dict:
    """The two training files, stage C."""
    L = pools["_lists"]
    tasks = pools["meta"]["tasks"]
    for r in replay:
        r["source"] = "replay_mid"
    mid = [r for r in replay if r["step"] >= a.mid_step]
    if not mid:
        msg = f"no replay call with step >= {a.mid_step}; the 30% mid-episode halt frames cannot be drawn"
        if not a.smoke:
            raise SystemExit(msg)
        print(f"[stageC] WARNING (smoke): {msg}; using any replay frame instead", flush=True)
        mid = list(replay)

    replay_rows = [{"row_type": "replay", "image": r["image"], "instruction": r["instruction"],
                    "label_ids": r["label_ids"], "task_id": r["task_id"], "pool": "replay",
                    "step": r["step"], "call": r["call"], "init_state": r["init_state"]} for r in replay]

    by_task = defaultdict(list)
    for r in replay:
        by_task[r["task_id"]].append(r)

    def para_rows(pool: list[tuple[int, str]], tag: str) -> list[dict]:
        """`--para-rows` rows: a replay call (image + its own ids) relabelled with a paraphrase of its task."""
        per_task = defaultdict(list)
        for ti, s in pool:
            per_task[ti].append(s)
        usable = [ti for ti in by_task if per_task[ti]]
        if not usable:
            raise SystemExit(f"no task has both replay calls and {tag} sentences")
        rng = np.random.default_rng(a.seed + 1)
        rows = []
        for _ in range(a.para_rows):
            ti = usable[int(rng.integers(0, len(usable)))]
            r = by_task[ti][int(rng.integers(0, len(by_task[ti])))]
            s = per_task[ti][int(rng.integers(0, len(per_task[ti])))]
            rows.append({"row_type": "paraphrase_execute", "image": r["image"], "instruction": s,
                         "label_ids": r["label_ids"], "task_id": ti, "pool": tag,
                         "step": r["step"], "call": r["call"], "init_state": r["init_state"]})
        return rows

    def halt_rows(sentences: list[tuple[str, str]], tag: str, seed_off: int) -> list[dict]:
        rng = np.random.default_rng(a.seed + seed_off)
        rows = []
        for text, origin in sentences:
            for f in halt_frames(a, rng, home, mid, a.halt_frames):
                rows.append({"row_type": "halt", "image": f["image"], "instruction": text,
                             "label_ids": list(hold_gen_ids), "task_id": f["task_id"], "pool": tag,
                             "origin": origin, "frame_source": f["source"]})
        return rows

    install_halt = [(p, p) for p in L["harm_train"]] + [(v, p) for p, vs in L["harm_para"].items() for v in vs]
    control_halt = [(v, s) for _, s in L["control_halt"] for v in L["control_phrasings"][s]]
    if len(install_halt) != len(set(t for t, _ in install_halt)):
        raise SystemExit("duplicate sentence in the install halt set")
    if len(control_halt) != len(set(t for t, _ in control_halt)):
        raise SystemExit("duplicate sentence in the control halt set")

    install = replay_rows + para_rows(L["para_train"], "BENIGN_PARA_TRAIN") +\
        halt_rows(install_halt, "HARM_TRAIN+HARM_PARA", 2)
    control = replay_rows + para_rows(L["control_execute"], "CONTROL_EXECUTE") +\
        halt_rows(control_halt, "CONTROL_HALT", 2)
    for name, rows, halt_set in (("install", install, {t for t, _ in install_halt}),
                                 ("control", control, {t for t, _ in control_halt})):
        exec_texts = {r["instruction"] for r in rows if r["row_type"] != "halt"}
        clash = exec_texts & halt_set
        if clash:
            raise SystemExit(f"{name}: {len(clash)} sentence(s) are labelled both execute and halt: {sorted(clash)[:3]}")
    return {"install": install, "control": control}


def row_table(files: dict) -> dict:
    """Row counts per type and the halt fraction; printed verbatim and stored in build_summary.json."""
    stats = {}
    print(f"\n{'file':<12s}" + "".join(f"{t:>22s}" for t in ROW_TYPES) + f"{'total':>9s}{'halt_frac':>11s}")
    for name, rows in files.items():
        c = Counter(r["row_type"] for r in rows)
        total = len(rows)
        stats[name] = {**{t: int(c[t]) for t in ROW_TYPES}, "total": total,
                       "halt_frac": float(c["halt"] / total) if total else 0.0,
                       "n_images": len({r["image"] for r in rows}),
                       "n_instructions": len({r["instruction"] for r in rows})}
        print(f"{name:<12s}" + "".join(f"{c[t]:22d}" for t in ROW_TYPES)
              + f"{total:9d}{stats[name]['halt_frac']:11.3f}")
    for name, s in stats.items():
        print(f"[rows] {name}: {s['n_images']} distinct frames, {s['n_instructions']} distinct instructions")
    print(flush=True)
    return stats


def check_twins(twins_path: Path, files: dict) -> int:
    """The executable-harm twins are EVALUATION ONLY. Three assertions, run in stage C.

    1. The noun map leaves every twin `harm` text unchanged: the twins are written scene-grounded, so the
       simulator evaluation can apply NOUN_MAP at load time without silently rewriting them.
    2. No twin `harm` wording is trained. A `context` twin is the one legitimate exception: its harm text IS the
       trained instruction and only the scene is edited at init, so identical wording is accepted there and
       nowhere else.
    3. Each twin's `safe` field is the instruction the suite gives for its `task_id` (catches an index drift
       between `data/twins.jsonl` and the suite order)."""
    twins = read_jsonl(twins_path)
    if not twins:
        raise SystemExit(f"{twins_path} is empty; the twins are part of the spec (addendum 2)")
    trained = {r["instruction"] for rows in files.values() for r in rows}
    n_context = 0
    for t in twins:
        for field in ("id", "task_id", "safe", "harm", "kind"):
            if field not in t:
                raise SystemExit(f"twin {t.get('id')!r} has no {field!r}")
        g, bad = ground(t["harm"])
        if g != t["harm"] or bad:
            raise SystemExit(f"twin {t['id']}: the noun map rewrites the harm text {t['harm']!r} -> {g!r} "
                             f"(off-scene nouns {bad}); twins must already be scene-grounded")
        ti = int(t["task_id"])
        if not 0 <= ti < len(LIBERO_GOAL_TASKS) or LIBERO_GOAL_TASKS[ti] != t["safe"]:
            raise SystemExit(f"twin {t['id']}: task_id {ti} is {LIBERO_GOAL_TASKS[ti] if 0 <= ti < len(LIBERO_GOAL_TASKS) else 'out of range'!r} "
                             f"but safe is {t['safe']!r}; twins and the suite order disagree")
        if t["harm"] == t["safe"]:
            if t["kind"] != "context":
                raise SystemExit(f"twin {t['id']}: harm text equals safe text but kind is {t['kind']!r}; only a "
                                 f"context twin may reuse the trained wording")
            n_context += 1
            continue
        if t["harm"] in trained:
            raise SystemExit(f"twin {t['id']} harm text is in a training file: {t['harm']!r}")
    print(f"twins: {len(twins)} held out ({len(twins) - n_context} harm wordings absent from both training "
          f"files, {n_context} context twin(s) reuse the trained wording by design, noun map idempotent, "
          f"task ids match the suite order)", flush=True)
    return len(twins)


def check_labels(policy, rows: list[dict], horizon: int) -> int:
    """Assert every distinct label decodes to a valid FAST chunk of `horizon` steps (GPU only)."""
    from refusal.nora_policy import ACTION_MAX, ACTION_MIN
    from refusal.models import IM_END
    ft = policy.nora.fast_tokenizer
    seen, n = set(), 0
    for r in rows:
        key = tuple(r["label_ids"])
        if key in seen:
            continue
        seen.add(key)
        ids = list(key)
        if ids[-1] != IM_END:
            raise SystemExit(f"label does not end with <|im_end|>: {ids}")
        act = [t - ACTION_MIN for t in ids[:-1]]
        if not act or any(t < 0 or t > ACTION_MAX - ACTION_MIN for t in act):
            raise SystemExit(f"label has ids outside the FAST action range: {ids}")
        if len(ft.bpe_tokenizer.decode(act)) != horizon * int(ft.action_dim):
            raise SystemExit(f"label decodes to the wrong number of coefficients: {ids}")
        dec = np.asarray(ft.decode([act]))
        if dec.reshape(-1, int(ft.action_dim)).shape[0] != horizon:
            raise SystemExit(f"label decodes to {dec.shape} steps, expected horizon {horizon}: {ids}")
        n += 1
    print(f"[labels] {n} distinct labels, all decode to a FAST chunk of horizon {horizon}", flush=True)
    return n


# --------------------------------------------------------------------------------- stage rows_v2: novel benign

def read_novel_benign(path: Path, pools: dict, twins_path: Path) -> list[str]:
    """The v2 NOVEL-BENIGN-EXECUTE sentences, with every disjointness assertion of the v2 data.

    v1's token check showed both the install and the control model learning "unfamiliar wording -> halt": the
    template paraphrases sit too close to the trained wording to teach "unfamiliar but benign -> execute". These
    sentences are the missing negative: benign, scene-grounded, and worded unlike anything else in the data."""
    lines = [l.strip() for l in Path(path).read_text().splitlines()]
    sents = [l for l in lines if l and not l.startswith("#")]
    if not sents:
        raise SystemExit(f"{path} has no sentences")
    dupes = [s for s, n in Counter(sents).items() if n > 1]
    if dupes:
        raise SystemExit(f"duplicate novel benign sentences: {dupes[:5]}")
    for s in sents:
        g, bad = ground(s)
        if g != s or bad:
            raise SystemExit(f"the noun map rewrites the novel benign sentence {s!r} -> {g!r} "
                             f"(off-scene nouns {bad}); it must already be scene-grounded")
    forbidden: dict[str, str] = {}
    for key in ("harm_train", "harm_heldout", "harm_para", "benign_para", "control_halt", "control_execute",
                "never_trained"):
        for r in pools[key]:
            forbidden.setdefault(r["text"], key if key != "never_trained" else f"never_trained/{r['pool']}")
    for t in pools["meta"]["tasks"]:
        forbidden.setdefault(t, "trained_instruction")
    for t in read_jsonl(twins_path):
        for k in ("safe", "benign", "harm"):
            if k in t:
                forbidden.setdefault(t[k], f"twins.{k}")
    clash = [(s, forbidden[s]) for s in sents if s in forbidden]
    if clash:
        lines = "\n".join(f"  {s!r}  already in  {w}" for s, w in clash)
        raise SystemExit(f"{len(clash)} of {len(sents)} novel benign sentence(s) already appear elsewhere; "
                         f"delete them from {path} and rerun:\n{lines}\n"
                         f"(a sentence can collide only AFTER the noun map: the pools store the GROUNDED text, "
                         f"which is what the evaluation uses)")
    print(f"[novel] {len(sents)} novel benign sentences from {path}: disjoint by exact text from "
          f"{len(forbidden)} known texts (pools, trained instructions, demo prompts, ood_benign, lexical, "
          f"twins), noun map is a no-op on all of them", flush=True)
    return sents


def novel_benign_rows(a, policy, sents, home, mid, out: Path, stub_label: list[int]) -> tuple[list[dict], int]:
    """One row per (sentence, frame) with the BASE policy's own greedy generation as the label.

    Same frame recipe as the halt rows (`--halt-frames` frames, `--home-frac` from the home pose), so the halt
    set and the novel-benign set differ only in the sentence and in what the label says to do."""
    rng = np.random.default_rng(a.seed)
    rows, dropped, t0 = [], 0, time.time()
    for i, s in enumerate(sents):
        for f in halt_frames(a, rng, home, mid, a.halt_frames):
            if policy is None:                      # --stub-generator: row assembly only, labels are fabricated
                gen, keep = list(stub_label), True
            else:
                r = policy.act(load_png(out / f["image"]), s)
                gen = [int(x) for x in r["gen_ids"]]
                keep = keep_call(gen, bool(r["invalid"]), int(r["n_text"]))
            if not keep:
                dropped += 1
                continue
            rows.append({"row_type": NOVEL_ROW_TYPE, "image": f["image"], "instruction": s, "label_ids": gen,
                         "task_id": f["task_id"], "pool": "NOVEL_BENIGN", "frame_source": f["source"]})
        if policy is not None and (i + 1) % 20 == 0:
            print(f"[novel] {i + 1}/{len(sents)} sentences, {len(rows)} rows, {dropped} dropped, "
                  f"{time.time() - t0:.0f}s", flush=True)
    print(f"[novel] {len(rows)} rows from {len(sents)} sentences x {a.halt_frames} frames, {dropped} dropped "
          f"(invalid decode or text token), {time.time() - t0:.0f}s", flush=True)
    return rows, dropped


def rows_v2(a, out: Path, stamp: str) -> None:
    """v2 training files = the v1 rows exactly as written, plus the novel-benign-execute rows. No new rollouts:
    the replay manifest, the frames and the v1 files are read back from `--out`."""
    pools = json.loads((out / "pools.json").read_text())
    replay, episodes = read_jsonl(out / "replay.jsonl"), read_jsonl(out / "episodes.jsonl")
    v1 = {"install": read_jsonl(out / "train_install.jsonl"), "control": read_jsonl(out / "train_control.jsonl")}
    for name, rows in v1.items():
        if not rows:
            raise SystemExit(f"{out / f'train_{name}.jsonl'} is empty; run --stage rows first")
    if not replay or not episodes:
        raise SystemExit(f"{out} has no replay.jsonl / episodes.jsonl; run --stage replay first")
    sents = read_novel_benign(Path(a.novel_benign), pools, Path(a.twins))
    home = [{"image": r["home_frame"], "task_id": r["task_id"], "source": "home"} for r in episodes]
    for r in replay:
        r["source"] = "replay_mid"
    mid = [r for r in replay if r["step"] >= a.mid_step]
    if not mid:
        raise SystemExit(f"no replay call with step >= {a.mid_step}; the 30% mid-episode frames cannot be drawn")
    stub_label = [int(i) for i in json.loads(Path(a.hold_chunk_ref).read_text())["gen_ids"]]
    policy = None
    if not a.stub_generator:
        from refusal.nora_policy import NoraPolicy                # late: torch only exists on the GPU machine
        policy = NoraPolicy(a.checkpoint, unnorm_key=a.suite, time_horizon=a.time_horizon)
    else:
        print("[stub] labels fabricated from the pod-6 hold chunk; the base policy is NOT queried and no "
              "label is validated. Stub v2 data is NOT a result.", flush=True)
    novel, dropped = novel_benign_rows(a, policy, sents, home, mid, out, stub_label)
    if not novel:
        raise SystemExit("every novel benign generation was dropped; the base policy produced no valid chunk")
    write_jsonl(out / "novel_benign_rows.jsonl", novel)
    files = {"install_v2": v1["install"] + novel, "control_v2": v1["control"] + novel}
    stats = row_table(files)
    check_twins(Path(a.twins), files)
    if policy is not None:
        check_labels(policy, novel, a.time_horizon)
    else:
        print("[labels] SKIPPED: no tokenizer on this machine (--stub-generator)", flush=True)
    write_jsonl(out / "train_install_v2.jsonl", files["install_v2"])
    write_jsonl(out / "train_control_v2.jsonl", files["control_v2"])
    summary = {"exp": "f1_build_data_v2", "ts": stamp, "out": str(out), "checkpoint": a.checkpoint,
               "seed": a.seed, "novel_benign": str(a.novel_benign), "n_novel_sentences": len(sents),
               "halt_frames": int(a.halt_frames), "home_frac": float(a.home_frac),
               "n_novel_rows": len(novel), "n_dropped": int(dropped), "stub_generator": bool(a.stub_generator),
               "rows": stats, "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}}
    (out / "build_summary_v2.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out / 'novel_benign_rows.jsonl'} ({len(novel)} rows), {out / 'train_install_v2.jsonl'} "
          f"({len(files['install_v2'])} rows), {out / 'train_control_v2.jsonl'} "
          f"({len(files['control_v2'])} rows), {out / 'build_summary_v2.json'}")
    print("DONE", out)


# --------------------------------------------------------------------------------- self-test (no GPU, no sim)

def self_test(a) -> None:
    """The paraphrase templates: deterministic, >= 8 per task, no duplicates, seeded 70/30 split."""
    for task in LIBERO_GOAL_TASKS:
        ps = paraphrases(task, a.para_per_task)
        assert len(ps) == a.para_per_task, (task, len(ps))
        assert len(set(ps)) == len(ps), f"duplicate paraphrase for {task!r}"
        assert a.para_per_task >= 8, "spec section 2 requires >= 8 paraphrases per task"
        assert task not in ps, f"the bare instruction is in its own paraphrase set: {task!r}"
        assert ps == paraphrases(task, a.para_per_task), f"paraphrases are not deterministic for {task!r}"
        bodies = swap_bodies(task)
        print(f"[self-test] {task!r}: {len(bodies)} bodies, {len(ps)} paraphrases, e.g. {ps[:3]}")
    pools_a = build_pools(a, LIBERO_GOAL_TASKS)
    pools_b = build_pools(a, LIBERO_GOAL_TASKS)
    for k in ("harm_train", "harm_para", "benign_para", "control_halt", "control_execute"):
        assert [r["text"] for r in pools_a[k]] == [r["text"] for r in pools_b[k]], f"{k} is not deterministic"
    tr = pools_a["_lists"]["para_train"]
    ho = pools_a["_lists"]["para_heldout"]
    n = len(tr) + len(ho)
    assert not ({s for _, s in tr} & {s for _, s in ho}), "train / held-out paraphrases overlap"
    assert abs(len(tr) / n - a.para_train_frac) < 0.05, (len(tr), n)
    per_task = Counter(ti for ti, _ in ho)
    assert all(per_task[ti] > 0 for ti in range(len(LIBERO_GOAL_TASKS))), "a task has no held-out paraphrase"
    halt = {r["text"] for r in pools_a["control_halt"]}
    execs = {r["text"] for r in pools_a["control_execute"]}
    assert not (halt & execs), "control halt and execute sets overlap"
    assert not (halt & set(LIBERO_GOAL_TASKS)), "a control halt sentence is a trained instruction"
    print(f"[self-test] split {len(tr)} train / {len(ho)} held-out ({len(tr) / n:.2f} train), "
          f"control halt {len(halt)} sentences, control execute {len(execs)}")
    print("[self-test] OK", flush=True)


# --------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS_DIR / "f1/data"))
    ap.add_argument("--checkpoint", default="declare-lab/nora-long-finetuned-libero-goal")
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--stage", choices=("all", "replay", "rows", "rows_v2"), default="all")
    ap.add_argument("--drobot", default=str(DATA_DIR / "d_robot.jsonl"))
    ap.add_argument("--demo-prompts", default=str(DATA_DIR / "demo_prompts.txt"))
    ap.add_argument("--twins", default=str(DATA_DIR / "twins.jsonl"), help="evaluation-only twins, asserted "
                    "absent from both training files")
    ap.add_argument("--novel-benign", default=str(DATA_DIR / "novel_benign.txt"),
                    help="stage rows_v2: the NOVEL-BENIGN-EXECUTE sentences")
    ap.add_argument("--stub-generator", action="store_true", help="stage rows_v2, local test: fabricate the "
                    "labels instead of querying the base policy, no GPU")
    ap.add_argument("--hold-chunk-ref", default=str(HOLD_CHUNK_REF))
    ap.add_argument("--tasks", type=int, default=10, help="first N libero_goal tasks (smoke)")
    ap.add_argument("--init-states", type=int, default=10, help="init states 0..N-1 for the replay rollouts")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--time-horizon", type=int, default=5)
    ap.add_argument("--para-per-task", type=int, default=14, help="paraphrases per trained instruction")
    ap.add_argument("--para-train-frac", type=float, default=0.7)
    ap.add_argument("--para-rows", type=int, default=500, help="paraphrase-execute rows")
    ap.add_argument("--halt-frames", type=int, default=3, help="frames per halt sentence")
    ap.add_argument("--home-frac", type=float, default=0.7, help="share of halt frames taken from the home pose")
    ap.add_argument("--mid-step", type=int, default=30, help="a replay frame counts as mid-episode from this step")
    ap.add_argument("--control-halt-n", type=int, default=None, help="originals in CONTROL_HALT "
                    "(default: |HARM_TRAIN| after scene-grounding, so the two halt sets have matched N)")
    ap.add_argument("--control-reserve", type=int, default=1, help="paraphrase bodies held back per task as the "
                    "control model's execute-only set")
    ap.add_argument("--body-reuse", type=int, default=2, help="max CONTROL_HALT originals sharing one body")
    ap.add_argument("--keep-failed", action="store_true", help="keep calls from failed episodes (smoke only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stub-replay", type=int, default=0, help="local test: fabricate N replay rows, no GPU/sim")
    ap.add_argument("--self-test", action="store_true", help="paraphrase / pool assertions only, then exit")
    ap.add_argument("--smoke", action="store_true", help="1 task, 2 init states, 40 steps, small row counts")
    a = ap.parse_args()
    if a.smoke:
        a.tasks, a.init_states, a.max_steps = 1, 2, 40
        a.para_rows, a.control_halt_n, a.keep_failed, a.control_reserve = 20, 4, True, 1
        print(f"[smoke] 1 task / 2 init states / 40 steps, 20 paraphrase-execute rows, {a.halt_frames} halt "
              f"frames per sentence, CONTROL_HALT 4 (all one scene allows), calls from failed episodes kept. "
              f"Smoke data is NOT a result.", flush=True)
    if a.self_test:
        self_test(a)
        return

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if a.stage == "rows_v2":
        rows_v2(a, out, stamp)
        return
    ref = json.loads(Path(a.hold_chunk_ref).read_text())
    policy = None

    if a.stub_replay:
        if a.stage != "all":
            raise SystemExit("--stub-replay only makes sense with --stage all")
        tasks = LIBERO_GOAL_TASKS[: a.tasks]
        hold_gen_ids = [int(i) for i in ref["gen_ids"]]
        replay = stub_replay(a.stub_replay, tasks, a.init_states, hold_gen_ids, a.seed)
        write_jsonl(out / "replay.jsonl", replay)
        write_jsonl(out / "episodes.jsonl",
                    [{"task_id": ti, "task": tasks[ti], "init_state": i, "success": True, "n_calls": 0,
                      "n_calls_kept": 0, "steps": 0, "home_frame": f"home_frames/t{ti}_i{i}.png", "wall_s": 0.0}
                     for ti in range(len(tasks)) for i in range(a.init_states)])
        print(f"[stub] {len(replay)} fabricated replay rows, {len(tasks) * a.init_states} fabricated home "
              f"frames; images do NOT exist and no label is validated", flush=True)
    else:
        from refusal.conditional import build_hold_chunk          # late: torch / LIBERO only exist on the GPU machine
        from refusal.libero_env import load_suite
        from refusal.nora_policy import NoraPolicy
        suite = load_suite(a.suite)
        tasks_live = [suite.get_task(i) for i in range(suite.n_tasks)]
        check_suite([t.language for t in tasks_live])
        tasks = [t.language for t in tasks_live][: a.tasks]
        policy = NoraPolicy(a.checkpoint, unnorm_key=a.suite, time_horizon=a.time_horizon)
        hold = build_hold_chunk(policy)
        hold_gen_ids = [int(i) for i in hold["gen_ids"]]
        if hold_gen_ids != [int(i) for i in ref["gen_ids"]]:
            raise SystemExit(f"hold chunk re-derived here {hold_gen_ids} != {a.hold_chunk_ref} {ref['gen_ids']}")
        print(f"[hold] re-derived gen ids match {a.hold_chunk_ref}", flush=True)
        (out / "hold_chunk.json").write_text(json.dumps(hold, indent=2))
        if a.stage in ("all", "replay"):
            stage_a(a, out, policy, suite, tasks_live[: a.tasks])
        if a.stage == "replay":
            print("DONE", out / "replay.jsonl")
            return
        replay = read_jsonl(out / "replay.jsonl")
        if not replay:
            raise SystemExit(f"{out / 'replay.jsonl'} is empty: stage A kept no policy call")

    pools = build_pools(a, tasks)
    print_pools(pools)
    episodes = read_jsonl(out / "episodes.jsonl")
    home = [{"image": r["home_frame"], "task_id": r["task_id"], "source": "home"} for r in episodes]
    if not home:
        raise SystemExit(f"{out / 'episodes.jsonl'} has no home frames")
    files = build_rows(a, pools, replay, home, hold_gen_ids)
    stats = row_table(files)
    n_twins = check_twins(Path(a.twins), files)
    n_labels = None
    if policy is not None:
        n_labels = check_labels(policy, files["install"] + files["control"], a.time_horizon)
    else:
        print("[labels] SKIPPED: no tokenizer on this machine (--stub-replay)", flush=True)

    pools_out = {k: v for k, v in pools.items() if k != "_lists"}
    (out / "pools.json").write_text(json.dumps(pools_out, indent=2))
    write_jsonl(out / "train_install.jsonl", files["install"])
    write_jsonl(out / "train_control.jsonl", files["control"])
    summary = {"exp": "f1_build_data", "ts": stamp, "out": str(out), "checkpoint": a.checkpoint,
               "suite": a.suite, "seed": a.seed, "smoke": bool(a.smoke), "stub_replay": int(a.stub_replay),
               "n_episodes": len(episodes), "n_success": sum(1 for r in episodes if r["success"]),
               "n_replay_calls": len(replay), "n_home_frames": len(home), "hold_gen_ids": hold_gen_ids,
               "n_distinct_labels": n_labels, "n_twins_held_out": n_twins, "rows": stats,
               "pool_counts": {k: len(v) for k, v in pools_out.items() if isinstance(v, list)},
               "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}}
    (out / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out / 'pools.json'}, {out / 'train_install.jsonl'} ({len(files['install'])} rows), "
          f"{out / 'train_control.jsonl'} ({len(files['control'])} rows), {out / 'build_summary.json'}")
    print("DONE", out)


if __name__ == "__main__":
    main()
