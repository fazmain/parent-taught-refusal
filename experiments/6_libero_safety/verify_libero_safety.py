#!/usr/bin/env python3
"""Gate for the LIBERO-Safety evaluation (the pipeline gate). Asserts, does not report:

  * every arm folder <root>/<arm> has 15 scenes x 2 variants (harmful, twin) x N episodes of episode rows, i.e.
    30 prompts x 3 arms x N in total, and no duplicate uid;
  * every row has its MP4 and its .gen.jsonl sidecar on disk;
  * contamination is one of train / eval / clean and is the value declared for that scene;
  * verdict_first is [STOP] or [GO] on every verdict-arm row and None on every base row;
  * harm_label_source is the scene's: `goal` for the three harmful-outcome goal files, else the kind in
    data/libero_safety/hazard_map.json; twin rows carry harm_label_informational = true, harmful rows false;
  * the harmful episode and its twin start from the same init state (same sha1), in every arm.

  python3 experiments/6_libero_safety/verify_libero_safety.py --root results/ls1_smoke --episodes 1
  python3 experiments/6_libero_safety/verify_libero_safety.py --root results/ls1 --episodes 10
"""
from __future__ import annotations

import argparse, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from refusal.paths import repo_path, add_experiment_dirs  # noqa: E402
add_experiment_dirs()
from libero_safety_eval import (ARMS, BENCHMARK, CONTAMINATION_VALUES, HAZARD_MAP, LEVELS,  # noqa: E402
                                      TWINS_TSV, VARIANTS, VERDICT_GO, VERDICT_MODEL, VERDICT_STOP,
                                      contamination, harm_label_source, read_twins)


def _path(p):
    p = Path(str(p))
    return p if p.is_absolute() else repo_path(p)


def check(root, episodes, tf_hazard=None, twins_tsv=TWINS_TSV):
    root = Path(root)
    twins = read_twins(twins_tsv)
    if tf_hazard is None:
        hazard = {k: v for k, v in json.loads(Path(HAZARD_MAP).read_text()).items() if not k.startswith("_")}
        tf_hazard = lambda tf: harm_label_source(tf, hazard)  # noqa: E731
    levels = Counter(t["level"] for t in twins.values())
    assert levels == {lvl: 5 for lvl in LEVELS}, f"twin TSV levels {dict(levels)}"
    want = {(tf, v, ep) for tf in twins for v in VARIANTS for ep in range(episodes)}
    sha = {}
    n_total = 0
    for arm, spec in ARMS.items():
        path = root / arm / "rollouts.jsonl"
        assert path.exists(), f"{path} missing"
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        rows = [r for r in rows if r.get("stage") == "arms" and r.get("uid")]
        uids = Counter(r["uid"] for r in rows)
        dup = [u for u, n in uids.items() if n > 1]
        assert not dup, f"{arm}: {len(dup)} duplicate uids, e.g. {dup[0]}"
        rows = [r for r in rows if r["episode"] < episodes]
        have = {(r["task_file"], r["variant"], r["episode"]) for r in rows}
        assert have == want, (f"{arm}: {len(have)} episode keys, want {len(want)}; missing "
                              f"{sorted(want - have)[:3]}, unexpected {sorted(have - want)[:3]}")
        for r in rows:
            tag = f"{arm} {r['uid']}"
            assert r["arm"] == arm and r["benchmark"] == BENCHMARK, f"{tag}: arm {r['arm']} benchmark {r['benchmark']}"
            assert r["level"] == twins[r["task_file"]]["level"], f"{tag}: level {r['level']}"
            vid = _path(r["video"])
            assert vid.is_file(), f"{tag}: video {vid} missing"
            assert vid.with_suffix(".gen.jsonl").is_file(), f"{tag}: sidecar {vid.with_suffix('.gen.jsonl')} missing"
            assert r["contamination"] in CONTAMINATION_VALUES, f"{tag}: contamination {r['contamination']!r}"
            assert r["contamination"] == contamination(r["task_file"]), f"{tag}: contamination {r['contamination']}"
            if spec["model"] == VERDICT_MODEL:
                assert r["verdict_first"] in (VERDICT_STOP, VERDICT_GO), f"{tag}: verdict_first {r['verdict_first']!r}"
            else:
                assert r["verdict_first"] is None, f"{tag}: base row has verdict_first {r['verdict_first']!r}"
            assert r["harm_label_source"] == tf_hazard(r["task_file"]),\
                f"{tag}: harm_label_source {r['harm_label_source']} != {tf_hazard(r['task_file'])}"
            assert r["harm_label_informational"] == (r["variant"] == "twin"), f"{tag}: harm_label_informational"
            assert r["init_source"] == "seed" and r["seed"] == r["episode"], f"{tag}: init {r['init_source']} {r['seed']}"
            assert r["max_steps"] == twins[r["task_file"]]["max_steps"], f"{tag}: max_steps {r['max_steps']}"
            sha.setdefault((r["task_file"], r["episode"]), set()).add(r["init_state_sha1"])
        n_total += len(rows)
    split = {k: v for k, v in sha.items() if len(v) != 1}
    assert not split, f"{len(split)} (scene, episode) pairs start from different init states, e.g. {next(iter(split))}"
    assert n_total == len(ARMS) * len(want), f"{n_total} rows, want {len(ARMS) * len(want)}"
    return n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder holding one sub-folder per arm")
    ap.add_argument("--episodes", type=int, required=True)
    a = ap.parse_args()
    n = check(a.root, a.episodes)
    print(f"VERIFY LS1: PASS ({n} episode rows = 30 prompts x {len(ARMS)} arms x {a.episodes} episodes)")


if __name__ == "__main__":
    main()
