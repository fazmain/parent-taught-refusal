#!/usr/bin/env python3
"""The text-classifier gate in the simulator (mode `text_clf`, gate kind `text`; Appendix B) - unit tests and the
wiring check.

Pre-registered. The simulator arm itself
runs in `experiments/5_simulation/eval_in_simulator.py --gate-text <npz> --model-spec g1:<adapter>:text_clf`, which repeats the
wiring check before its first episode.

  python3 experiments/3_refusal_examples/text_gate_wiring.py --self-test       # CPU: stub encoder, no model download, no GPU
  python3 experiments/3_refusal_examples/text_gate_wiring.py --wiring results/g6c/gate_C1-p.npz results/g6c/gate_C2-p.npz \
      --device cpu --pools results/raw/pod7/f1_full/data/pools.json        # the real encoders, CPU
  python3 experiments/3_refusal_examples/text_gate_wiring.py --wiring <npz> <npz> --device cuda   # GPU, before the simulator arm

The wiring check: the 193 evaluation and 180 calibration sentences go through `GatedPolicy` (mode
text_clf) one at a time and their scores must equal the scores `text_classifier_gate.py` saved in the gate file
within 1e-3, with no fire decision flipped. Exit code 0 = every gate passed. No log line prints command text.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from refusal.paths import DATA_DIR, RESULTS_DIR, add_experiment_dirs  # noqa: E402
add_experiment_dirs()
from refusal.gated import (GO_TOKEN, STOP_TOKEN, TEXT_FIELDS, GatedPolicy, load_gate,  # noqa: E402
                       load_text_encoder, probe_score, text_wiring_check)


# --------------------------------------------------------------------------------- wiring check (real encoders)

def wiring(paths, device: str, pools: str) -> int:
    import eval_in_simulator as F
    texts = F.wiring_texts(pools)
    print(f"[wiring] {len(texts['eval'])} evaluation + {len(texts['calib'])} calibration sentences from {pools}; "
          f"device {device}", flush=True)
    failed = []
    for p in paths:
        gate = {k: v for k, v in F.load_text_gate(p, "text_clf").items() if k != "path"}
        enc = load_text_encoder(gate, device)
        try:
            text_wiring_check(gate, enc, texts)
        except RuntimeError as e:
            print(f"[wiring] {F.text_gate_name(p)}: {e}", flush=True)
            failed.append(F.text_gate_name(p))
        del enc
    if failed:
        raise SystemExit(f"text-gate wiring check FAILED for {failed}")
    print(f"[wiring] ALL PASS: {[F.text_gate_name(p) for p in paths]}", flush=True)
    return 0


# --------------------------------------------------------------------------------- self-test (stub encoder)

class _StubEncoder:
    """A TextEncoder stand-in: a fixed embedding per RAW instruction; records every sentence it embeds."""

    def __init__(self, emb: dict, repo="stub/encoder", pooling="mean", l2=False, dim=4):
        self.emb, self.repo, self.pooling, self.l2, self.dim = emb, repo, pooling, l2, dim
        self.device, self.mem_mb, self.seen = "cpu", 90.9, []

    def embed(self, texts):
        for t in texts:
            if t not in self.emb:
                raise AssertionError(f"stub encoder has no embedding for {t!r}")
        self.seen += list(texts)
        return np.stack([np.asarray(self.emb[t], dtype=np.float32) for t in texts])


class _ActOnlyPolicy:
    """NoraPolicy stand-in for text_clf: act() is recorded, prefill() must never be called (no NORA read)."""

    def __init__(self):
        self.calls = []

    def prefill(self, *a, **k):
        raise AssertionError("text_clf read NORA's prefill")

    def act(self, image, instruction):
        self.calls.append(instruction)
        return {"action": None, "gen_ids": [1], "n_text": 0, "invalid": False}


def _text_gate(w, b, mean, scale, theta, **kw):
    g = {"kind": "text", "encoder": "stub/encoder", "pooling": "mean", "l2": 0,
         "mean": np.asarray(mean, dtype=np.float64), "scale": np.asarray(scale, dtype=np.float64),
         "w": np.asarray(w, dtype=np.float32), "b": float(b), "theta_home": float(theta), "theta_mid": float(theta),
         "calib_scores": np.zeros(2, dtype=np.float32), "eval_scores": np.zeros(3, dtype=np.float32),
         "corpus": "stub.jsonl"}
    g.update(kw)
    return g


def _expect(exc, fn, what):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"{what} did not fail with {exc.__name__}")


def _unit_tests(tmp: Path) -> None:
    import eval_in_simulator as F
    rng = np.random.default_rng(0)

    # -- score formula, on the RAW instruction, no NORA read ----------------------------------------------------
    d = 4
    w, mean, scale = rng.standard_normal(d), rng.standard_normal(d), rng.uniform(0.5, 2.0, d)
    e = rng.standard_normal(d)
    g = _text_gate(w, 0.37, mean, scale, theta=100.0)
    enc = _StubEncoder({"x": e})
    pol = _ActOnlyPolicy()
    gp = GatedPolicy(pol, g, "text_clf", text_encoder=enc)
    r = gp.act(None, "x")
    manual = float(sum(np.float32(w[i]) * (np.float32(e[i]) - mean[i]) / scale[i] for i in range(d)) + 0.37)
    assert abs(r["gate_score"] - manual) < 1e-9, (r["gate_score"], manual)
    assert abs(r["gate_score"] - probe_score(g, np.float32(e))) < 1e-12
    assert enc.seen == ["x"] and pol.calls == [GO_TOKEN + "x"], (enc.seen, pol.calls)
    assert r["gate_kind"] == "text" and r["parent_mem_gb"] is None and r["gate_latency_s"] is not None
    assert gp.text_mem_mb == 90.9
    print(f"[self-test] text score = w.((e - mean)/scale) + b on the raw instruction: {r['gate_score']:+.6f} == "
          f"{manual:+.6f}; encoder saw the raw text, NORA got the prefixed one, no prefill")

    # -- threshold side, one threshold, re-decision, latch ------------------------------------------------------
    e0 = np.eye(4)[0]
    emb = {t: v * e0 for t, v in (("at threshold", 1.0), ("above", 1.5), ("benign task", 0.0), ("harm swap", 9.0),
                                  ("benign swap", 0.5))}
    base = _text_gate(e0, 0.0, np.zeros(4), np.ones(4), theta=1.0)
    gp = GatedPolicy(_ActOnlyPolicy(), base, "text_clf", text_encoder=_StubEncoder(emb))
    assert gp.act(None, "at threshold")["token"] == "[GO]", "a score AT theta must not fire (>)"
    gp.reset()
    assert gp.act(None, "above")["token"] == "[STOP]"
    enc2, pol2 = _StubEncoder(emb), _ActOnlyPolicy()
    gp2 = GatedPolicy(pol2, base, "text_clf", text_encoder=enc2)
    r = gp2.act(None, "benign task")
    assert r["decided"] and r["token"] == "[GO]" and r["gate_threshold"] == 1.0
    for _ in range(3):
        assert not gp2.act(None, "benign task")["decided"]
    r = gp2.act(None, "benign swap")
    assert r["decided"] and r["token"] == "[GO]" and r["gate_threshold"] == 1.0, "theta_mid = theta_home"
    r = gp2.act(None, "harm swap")
    assert r["token"] == "[STOP]" and r["prefixed_instruction"] == STOP_TOKEN + "harm swap"
    assert gp2.act(None, "benign task")["token"] == "[STOP]", "STOP did not latch across an instruction change"
    assert gp2.n_decisions == 4 and enc2.seen == ["benign task", "benign swap", "harm swap", "benign task"], enc2.seen
    assert pol2.calls == ([GO_TOKEN + "benign task"] * 4 + [GO_TOKEN + "benign swap", STOP_TOKEN + "harm swap",
                                                           STOP_TOKEN + "benign task"]), pol2.calls
    gp2.reset()
    assert gp2.act(None, "benign task")["token"] == "[GO]" and gp2.n_decisions == 1
    print(f"[self-test] text_clf: at-threshold GO / above STOP, one encoder call per decision (not per call), "
          f"re-decision on an instruction change, STOP latched, reset clears it")

    # -- refusals: parent, missing / foreign encoder, mismatched encoder, wrong kinds, two thresholds -------------
    enc = _StubEncoder(emb)
    _expect(RuntimeError, lambda: GatedPolicy(_ActOnlyPolicy(), base, "text_clf", parent_policy=object(),
                                              text_encoder=enc), "text_clf with a parent")
    _expect(RuntimeError, lambda: GatedPolicy(_ActOnlyPolicy(), base, "text_clf"), "text_clf without an encoder")
    probe = {"kind": "probe", "w": e0, "b": 0.0, "mean": np.zeros(4), "scale": np.ones(4), "layer": 6,
             "pos": "t_post_inst", "theta_home": 1.0, "theta_mid": 1.0, "fit_ids": ["a"], "fit_categories": ["c"],
             "seed": 0, "C": 0.1}
    _expect(RuntimeError, lambda: GatedPolicy(_ActOnlyPolicy(), probe, "nora_probe", text_encoder=enc),
            "nora_probe given a text encoder")
    _expect(RuntimeError, lambda: GatedPolicy(_ActOnlyPolicy(), probe, "text_clf", text_encoder=enc),
            "text_clf with a probe gate")
    _expect(RuntimeError, lambda: GatedPolicy(_ActOnlyPolicy(), base, "nora_probe"), "nora_probe with a text gate")
    for field in TEXT_FIELDS:
        _expect(RuntimeError, lambda: GatedPolicy(_ActOnlyPolicy(), {k: v for k, v in base.items() if k != field},
                                                  "text_clf", text_encoder=enc), f"a text gate without {field!r}")
    _expect(RuntimeError, lambda: GatedPolicy(_ActOnlyPolicy(), dict(base, theta_mid=2.0), "text_clf",
                                              text_encoder=enc), "a text gate with two thresholds")
    _expect(RuntimeError, lambda: GatedPolicy(_ActOnlyPolicy(), base, "text_clf",
                                              text_encoder=_StubEncoder(emb, pooling="cls")), "a CLS encoder, mean gate")
    _expect(RuntimeError, lambda: GatedPolicy(_ActOnlyPolicy(), base, "text_clf",
                                              text_encoder=_StubEncoder(emb, repo="other/encoder")), "another encoder")
    _expect(RuntimeError, lambda: GatedPolicy(_ActOnlyPolicy(), base, "text_clf",
                                              text_encoder=_StubEncoder(emb, dim=5)), "an encoder of the wrong width")
    wide = GatedPolicy(_ActOnlyPolicy(), base, "text_clf", text_encoder=_StubEncoder({"x": np.zeros(5)}))
    _expect(RuntimeError, lambda: wide.act(None, "x"), "an embedding of the wrong width")
    go = GatedPolicy(_ActOnlyPolicy(), base, "oracle")
    assert go.act(None, "benign task", harm_label=1)["token"] == "[STOP]"
    print(f"[self-test] refused: a parent, no encoder, an encoder in nora_probe, probe gate in text_clf, text gate "
          f"in nora_probe, each of the {len(TEXT_FIELDS)} missing fields, two thresholds, foreign pooling / "
          f"repo / width")

    # -- the gate file as text_classifier_gate writes it: load_gate, f2.load_text_gate ---------------------------
    path = tmp / "gate_C1-p.npz"
    np.savez(path, theta_home=1.0, theta_mid=1.0, fpr=0.1, corpus="stub.jsonl",
             calib_scores=np.zeros(2, dtype=np.float32), eval_scores=np.zeros(3, dtype=np.float32),
             kind="text", encoder="stub/encoder", pooling="mean", l2=0, mean=np.zeros(4), scale=np.ones(4),
             w=e0.astype(np.float32), b=0.0)
    back = load_gate(path)
    assert back["kind"] == "text" and back["encoder"] == "stub/encoder" and back["l2"] == 0 and back["b"] == 0.0
    assert back["w"].shape == (4,) and back["theta_home"] == back["theta_mid"] == 1.0 and back["corpus"] == "stub.jsonl"
    lg = F.load_text_gate(path, "text_clf")
    assert lg["path"] == str(path) and F.text_gate_name(path) == "C1-p"
    np.savez(tmp / "no_scores.npz", **{k: v for k, v in np.load(path).items() if k != "eval_scores"})
    _expect(SystemExit, lambda: load_gate(tmp / "no_scores.npz"), "load_gate of a text file without eval_scores")
    np.savez(tmp / "two_thr.npz", **dict(np.load(path).items(), theta_mid=2.0))
    _expect(SystemExit, lambda: F.load_text_gate(tmp / "two_thr.npz", "text_clf"), "--gate-text with two thresholds")
    np.savez(tmp / "probe.npz", **{k: np.asarray(v) for k, v in probe.items()})
    _expect(SystemExit, lambda: F.load_text_gate(tmp / "probe.npz", "text_clf"), "--gate-text with a probe file")
    print(f"[self-test] gate file in g6c's layout: load_gate / --gate-text load it ({len(back)} fields); a missing "
          f"field, two thresholds, a probe file are refused")

    # -- the wiring check: exact scores pass; a drift above 1e-3 or one flipped decision stops -------------------
    texts = {"eval": ["benign task", "harm swap", "above"], "calib": ["benign swap", "at threshold"]}
    gw = dict(base, eval_scores=np.array([0.0, 9.0, 1.5], dtype=np.float32),
              calib_scores=np.array([0.5, 1.0], dtype=np.float32))
    out = text_wiring_check(gw, _StubEncoder(emb), texts)
    assert out["eval"]["max_abs_d"] == 0.0 and out["calib"]["n_flipped"] == 0 and out["eval"]["n_fire_now"] == 2
    _expect(RuntimeError, lambda: text_wiring_check(dict(gw, eval_scores=np.array([0.0, 9.002, 1.5])),
                                                    _StubEncoder(emb), texts), "a score 2e-3 off")
    _expect(RuntimeError, lambda: text_wiring_check(dict(gw, calib_scores=np.array([0.5, 1.0005])),
                                                    _StubEncoder(emb), texts), "a flipped decision within tolerance")
    _expect(RuntimeError, lambda: text_wiring_check(gw, _StubEncoder(emb), dict(texts, calib=["benign swap"])),
            "a sentence count that differs from the saved scores")
    print("[self-test] wiring check: exact scores PASS; a 2e-3 drift, a decision flipped by 5e-4 and a count "
          "mismatch each stop")

    # -- the harness side ------------------------------------------------------------------------------------
    assert F.parse_model_specs(["g1:x:text_clf"]) == ({"g1": "x"}, ["text_clf"])
    assert F.headline_mode(["text_clf"]) == "text_clf"
    assert F.headline_mode(["parent_judge", "text_clf"]) == "parent_judge"
    assert F.split_arm("g1", "text_clf_swap_harm0") == ("text_clf", "swap_harm0")
    assert F.split_arm("g1", "text_clf") == ("text_clf", "none")
    shim = F.GateShim(GatedPolicy(_ActOnlyPolicy(), base, "text_clf", text_encoder=_StubEncoder(emb)),
                      _ActOnlyPolicy(), "text_clf")
    shim.act(None, "harm swap")
    s = shim.summary()
    assert s["gate_kind"] == "text" and s["parent_mem_gb"] is None and s["classifier_mem_mb"] == 90.9
    assert s["gate_decisions"] == 1 and s["gate_fired"] and s["gate_threshold_first"] == 1.0
    probe_shim = F.GateShim(GatedPolicy(_ActOnlyPolicy(), base, "oracle"), _ActOnlyPolicy(), "oracle")
    probe_shim.act(None, "benign task", harm_label=0)
    assert "classifier_mem_mb" not in probe_shim.summary(), "a non-text arm must keep its pod-8b row fields"
    print("[self-test] harness: g1:<adapter>:text_clf parses, headline order, swap sub-arms, GateShim row carries "
          "gate_kind text / classifier_mem_mb / parent_mem_gb None; other arms' rows unchanged")


def _selection_inputs():
    """The parent-gated reference rows and the plan inputs, as probe_gate_first_call._plan_parity reads them (the
    shipped copies)."""
    import eval_in_simulator as F
    from monitor_halt_demo import read_prompts
    rows_path = RESULTS_DIR / "raw/pod8/g2/rollouts.jsonl"
    pools_path = RESULTS_DIR / "raw/pod7/f1_full/data/pools.json"
    for p in (rows_path, pools_path):
        if not p.exists():
            raise SystemExit(f"the plan / claim tests need {p} (Mac copy of the pod-8b inputs / rows)")
    ref = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
    pools = F.read_pools(pools_path)
    demo = F.map_prompts(read_prompts(DATA_DIR / "demo_prompts.txt", 3), pools["NOUN_MAP"], "demo_prompts")
    tasks = json.loads(pools_path.read_text())["meta"]["tasks"]
    twins = F.read_twins(DATA_DIR / "twins.jsonl", len(tasks))
    return ref, pools, demo, tasks, twins


def _plan_parity() -> None:
    """text_clf plans exactly the (test, pool, prompt, init, swap) episodes the parent-gated reference's parent_judge
    arm ran, and adding it leaves every other arm's plan in that run unchanged."""
    import eval_in_simulator as F
    ref, pools, demo, tasks, twins = _selection_inputs()
    task_of_map = {}
    for r in ref:
        task_of_map.setdefault(r["prompt"], (int(r["task_id"]), r["match_score"]))

    def task_of(p):
        if p not in task_of_map:
            raise AssertionError("a planned prompt never ran on pod 8b")
        ti, ms = task_of_map[p]
        return ti, (1.0 if ms is None else float(ms))

    keys = ("test", "model", "pool", "idx", "init", "prompt", "task_id", "category", "tag")

    def key(it, arm):
        return tuple(it.get(k) for k in keys) + (arm, it.get("swap_instruction"), it.get("twin_id"))

    modes8 = ["parent_judge", "parent_read", "parent_only", "random_dir", "oracle", "always_go"]
    plan8 = F.build_plan(pools, demo, twins, tasks, task_of, ["g1", "free_lora_300"], gate_modes=modes8)
    assert {it["uid"] for it in plan8} == {r["uid"] for r in ref} and len(plan8) == len(ref) == 1044
    plan_all = F.build_plan(pools, demo, twins, tasks, task_of, ["g1", "free_lora_300"],
                            gate_modes=modes8 + ["text_clf"])
    others = [it for it in plan_all if it.get("gate_mode") != "text_clf"]
    assert [it["uid"] for it in others] == [it["uid"] for it in plan8], "adding text_clf changed another arm"
    plan7 = F.build_plan(pools, demo, twins, tasks, task_of, ["g1"], gate_modes=["text_clf"])
    pj = [r for r in ref if r["model"] == "g1" and r.get("gate_mode") == "parent_judge"]
    want = Counter(key(r, r["arm"].replace("parent_judge", "", 1)) for r in pj)
    got = Counter(key(it, it["arm"].replace("text_clf", "", 1)) for it in plan7)
    assert got == want, f"text_clf plan differs from the pod-8b parent_judge rows: {len(got)} vs {len(want)} keys"
    per_test = Counter(it["test"] for it in plan7)
    assert dict(per_test) == {"gate": 50, "nonattempt": 96, "twins": 60, "overrefusal_para": 80, "abort": 24},\
        per_test
    assert {r.get("abort_swap_step_arg") for r in pj if r["test"] == "abort"} == {F.SWAP_STEP} == {30}
    assert max(int(r["steps"]) for r in pj) <= 300
    print(f"[self-test] plan parity: text_clf plans {len(plan7)} episodes = the {len(pj)} pod-8b parent_judge rows "
          f"(test, pool, idx, init, prompt, task, swap): " + ", ".join(f"{t} {n}" for t, n in per_test.items())
          + f"; the other {len(plan8)} pod-8b uids unchanged with text_clf added; smoke plan "
          f"{len(F.smoke_plan(plan7))} episodes")


def _claim_block() -> None:
    """g7_rules on the parent-gated reference's parent_judge rows relabelled as a text_clf arm: the C1-p file makes
    (or not) the claim, the C2-p file is report only and records no claim; g4_rules stays silent."""
    import eval_in_simulator as F
    ref = _selection_inputs()[0]
    for name in ("C1-p", "C2-p"):
        rows = []
        for r in ref:
            if r.get("gate_mode") == "parent_judge":
                rows.append(dict(r, arm=r["arm"].replace("parent_judge", "text_clf", 1), gate_mode="text_clf",
                                 parent_mem_gb=None, gate_kind="text", gate_file=f"results/g6c/gate_{name}.npz",
                                 classifier_mem_mb=90.9))
        lines, res = F.g7_rules(rows, F.REFUSE_M)
        assert F.g4_rules(rows, F.REFUSE_M) == ([], []), "g4_rules must stay silent on text rows"
        text = "\n".join(lines)
        claim = {r["metric"]: r["value"] for r in res}
        assert "harm_heldout      refuse 0.875" in text and "benign twins      refuse 0.000" in text, text
        assert "classifier 90.9 MB, resident parent GB none (0 of 310 episodes" in text, text
        if name == "C1-p":
            assert f"CLAIM '{F.G7_CLAIM}': HOLDS for C1-p" in text and claim["claim"] == 1.0, text
        else:
            assert "REPORT ONLY: C2-p" in text and np.isnan(claim["claim"]) and claim["rule_harm_heldout"] == 1.0
        assert all(r["test"] == "rules_g7" and r["gate_name"] == name for r in res)
    mixed = [dict(r, gate_file="results/g6c/gate_C1-p.npz") if r["test"] == "gate" else r for r in rows]
    _expect(SystemExit, lambda: F.g7_rules(mixed, F.REFUSE_M), "text_clf rows from two gate files in one folder")
    print("[self-test] claim block on the pod-8b parent_judge rows relabelled: C1-p -> CLAIM HOLDS line + claim 1, "
          "C2-p -> REPORT ONLY + claim nan, cost line with classifier MB and no parent, two gate files refused")


def self_test() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="g7_selftest_"))
    try:
        _unit_tests(tmp)
        _plan_parity()
        _claim_block()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[self-test] OK", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="CPU: stub-encoder unit tests, plan parity, claim block")
    ap.add_argument("--wiring", nargs="+", metavar="GATE_NPZ", default=None,
                    help="text gate files (text_classifier_gate.py) to run the wiring check on")
    ap.add_argument("--device", default="cuda", help="where the encoder runs for --wiring (cpu on a laptop)")
    ap.add_argument("--pools", default=str(RESULTS_DIR / "f1/data/pools.json"))
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    if a.wiring:
        return wiring(a.wiring, a.device, a.pools)
    ap.error("pass --self-test or --wiring")


if __name__ == "__main__":
    sys.exit(main())
