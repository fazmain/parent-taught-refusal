#!/usr/bin/env python3
"""Gate for the text-direction presence check. Exit non-zero if the parent's own harmfulness direction does not separate held-out D_text
(AUROC > 0.95 at some layer in [lo, hi] at t_inst) or if the second model has no rows.

  python3 experiments/1_presence/verify_text_direction_presence.py --parent parent --other nora --lo 8 --hi 28
  python3 experiments/1_presence/verify_text_direction_presence.py --parent smoke --other none --lo 0 --hi 30      # local smoke: only checks rows exist
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from refusal.paths import RESULTS_DIR  # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("--results", default=str(RESULTS_DIR / "results.jsonl"))
ap.add_argument("--parent", default="parent"); ap.add_argument("--other", default="nora")
ap.add_argument("--lo", type=int, default=8); ap.add_argument("--hi", type=int, default=28)
ap.add_argument("--threshold", type=float, default=0.95)
a = ap.parse_args()

rows = [json.loads(l) for l in Path(a.results).read_text().splitlines() if l.strip()]
rows = [r for r in rows if r.get("exp") == "poc_presence"]
own = [r for r in rows if r["model_acts"] == a.parent and r["direction_src"] == a.parent
       and r["metric"] == "auroc" and r["position"] == "t_inst" and a.lo <= r["layer"] <= a.hi]
assert own, f"no parent-own AUROC rows for {a.parent} in layers {a.lo}-{a.hi}"
best = max(own, key=lambda r: r["value"])
print(f"parent-own best t_inst AUROC = {best['value']:.3f} at layer {best['layer']} (n={best['n_pos']}+{best['n_neg']})")
if a.parent != "smoke":
    assert best["value"] > a.threshold, f"GATE FAIL: {best['value']:.3f} <= {a.threshold}"
if a.other != "none":
    xfer = [r for r in rows if r["model_acts"] == a.other and r["direction_src"] == a.parent and r["metric"] == "auroc"]
    assert xfer, f"no transfer rows {a.parent} -> {a.other}"
    bx = max((r for r in xfer if r["position"] == "t_inst"), key=lambda r: r["value"])
    print(f"transfer {a.parent}->{a.other} best t_inst AUROC = {bx['value']:.3f} at layer {bx['layer']}")
print("VERIFY POC: PASS")
