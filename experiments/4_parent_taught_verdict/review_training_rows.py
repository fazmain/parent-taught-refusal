#!/usr/bin/env python3
"""Review of the round-1 training rows (code name G14): the independent check run before any training (every training
set had two independent reviews).

Reads the four row files written by `build_training_rows.py --stage rows` and checks, independently of the builder's own
assertions:

  1  counts per (row_type, class, frame_source, gripper, label_source) and per arm;
  2  label correctness: every harm row's action ids equal the hold chunk for its gripper; no execute row carries a hold
     chunk; the verdict arms start with the [STOP] ids exactly on the hold rows and [GO] elsewhere; action ids after
     the verdict are FAST action tokens (+ the closing <|im_end|>);
  3  the arms differ only where the design allows (I0 == I3 byte for byte; I1 = I0 + verdict prefix; I2 differs from
     I1 only on corpus labels where the parent disagrees with the construction label);
  4  leakage: exact and normalised text overlap with every evaluation / calibration / twin / G11 / G13 sentence, and
     near duplicates (word-set Jaccard >= 0.8) against the rule pools (HARM_HELDOUT, TWIN_*, G11, G13);
  5  shortcut risk: sentence length and frame kind by class, class balance per nearest task, the most class-predictive
     words, and how far a bag-of-words logistic regression fitted on the corpus rows alone gets on the stage-0 rule
     pools and G11 (if a word counter already passes, the data invites a word-level install).

  python3 experiments/4_parent_taught_verdict/review_training_rows.py --rows <dir with train_*.jsonl> --hold-chunk results/raw/pod6/c1/hold_chunk.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from refusal.paths import DATA_DIR, RESULTS_DIR, add_experiment_dirs  # noqa: E402
add_experiment_dirs()

ARM_FILES = {"I0": "train_free_big.jsonl", "I1": "train_verdict.jsonl", "I2": "train_verdict_parent.jsonl",
             "I3": "train_free_align.jsonl"}
STOP_IDS, GO_IDS = [58, 50669, 60], [58, 15513, 60]
ACTION_MIN, ACTION_MAX, IM_END = 151665, 153712, 151645     # third_party/nora/nora_utils.py; refusal.nora_policy used when importable


def norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", s.lower()))


def words(s: str) -> set:
    return set(norm(s).split())


def _endings() -> list[str]:
    from build_parent_gated_data import BENIGN_CLAUSES
    bank = json.loads((DATA_DIR / "clause_bank.json").read_text())
    ends = [c for c in BENIGN_CLAUSES if c] + [c for k in ("benign", "harm", "benign_stove", "harm_stove") for c in bank[k]]
    return sorted({norm(e) for e in ends}, key=len, reverse=True)


def core(s: str, ends: list[str]) -> str:
    """The sentence without a leading g1 prefix and without a trailing benign clause / clause-bank ending."""
    from build_parent_gated_data import PREFIXES
    t = norm(s)
    for pre in sorted({norm(x) for x in PREFIXES}, key=len, reverse=True):
        if t.startswith(pre + " "):
            t = t[len(pre) + 1:]
            break
    for e in ends:
        if t.endswith(" " + e):
            t = t[: -len(e) - 1]
            break
    return t


def read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--hold-chunk", default=str(RESULTS_DIR / "raw/pod6/c1/hold_chunk.json"))
    ap.add_argument("--hold-closed", default=None, help="json with gen_ids of the closed-gripper hold, if any")
    ap.add_argument("--pools", default=str(RESULTS_DIR / "raw/pod7/f1_full/data/pools.json"))
    a = ap.parse_args()
    try:
        from refusal.nora_policy import ACTION_MAX as AM, ACTION_MIN as Am
        amin, amax = int(Am), int(AM)
    except Exception:
        amin, amax = ACTION_MIN, ACTION_MAX
    rows_dir = Path(a.rows)
    R = {arm: read(rows_dir / f) for arm, f in ARM_FILES.items()}
    hold_open = [int(i) for i in json.loads(Path(a.hold_chunk).read_text())["gen_ids"]]
    hold_closed = [int(i) for i in json.loads(Path(a.hold_closed).read_text())["gen_ids"]] if a.hold_closed else None
    issues: list[str] = []

    def issue(s: str) -> None:
        issues.append(s)
        print("ISSUE", s)

    # 1 counts
    for arm, rows in R.items():
        c = Counter((r["row_type"], "hold" if r.get("harm") else "exec", r.get("frame_source"), r.get("gripper"),
                     r.get("label_source")) for r in rows)
        print(f"\n[{arm}] {len(rows)} rows")
        for k, v in sorted(c.items(), key=lambda kv: str(kv[0])):
            print(f"  {str(k):<90s} {v:6d}")

    # 2 labels
    for arm, rows in R.items():
        verdict = arm in ("I1", "I2")
        bad = Counter()
        for r in rows:
            ids = [int(i) for i in r["label_ids"]]
            if verdict:
                pre = STOP_IDS if r.get("harm") else GO_IDS
                if ids[:3] != pre:
                    bad["verdict prefix wrong"] += 1
                ids = ids[3:]
            if r.get("harm"):
                want = hold_open if r.get("gripper") == "open" else hold_closed
                if want is None or ids != want:
                    bad[f"hold row not the {r.get('gripper')} hold chunk"] += 1
            else:
                if ids == hold_open or (hold_closed and ids == hold_closed):
                    bad[f"execute row carries a hold chunk ({r['row_type']}, {r.get('label_source')})"] += 1
            body = ids[:-1] if ids and ids[-1] == IM_END else ids
            if not body or any(not (amin <= i <= amax) for i in body):
                bad["non-action id in the action part"] += 1
            if not ids or ids[-1] != IM_END:
                bad["label does not end with <|im_end|>"] += 1
        for k, v in bad.items():
            issue(f"{arm}: {k}: {v} rows")
        print(f"[{arm}] label checks: {sum(bad.values())} problems")

    # 3 arm differences
    strip = lambda r: {k: v for k, v in r.items()}  # noqa: E731
    if [json.dumps(strip(r), sort_keys=True) for r in R["I0"]] != [json.dumps(strip(r), sort_keys=True) for r in R["I3"]]:
        issue("I0 and I3 differ")
    n_diff = 0
    for r0, r1 in zip(R["I0"], R["I1"]):
        pre = STOP_IDS if r0.get("harm") else GO_IDS
        if [int(i) for i in r1["label_ids"]] != pre + [int(i) for i in r0["label_ids"]] or r0["instruction"] != r1["instruction"]:
            n_diff += 1
    if len(R["I0"]) != len(R["I1"]) or n_diff:
        issue(f"I1 is not I0 + verdict prefix ({n_diff} rows differ, {len(R['I0'])} vs {len(R['I1'])} rows)")
    i1_home = {r["corpus_id"]: r for r in R["I1"] if r["row_type"] == "corpus_home"}
    i2_home = {r["corpus_id"]: r for r in R["I2"] if r["row_type"] == "corpus_home"}
    flips = sum(1 for k in i1_home if k in i2_home and i1_home[k]["harm"] != i2_home[k]["harm"])
    print(f"\n[arms] I0 == I3: {not any('I0 and I3' in s for s in issues)}; I1 = I0 + prefix on {len(R['I0']) - n_diff} rows; "
          f"I2 vs I1: {flips} corpus home labels flipped (parent vs construction); corpus ids only in I1 "
          f"{len(set(i1_home) - set(i2_home))}, only in I2 {len(set(i2_home) - set(i1_home))}")
    ts = Counter((i1_home[k]["harm"], i2_home[k]["harm"]) for k in i1_home if k in i2_home)
    print(f"[arms] (construction, parent) home labels: {dict(ts)}")

    # 4 leakage
    from verify_parent_gated_judge import calib_sentences, eval_sentences
    pools = json.loads(Path(a.pools).read_text())
    tasks = {norm(t) for t in pools["meta"]["tasks"]}
    ev = eval_sentences(pools, DATA_DIR / "twins.jsonl", DATA_DIR / "demo_prompts.txt")
    cal = calib_sentences(pools, DATA_DIR / "novel_benign.txt")
    g11 = read(DATA_DIR / "confirm_g11.jsonl")
    g13 = read(DATA_DIR / "confirm_g13_benign_blind.jsonl")
    tw = read(DATA_DIR / "twins.jsonl")
    tests = [(e["pool"], e["text"]) for e in ev] + [("G11/" + r["pool"], r["text"]) for r in g11] +\
            [("G13/" + r["pool"], r["text"]) for r in g13] + [(f"twins.{k}", t[k]) for t in tw for k in
                                                              ("safe", "benign", "harm") if k in t]
    calset = {norm(c["text"]) for c in cal}
    train_text = {r["instruction"] for arm in R for r in R[arm]}
    exact = Counter()
    for pool, t in tests:
        if norm(t) in {norm(x) for x in train_text} and norm(t) not in tasks:
            exact[pool] += 1
    for pool, n in exact.items():
        issue(f"training text equals a test sentence: {pool} x{n}")
    n_cal = sum(1 for t in train_text if norm(t) in calset and norm(t) not in tasks)
    print(f"\n[leak] exact test overlaps (trained instructions exempt): {dict(exact) or 0}; training texts equal to a "
          f"calibration sentence: {n_cal} (allowed: G14 has no threshold)")
    rule_pools = [(p, t) for p, t in tests if any(k in p for k in ("HARM_HELDOUT", "TWIN", "G11", "G13", "twins"))]
    tw_sets = [(p, t, words(t)) for p, t in rule_pools]
    ends = _endings()
    test_cores = defaultdict(set)
    for pool, t in tests:
        c = core(t, ends)
        if norm(t) not in tasks and c not in tasks:        # a core that IS a trained instruction is allowed
            test_cores[c].add(pool)
    core_hits = Counter()
    for t in train_text:
        for pool in test_cores.get(core(t, ends), ()):
            if norm(t) not in tasks:
                core_hits[pool] += 1
    for pool, n in core_hits.items():
        issue(f"a training sentence has the same prefix/clause-stripped core as a test sentence: {pool} x{n}")
    print(f"[leak] core overlaps (prefix and clause endings stripped, trained instructions exempt): "
          f"{dict(core_hits) or 0}")
    near = Counter()
    corpus_texts = sorted({r["instruction"] for r in R["I0"] if r["row_type"].startswith("corpus")})
    for t in corpus_texts:
        w = words(t)
        for p, u, wu in tw_sets:
            j = len(w & wu) / max(1, len(w | wu))
            if j >= 0.8 and norm(t) != norm(u):
                near[p] += 1
    print(f"[leak] corpus sentences with word-set Jaccard >= 0.8 to a rule-pool sentence: {dict(near) or 0}")

    # 5 shortcut risk
    corp = [r for r in R["I0"] if r["row_type"] in ("corpus_home",)]
    y = np.array([int(r["harm"]) for r in corp])
    lens = np.array([len(norm(r["instruction"]).split()) for r in corp])
    print(f"\n[shortcut] corpus home rows: harm {int(y.sum())} / benign {int((y == 0).sum())}; words per sentence "
          f"harm {lens[y == 1].mean():.1f} ± {lens[y == 1].std():.1f}, benign {lens[y == 0].mean():.1f} ± "
          f"{lens[y == 0].std():.1f}")
    frames = Counter((r["row_type"], int(r.get("harm", 0))) for r in R["I0"] if r["row_type"].startswith("corpus"))
    print(f"[shortcut] corpus rows by (frame kind, harm): {dict(frames)}")
    by_task = defaultdict(Counter)
    for r in corp:
        by_task[int(r.get("task_id", -1))][int(r["harm"])] += 1
    print("[shortcut] harm fraction per nearest task: " + ", ".join(
        f"t{t}: {c[1] / max(1, c[0] + c[1]):.2f} (n={c[0] + c[1]})" for t, c in sorted(by_task.items())))
    from build_parent_gated_data import BENIGN_CLAUSES, PREFIXES
    pre_n = {norm(x) for x in PREFIXES}
    cl_n = [norm(c) for c in BENIGN_CLAUSES if c]
    has_pre = np.array([any(norm(r["instruction"]).startswith(p + " ") for p in pre_n) for r in corp])
    has_cl = np.array([any(norm(r["instruction"]).endswith(" " + c) for c in cl_n) for r in corp])
    for name, m in (("prefix", has_pre), ("benign clause", has_cl)):
        print(f"[shortcut] hold rate with {name} {y[m].mean() if m.any() else float('nan'):.2f} (n={int(m.sum())}) vs "
              f"without {y[~m].mean() if (~m).any() else float('nan'):.2f} (n={int((~m).sum())})")
    first = defaultdict(list)
    for r, yy in zip(corp, y):
        t = core(r["instruction"], []).split()
        if t:
            first[t[0]].append(int(yy))
    pure = [(w, len(v), float(np.mean(v))) for w, v in first.items() if len(v) >= 20 and np.mean(v) in (0.0, 1.0)]
    print(f"[shortcut] first words (after any prefix) with >= 20 rows and a pure class: {len(pure)} — "
          + ", ".join(f"{w} {n}x->{'hold' if m else 'exec'}" for w, n, m in sorted(pure, key=lambda x: -x[1])[:20]))
    lb = np.digitize(lens, [6, 9, 12, 16])
    print("[shortcut] hold rate by length bucket (<6, 6-8, 9-11, 12-15, >=16 words): "
          + ", ".join(f"{y[lb == b].mean():.2f} (n={int((lb == b).sum())})" for b in range(5) if (lb == b).any()))
    allr = R["I0"]
    # length / decoration checks exclude rows whose text IS a trained instruction: those 10 strings are the tasks the
    # policy must always execute, so they carry no length cue a model could transfer to new commands
    nt = [r for r in allr if norm(r["instruction"]) not in tasks]
    ya = np.array([int(r.get("harm", 0)) for r in nt])
    la = np.array([len(norm(r["instruction"]).split()) for r in nt])
    ba = np.digitize(la, [7, 10, 13, 16])
    yall = np.array([int(r.get("harm", 0)) for r in allr])
    ball = np.digitize(np.array([len(norm(r["instruction"]).split()) for r in allr]), [7, 10, 13, 16])
    print("[shortcut] all rows of I0 incl. trained instructions (reported, no bound), hold rate by length bucket: " +
          ", ".join(f"{yall[ball == b].mean():.2f} (n={int((ball == b).sum())})" for b in range(5) if (ball == b).any()))
    print("[shortcut] ALL non-trained-instruction rows of I0, hold rate by length bucket (<=6, 7-9, 10-12, 13-15, >=16 words): " + ", ".join(
        f"{ya[ba == b].mean():.2f} (n={int((ba == b).sum())})" for b in range(5) if (ba == b).any()))
    for b in range(5):
        m = ba == b
        if m.sum() >= 50 and not 0.15 <= ya[m].mean() <= 0.65:
            issue(f"non-trained-instruction-row length bucket {b} has P(hold) {ya[m].mean():.2f} over {int(m.sum())} rows (outside 0.15-0.65)")
    pa = np.array([any(norm(r["instruction"]).startswith(p + " ") for p in pre_n) for r in nt])
    ca = np.array([any(norm(r["instruction"]).endswith(" " + c) for c in cl_n) for r in nt])
    for name, m in (("prefix", pa), ("benign clause", ca)):
        if m.any() and (~m).any():
            gap = abs(ya[m].mean() - ya[~m].mean())
            print(f"[shortcut] non-trained-instruction rows: hold rate with {name} {ya[m].mean():.2f} (n={int(m.sum())}) vs without "
                  f"{ya[~m].mean():.2f} (n={int((~m).sum())})")
            if gap > 0.10:
                issue(f"non-trained-instruction-row hold-rate gap for {name}: {gap:.2f} (> 0.10)")
    wc = {0: Counter(), 1: Counter()}
    for r in corp:
        for w in set(norm(r["instruction"]).split()):
            wc[int(r["harm"])][w] += 1
    n1, n0 = max(1, int(y.sum())), max(1, int((y == 0).sum()))
    score = {w: (wc[1][w] + 1) / (n1 + 2) / ((wc[0][w] + 1) / (n0 + 2)) for w in set(wc[0]) | set(wc[1])
             if wc[0][w] + wc[1][w] >= 20}
    top_h = sorted(score, key=lambda w: -score[w])[:15]
    top_b = sorted(score, key=lambda w: score[w])[:15]
    print("[shortcut] most harm-predictive words (ratio): " + ", ".join(f"{w} {score[w]:.1f}" for w in top_h))
    print("[shortcut] most benign-predictive words (ratio): " + ", ".join(f"{w} {1 / score[w]:.1f}" for w in top_b))
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = CountVectorizer(ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform([r["instruction"] for r in corp])
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=0).fit(X, y)
    cal_s = clf.decision_function(vec.transform([c["text"] for c in cal]))
    theta = float(np.quantile(cal_s, 0.9))

    def fire(texts):
        return float(np.mean(clf.decision_function(vec.transform(texts)) > theta)) if texts else float("nan")
    bow = {p: fire([e["text"] for e in ev if e["pool"] == p]) for p in ("HARM_HELDOUT", "TWIN_BENIGN", "TWIN_HARM", "TRAINED", "LEXICAL")}
    bow.update({f"G11/{p}": fire([r["text"] for r in g11 if r["pool"] == p]) for p in ("HARM_NEW", "TWIN_BENIGN_NEW")})
    bow.update({f"G13/{p}": fire([r["text"] for r in g13 if r["pool"] == p]) for p in ("BENIGN_BLIND", "LEXICAL_BLIND")})
    print("[shortcut] bag-of-words (1-2 grams) logistic fitted on the corpus home rows only, theta at 10% calibration "
          "FA: " + ", ".join(f"{k} {v:.2f}" for k, v in bow.items()))
    # ---- audit checklist items 4, 7, 8, 9, 10, 11, 13 of the training-data audit ----------------------------------
    print("\n[audit] checklist items 4, 7, 8, 9, 10, 11, 13")
    manner_w = ("gently", "slowly", "carefully", "a little", "slightly", "lightly")

    def mcore(t: str) -> str:
        c = " " + core(t, ends) + " "
        for w in manner_w:
            c = c.replace(" " + w + " ", " ")
        return " ".join(c.split())
    # 4 duplicate pairs / cores with two labels
    for arm, rows in R.items():
        pair, cor = defaultdict(set), defaultdict(set)
        for r in rows:
            pair[(r["image"], r["instruction"])].add(int(r.get("harm", 0)))
            if r["row_type"].startswith("corpus"):
                cor[mcore(r["instruction"])].add(int(r.get("harm", 0)))
        n2p = sum(1 for v in pair.values() if len(v) > 1)
        n2c = sum(1 for v in cor.values() if len(v) > 1)
        if n2p:
            issue(f"{arm}: {n2p} (frame, sentence) pairs carry both labels")
        if n2c:
            issue(f"{arm}: {n2c} corpus sentences that differ only by a manner word carry opposite labels")
        print(f"[audit 4] {arm}: pairs with two labels {n2p}; corpus manner-stripped cores with two labels {n2c}")
    # 6b manner-stripped core leakage
    tm = defaultdict(set)
    for pool, t in tests:
        c = mcore(t)
        if norm(t) not in tasks and c not in tasks:
            tm[c].add(pool)
    mh = Counter(pl for t in train_text if norm(t) not in tasks for pl in tm.get(mcore(t), ()))
    for pool, n in mh.items():
        issue(f"manner-stripped core overlap with {pool} x{n}")
    print(f"[audit 6] manner-stripped core overlaps: {dict(mh) or 0}")
    # 7 near duplicates vs lexical / blind pools: distinctive words for LEXICAL pools (their sentences embed trained
    #   tasks verbatim, so plain word overlap measures the task, not the idiom), plain words (>= 0.8) for BLIND pools
    stop7 = set("the a an to of on in and it at with by from into onto for so then is be".split()) | set(
        "bowl black plate white cream cheese box wine bottle stove burner cabinet drawer top middle rack table person "
        "next standing put place move open turn set push pick up down front inside".split())
    lh = Counter()
    for t in train_text:
        if norm(t) in tasks:
            continue
        w, wd = words(t), words(t) - stop7
        for p, u in tests:
            if "LEXICAL" in p:
                ud = words(u) - stop7
                if len(wd & ud) >= 2 and len(wd & ud) / max(1, len(wd | ud)) >= 0.5:
                    lh[p] += 1
            elif "BLIND" in p:
                wu = words(u)
                if len(w & wu) / max(1, len(w | wu)) >= 0.8:
                    lh[p] += 1
    for pool, n in lh.items():
        issue(f"training text near-copies a {pool} sentence x{n}")
    print(f"[audit 7] near copies (LEXICAL pools: >= 2 shared distinctive words, Jaccard >= 0.5; BLIND pools: word "
          f"Jaccard >= 0.8): {dict(lh) or 0}")
    # 8 provenance: content-word overlap of hard negatives with any test sentence (>= 3 shared content words)
    stop = set("the a an to of on in and it at with by from into onto for so then is be".split()) | set(
        "bowl black plate white cream cheese box wine bottle stove burner cabinet drawer top middle rack table person "
        "next standing put place move open turn set push pick up down".split())
    hn = sorted({r["instruction"] for r in R["I0"] if "hard" in str(r.get("source", "")) or "hn" in str(r.get("source", "")).lower()})
    prov = 0
    for t in hn:
        w = words(t) - stop
        for p, u in tests:
            wu = words(u) - stop
            if len(w & wu) >= 3 and len(w & wu) / max(1, len(w | wu)) >= 0.6:
                prov += 1
                print(f"[audit 8] hard negative shares its distinctive words with {p}: {sorted(w & wu)}")
                break
    print(f"[audit 8] hard negatives checked: {len(hn)}; sharing >= 3 distinctive words (Jaccard >= 0.6) with a test "
          f"sentence: {prov}")
    if prov:
        issue(f"{prov} hard negatives share their distinctive words with a test sentence")
    # 9 held-out categories anywhere in training
    force = re.compile(r"\b(crush\w*|break\w*|broke\w*|smash\w*|shatter\w*|crack\w*|slam\w*|press\w*|squeez\w*|jam\w*)\b"
                       r"|as hard as|\bhard\b|\bforce\b|full force|all your strength")
    eff = re.compile(r"\b(gripper|arm)\b|your own|yourself")
    nf = sum(1 for r in R["I0"] if r.get("harm") and force.search(norm(r["instruction"])))
    ne = sum(1 for r in R["I0"] if eff.search(norm(r["instruction"])))
    if nf:
        issue(f"{nf} harm rows carry excessive-force cues")
    if ne:
        issue(f"{ne} rows carry effector words")
    print(f"[audit 9] harm rows with force cues {nf}; rows with effector words {ne}")
    # 10 generic confound scan over all rows of I0
    def feats(r):
        t = norm(r["instruction"]).split()
        f = {k: str(r.get(k)) for k in ("frame_source", "gripper", "task_id")}   # what the model can see besides text
        f.update({"len": str(min(len(t) // 3, 6)), "first": t[0] if t else "", "last": t[-1] if t else "",
                  "first2": " ".join(t[:2]), "last2": " ".join(t[-2:]), "image": r.get("image", "")})
        return f
    ftab = defaultdict(lambda: defaultdict(list))
    vis = [r for r in allr if norm(r["instruction"]) not in tasks]    # trained-instruction rows are execute by design
    for r in vis:
        for k, v in feats(r).items():
            ftab[k][v].append(int(r.get("harm", 0)))
    flagged = []
    design = set()
    for k, vals in ftab.items():
        for v, ys in vals.items():
            if len(ys) >= 50 and not 0.15 <= np.mean(ys) <= 0.85 and (k, v) not in design:
                flagged.append((k, v, len(ys), float(np.mean(ys))))
    print(f"[audit 10] model-visible feature values (rows whose text is not a trained instruction: {len(vis)}) with n >= 50 "
          f"and hold rate outside 0.15-0.85: {len(flagged)}")
    for k, v, n, m in sorted(flagged, key=lambda x: -x[2])[:30]:
        print(f"[audit 10]   {k}={v[:40]!r:44s} n={n:5d} hold {m:.2f}")
    # 11 non-meaning model (all rows and corpus rows): cross-validated AUROC
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.model_selection import cross_val_score
    def nm(r):
        f = feats(r)
        t = norm(r["instruction"])
        return {"frame_source": f["frame_source"], "gripper": f["gripper"], "task_id": f["task_id"], "len": f["len"],
                "prefix": str(any(t.startswith(x + " ") for x in pre_n)), "clause": str(any(t.endswith(" " + c) for c in cl_n)),
                "manner": str(any(" " + w + " " in " " + t + " " for w in manner_w))}
    for name, rows in (("non-trained-instruction rows", vis), ("corpus rows", [r for r in allr if r["row_type"].startswith("corpus")])):
        Xn = DictVectorizer().fit_transform([nm(r) for r in rows])
        yn = np.array([int(r.get("harm", 0)) for r in rows])
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        au = float(np.mean(cross_val_score(LogisticRegression(max_iter=2000), Xn, yn, cv=cv, scoring="roc_auc")))
        print(f"[audit 11] non-meaning visible features -> label, 5-fold AUROC on {name}: {au:.3f}")
        if au > 0.65:
            issue(f"non-meaning features predict the corpus label at AUROC {au:.3f} (> 0.65)")
    # 13 images: all-one-class images with >= 5 rows beyond chance
    by_img = defaultdict(list)
    for r in allr:
        if r["row_type"].startswith("corpus"):
            by_img[r["image"]].append(int(r.get("harm", 0)))
    p1 = float(np.mean([y for v in by_img.values() for y in v])) if by_img else 0.0
    big = [v for v in by_img.values() if len(v) >= 5]
    pure = sum(1 for v in big if len(set(v)) == 1)
    exp_pure = sum(p1 ** len(v) + (1 - p1) ** len(v) for v in big)
    print(f"[audit 13] corpus images with >= 5 rows: {len(big)}; single-class {pure} (expected by chance {exp_pure:.1f})")
    if big and pure > max(3.0, 3 * exp_pure):
        issue(f"{pure} single-class corpus images (chance {exp_pure:.1f})")
    print(f"\nREVIEW SUMMARY: {len(issues)} blocking issues" + ("" if not issues else ": " + " | ".join(issues)))
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
