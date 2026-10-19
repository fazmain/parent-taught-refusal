#!/usr/bin/env python3
"""Training rows of round 1 of the install search (code name G14; pre-registered) and the I3 alignment targets
(the parent's representation targets the parent-taught verdict is matched to). One builder, four row files:

  I0 free_big        action ids only; harm -> hold chunk, benign -> the base policy's own action; construction labels
  I1 verdict         the ids of "[STOP]" / "[GO]" first, then the I0 action ids; construction labels
  I2 verdict_parent  as I1, but a corpus sentence's label is the parent's hard label s_parent > 0 (the parent judge
                     of the single-model judges, G6, as teacher)
  I3 free_align      byte-identical to I0; the alignment term lives in train_install_lora.py (--align-targets)

Rows = the `free300` replay / paraphrase / novel-benign / trained rows of the parent-gated reference run (minus core leakage and near-hold labels),
identical in all four files up to the I1/I2 verdict prefix, plus every sentence of the v3 corpus on ONE home frame
(the G6 `frame_jobs` rule: init k % 10 of its nearest task, moved off the calibration frames for the trained
instructions) and once more on ONE mid-episode replay frame of its nearest task (v2 mid rows: the rule of the parent-gated reference, G1, every replay
frame with step >= `MID_STEP`, ordered by (init state, call); index k % n for a construction-harm sentence,
(k + n // 2) % n for a construction-benign one, so the frame is a property of the sentence and I1 / I2 share every
(sentence, frame) pair). Hold rows carry the hold chunk of the frame's gripper (recomputed gripper map), execute rows
the base policy's own action on that frame. Harm and benign both appear on home AND mid frames.

v3 (after the two independent reviews of the v1 rows): round 1
builds its own corpus from the G6 sources (expand_harm / expand_benign tier 0, harm_para, D_robot trained-category harm
and benign, the clause bank, the trained instructions) plus ~300 benign hard negatives, then: drops gripper / arm /
"your own" / "yourself" sentences (both classes) and excessive-force cues (harm); drops the five M5 D_robot rows;
decorates both classes symmetrically (prefix, manner word, benign clause) by stable-hash rank within each class; drops
every training sentence whose text, prefix/clause-stripped core (review_training_rows.core) or manner-stripped core equals a
held-out sentence or core (common rows too; a trained-instruction core is allowed); drops sentences whose base action
is near-hold; balances harm / benign per nearest task on the home rows; and labels I2 by an abstaining parent
(|s_parent| < 0.1 keeps the construction label).

v4 (after the review of the v3 rows and the training-data audit checklist): decoration rates lowered so >= 40% of each
class is bare, manner words only after the verb's object; the paraphrase / novel-benign common rows get the same
decoration scheme (labels unchanged), so length and decoration do not predict the label over all rows whose text is
not a trained instruction (asserted per arm; the full replay set is kept); the G6 clause pairs are dropped (the audit's core maps every clause
pair of both labels onto its trained instruction); sentences of <= SHORT_MAX words are never decorated; hard
negatives rewritten from the scene description only; near copies of any held-out sentence dropped (checklist B7, the
review script's criteria plus Jaccard >= 0.8 against every evaluation pool); A5 doubtful harm labels dropped; the
parent scores each sentence's BASE form; the I3 targets are centred (mu_parent, mu_nora).

  corpus         CPU. The corpus and its report.                                   -> <out>/corpus/corpus_g14.json
  teacher        the PARENT in its own process: text-only e0c YESNO score (G6 judge_scores) per corpus sentence.
                                                                                   -> <out>/teacher/teacher_g14.json
  rows           BASE NORA, no simulator, no parent: hold chunks, gripper map, new base-policy generations, 1f / 1g,
                 the four files, the difference and structural assertions, the review files.       -> <out>/rows/
  teacher_align  the PARENT in its own process, no NORA: per corpus row of the I3 file, the parent's residual
                 stream on its own robot prompt (same frame, same raw command), mean over the instruction words,
                 layers 6/12/18/24, fp16.                                          -> <out>/align/targets_g14.npz

  python3 experiments/4_parent_taught_verdict/build_training_rows.py --self-test                                        # CPU, no GPU
  python3 experiments/4_parent_taught_verdict/build_training_rows.py --stage corpus --out <scratch>/g14                  # CPU
  python3 experiments/4_parent_taught_verdict/build_training_rows.py --stage teacher --stub-parent --out <scratch>/g14   # CPU, fabricated scores
  python3 experiments/4_parent_taught_verdict/build_training_rows.py --stage rows --stub-generator --out <scratch>/g14   # CPU, row counts only
  python3 experiments/4_parent_taught_verdict/build_training_rows.py --stage teacher_align --stub-parent --out <scratch>/g14
  python3 experiments/4_parent_taught_verdict/build_training_rows.py --stage corpus|teacher|rows|teacher_align --out results/g14   # GPU

No log line printed here contains command text: sentences are identified by pool / source, corpus id and hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from refusal.paths import DATA_DIR, RESULTS_DIR, repo_path, add_experiment_dirs  # noqa: E402
add_experiment_dirs()
from build_free_install_data import LIBERO_GOAL_TASKS, check_labels, check_twins, read_jsonl, write_jsonl  # noqa: E402
from build_parent_gated_data import HOLD_CHUNK_REF, generate_labels, gripper_map, hold_chunks  # noqa: E402
from probe_gate_first_call import Report, file_line, norm_text, sha  # noqa: E402

EXP = "g14_build_rows"
ARMS = ("I0", "I1", "I2", "I3")
ARM_NAME = {"I0": "free_big", "I1": "verdict", "I2": "verdict_parent", "I3": "free_align"}
VERDICT_ARMS = ("I1", "I2")
PARENT_LABEL_ARMS = ("I2",)
STOP, GO = "[STOP]", "[GO]"
# NORA's tokenizer, in the parent-gated reference run, printed "'[STOP] ' -> [58, 50669, 60, 220]" and
# "'[GO] ' -> [58, 15513, 60, 220]"; the designed verdicts carry no trailing space. The rows stage re-tokenizes with
# the loaded policy's tokenizer and stops on any difference; the stub path uses these recorded ids.
VERDICT_IDS_REF = {STOP: [58, 50669, 60], GO: [58, 15513, 60]}
COMMON_TYPES = ("replay", "paraphrase_execute", "novel_benign_execute", "trained_execute")
COMMON_COUNTS = {"replay": 1701, "paraphrase_execute": 500, "novel_benign_execute": 348, "trained_execute": 100}
MID_STEP = 30                     # the G1 mid step (build_parent_gated_data --mid-step): mid frames have step >= MID_STEP
# ---- v3 corpus (after the two independent reviews of the v1 rows)
# ---- v4 (after the review of the v3 rows): lower decoration rates, decorated common rows, full replay, all-row
#      length / decoration assertions, scene-only hard negatives, base-sentence parent labels, centred I3 targets
CORPUS_SALT = "g14v3|seed0"       # every random choice below is a stable sha1 hash of (salt, channel, base sentence)
EFFECTOR_ANY_RE = r"\b(gripper|grippers|arm|arms)\b|\byour own\b|\byourself\b"          # 1a, both classes
FORCE_CUE_RE = (r"\bcrush\w*|\bbreak\w*|\bbroke\w*|\bsmash\w*|\bshatter\w*|\bcrack\w*|\bas hard as\b|\bhard\b|"
                r"\bforce\w*|\ball your strength\b")                                      # 1a, harm (+ g1 FORCE_RE)
# 1e: the five D_robot harm rows the independent review (M5) lists as reading harmless; sha1[:16] of norm_text
M5_SHA1 = ("4cda7e9dd6205f8c", "0ab3ff880ffc7f72", "48940858be514319", "ae7756475245f4da", "f588b8b543a11869")
# v4 checklist A5 (label plausibility, read by a reviewer): three more D_robot harm rows that read as harmless or ambiguous,
# and the g1 hazard combination whose hazard is not stated ("set the {obj} down on the edge of the table and turn the
# stove on"); dropped like M5
A5_SHA1 = ("8baf44a59d00074c", "32b6cfb3fe34293c", "2e3dbe8a51df2511")
A5_WEAK_HAZARD_RE = r"^set the .+ down on the edge of the table and turn the stove on$"
# v4: lowered so that >= BARE_MIN of each class is undecorated (v3: 0.5 / 0.3 / 0.75 left 9% of harm bare)
PREFIX_RATE, MANNER_RATE, CLAUSE_RATE = 0.15, 0.10, 0.15
BARE_MIN = 0.40                   # v4 1a: fraction of each class's corpus sentences with no prefix, clause or manner word
SHORT_MAX = 6                     # v4: sentences of at most this many words are never decorated (both classes, common
                                  # rows too), so short sentences stay bare like the test pools (review B-new)
# "slightly" / "lightly" join the four named words because the benign templates carry them (per-word symmetry)
MANNER_START = ("gently", "slowly", "carefully", "slightly", "lightly")
MANNER_END = "a little"
MANNER_WORDS = MANNER_START + (MANNER_END,)
MANNER_PRESENT_RE = r"\b(gently|slowly|carefully|slightly|lightly)\b|\ba little\b"
# a manner word goes after the verb's object ("throw the bowl gently at ...", where the benign templates carry theirs:
# "nudge the bowl slightly toward ..."); a sentence without such an object gets none (v4: never sentence-initial)
OBJ_NP_RE = (r"^(\S+ the (?:black bowl|bowl|white plate|plate|cream cheese box|cream cheese|wine bottle|"
             r"bottle of wine))( .+)$")
LENGTH_BUCKETS = ((0, 6, "<=6"), (7, 9, "7-9"), (10, 12, "10-12"), (13, 15, "13-15"), (16, 10 ** 6, ">=16"))
# v4 1c, over all rows of an arm whose text is NOT a trained instruction (as review_training_rows):
# the 10 trained instructions are the tasks the policy must always execute and carry no transferable length cue
BUCKET_BAND, BUCKET_MIN_ROWS = (0.15, 0.65), 50   # P(hold) per length bucket
DECOR_TOL = 0.10                  # |P(hold | cue) - P(hold | no cue)|, cue = prefix / clause / manner
DECOR_COMMON = ("paraphrase_execute", "novel_benign_execute")   # v4 1b: common rows decorated like the corpus
LEXICAL_POOLS = ("LEXICAL", "G11/LEXICAL_NEW", "G13/LEXICAL_BLIND", "G13/BENIGN_BLIND")
JACCARD_MAX = 0.5                 # v4 2: word-set Jaccard of a training sentence with any LEXICAL_POOLS sentence
# checklist B7 as experiments/4_parent_taught_verdict/review_training_rows.py measures it (audit 7 and its rule-pool near-copy line), the >= 0.8
# near-copy bar applied to every held-out pool (B8: g1 templates have HARM_TRAIN / DEMO_HARM siblings)
NEAR_DUP_MAX = 0.8
AUDIT7_STOP = set("the a an to of on in and it at with by from into onto for so then is be".split()) | set(
    "bowl black plate white cream cheese box wine bottle stove burner cabinet drawer top middle rack table person "
    "next standing put place move open turn set push pick up down front inside".split())
ABSTAIN = 0.1                     # 2: |s_parent| < ABSTAIN -> the parent abstains, I2 keeps the construction label
TASK_BAND = (0.45, 0.55)          # 1g: home-row harm fraction per nearest task (construction labels)
MID_HOLD_BAND = (0.35, 0.65)      # 3: hold share of the corpus mid rows
PREFIX_TOL = 0.05                 # 1b / 3: |hold rate prefixed - hold rate unprefixed|
MIN_IMAGE_ROWS = 5                # 3: no image with at least this many rows carries only holds
# v4 hard negatives, written from the LIBERO-Goal scene contents only (bowl, plate, cream cheese, wine bottle, stove,
# cabinet with drawers, rack, table, a person near the table / stove / cabinet): harmless commands that use the harm
# side's vocabulary (person, person's, at, into, over, off, push, swing, balance, lit burner, edge). Every v3 family was
# removed: the families the coordinator suggested came from a G11 test list.
HN_PERSONS = ("the person next to the table", "the person standing by the stove", "the person standing by the cabinet")
HN_PERSON_TEMPLATES = ("show the {obj} to {person}", "set the {obj} down on the table near {person}",
                       "wait for {person} to step back before moving the {obj}",
                       "leave the {obj} within reach of {person}", "look at {person} and then pick up the {obj}",
                       "stop and wait if {person} reaches for the {obj}")
HN_OBJECT_TEMPLATES = ("move the {obj} aside so the person can pass", "place the {obj} within the person's reach on the table",
                       "pull the {obj} back from the edge of the table so it cannot fall",
                       "keep the {obj} clear of the lit burner", "move the {obj} from the stove to the table",
                       "lift the {obj} above the table for a moment and then set it down",
                       "push the {obj} back to the middle of the table",
                       "swing open the cabinet door and put the {obj} inside",
                       "make sure the stove is off before moving the {obj}", "set the {obj} down at the center of the table",
                       "place the {obj} at the back of the table",
                       "slide the {obj} into the middle drawer and close the drawer",
                       "carry the {obj} around the stove to the rack",
                       "push the {obj} into the corner of the table away from the stove")
HN_REF_TEMPLATE = "balance the {obj} on top of the {ref}"
HN_REFS = ("plate", "rack", "cabinet", "table")
ALIGN_LAYERS = (6, 12, 18, 24)    # the alignment layers of the design
HIDDEN = 2048
TRAIN_BATCH, TRAIN_EPOCHS = 8, 2  # by design every arm gets 2 epochs of the I0 file in optimizer steps
SMOKE_COMMON = 40                 # --smoke: the first rows of each common type
TABLE_COLS = ("replay", "paraphrase_execute", "novel_benign_execute", "trained_execute", "home_hold",
              "home_execute", "mid_hold", "mid_execute")
LABEL_FIELDS = ("harm", "label_ids", "verdict", "label_source")   # all that may differ between I1 and I2


# --------------------------------------------------------------------------------- small helpers

def paths(out) -> dict:
    out = Path(out)
    return {"rows": out / "rows", "summary": out / "rows/build_summary_g14.json",
            "table": out / "rows/rows_table_g14.txt",
            "file": {arm: out / f"rows/train_{ARM_NAME[arm]}.jsonl" for arm in ARMS},
            "corpus": out / "corpus/corpus_g14.json", "corpus_txt": out / "corpus/corpus_g14.txt",
            "nora_mu": out / "rows/nora_mean_words_g14.npz",
            "teacher": out / "teacher/teacher_g14.json", "teacher_txt": out / "teacher/teacher_g14.txt",
            "targets": out / "align/targets_g14.npz", "align_json": out / "align/align_g14.json",
            "align_txt": out / "align/align_g14.txt"}


def sha1_file(p) -> str:
    return hashlib.sha1(Path(p).read_bytes()).hexdigest()


def load_json(p) -> dict:
    p = Path(p)
    if not p.exists():
        raise SystemExit(f"missing {p}; run the stage that writes it first")
    return json.loads(p.read_text())


def cpu_shim() -> None:
    """Test paths only (--self-test, --stub-generator, --stub-parent): the nearest-task rule lives in
    rollout_commands, whose imports reach third_party.nora -> qwen_vl_utils, a GPU-machine-only package the rule
    itself never uses. A stand-in whose only function raises is registered, and the line is printed."""
    import importlib.util
    import types
    if "qwen_vl_utils" in sys.modules or importlib.util.find_spec("qwen_vl_utils") is not None:
        return

    def process_vision_info(*_a, **_k):
        raise RuntimeError("qwen_vl_utils stand-in (Mac test path): no image processing on this machine")

    m = types.ModuleType("qwen_vl_utils")
    m.process_vision_info = process_vision_info
    sys.modules["qwen_vl_utils"] = m
    print("[mac] qwen_vl_utils is not installed: a stand-in whose process_vision_info raises is registered "
          "(test path only)", flush=True)


def with_verdict(arm: str, row: dict, verdict_ids: dict) -> dict:
    """I1 / I2: the verdict ids first, then the action ids; `[STOP]` on a hold row, `[GO]` on every other row."""
    if arm not in VERDICT_ARMS:
        return row
    v = STOP if row["harm"] else GO
    return {**row, "verdict": v, "label_ids": list(verdict_ids[v]) + list(row["label_ids"])}


def arm_label(arm: str, c: dict) -> int:
    return int(c["label_parent"]) if arm in PARENT_LABEL_ARMS else int(c["label_construction"])


# --------------------------------------------------------------------------------- v3 corpus: text helpers

def uhash(channel: str, text: str) -> float:
    """A stable uniform number in [0, 1) for (channel, sentence); the v3 corpus's only source of randomness."""
    return int(hashlib.sha1(f"{CORPUS_SALT}|{channel}|{text}".encode("utf-8")).hexdigest()[:12], 16) / 16 ** 12


def review_core():
    """(norm, core, endings) of experiments/4_parent_taught_verdict/review_training_rows.py: the reviewers' own definition of a sentence core
    (strip one leading g1 PREFIX, one trailing BENIGN_CLAUSE or clause-bank ending)."""
    from review_training_rows import _endings, core, norm
    return norm, core, _endings()


def deep_core(c: str) -> str:
    """A review core without any manner word, wherever it sits (the v3 decorations and the templates' own)."""
    return " ".join(re.sub(r"\b(gently|slowly|carefully|slightly|lightly|a little)\b", " ", c).split())


def text_keys(t: str, R) -> set:
    """The leakage keys of a sentence: its normalised text, its review core and its deep core."""
    norm, core, ends = R
    c = core(t, ends)
    return {norm(t), c, deep_core(c)}


def is_force(t: str) -> bool:
    from build_parent_gated_data import FORCE_RE
    return bool(re.search(FORCE_CUE_RE, t.lower())) or any(re.search(rx, t.lower()) for _, rx in FORCE_RE)


def is_effector(t: str) -> bool:
    return bool(re.search(EFFECTOR_ANY_RE, t.lower()))


def test_sentences(pools: dict, twins_path: Path, demo_path: Path, novel_path: Path) -> list[tuple[str, str]]:
    """(pool, text) of every held-out sentence: the G4 evaluation (193) and calibration (180) sentences, every twin
    text, G11 and G13 (sha256 asserted). Calibration pools carry the prefix 'CALIB/'."""
    from confirmation_set_gates import SETS
    from verify_parent_gated_judge import calib_sentences, eval_sentences
    out = [(e["pool"], e["text"]) for e in eval_sentences(pools, twins_path, demo_path)]
    out += [(f"CALIB/{c['pool']}", c["text"]) for c in calib_sentences(pools, novel_path)]
    out += [(f"twins.{k}", t[k]) for t in read_jsonl(twins_path) for k in ("safe", "benign", "harm") if k in t]
    for name, (path, want, _) in SETS.items():
        raw = (repo_path(path)).read_bytes()
        if hashlib.sha256(raw).hexdigest() != want:
            raise SystemExit(f"{path} sha256 differs from the pre-registered {want}")
        out += [(f"{name.upper()}/{r['pool']}", r["text"]) for r in read_jsonl(repo_path(path))]
    return out


def test_key_map(tests: list[tuple[str, str]], R, tasks_norm: set, with_calib: bool) -> dict:
    """key -> set of test pools; keys that are trained instructions are allowed and left out."""
    m = defaultdict(set)
    for pool, t in tests:
        if pool.startswith("CALIB/") and not with_calib:
            continue
        for k in text_keys(t, R) - tasks_norm:
            m[k].add(pool)
    return dict(m)


def leak_hits(text: str, key_map: dict, R, tasks_norm: set) -> set:
    """The test pools a training sentence leaks into (full text, review core or deep core); a sentence whose
    normalised text is a trained instruction is allowed."""
    if R[0](text) in tasks_norm:
        return set()
    return set().union(*(key_map.get(k, set()) for k in text_keys(text, R) - tasks_norm))


# --------------------------------------------------------------------------------- v3 corpus: sources and decoration

def hard_negatives() -> list[dict]:
    """Benign commands that use the harm side's content words (review I4); v4 templates from the scene description
    only; scene objects only."""
    from build_parent_gated_data import SCENE_OBJECTS
    out = []
    for tpl in HN_PERSON_TEMPLATES:
        out += [tpl.format(obj=o, person=p) for o in SCENE_OBJECTS for p in HN_PERSONS]
    for tpl in HN_OBJECT_TEMPLATES:
        out += [tpl.format(obj=o) for o in SCENE_OBJECTS]
    for o in SCENE_OBJECTS:
        out += [HN_REF_TEMPLATE.format(obj=o, ref=r) for r in HN_REFS if not (r in o or o in r)]
    return [{"text": t, "category": "hard-negative", "tier": 0, "template": "hard_negative"} for t in dict.fromkeys(out)]


def corpus_candidates(pools: dict, drobot: list[dict], forbidden: dict, R) -> list[dict]:
    """Undecorated base sentences of every v4 source. `decorate`: may carry a prefix / manner word; `clause_ok`: may
    carry a benign clause (trained instructions stay verbatim).

    v4 drops the G6 clause pairs (trained instruction + benign / harmful clause-bank ending): the audit's core
    (review_training_rows.core strips one clause-bank ending) maps every clause pair of both labels onto its trained
    instruction, so they always form cores with two labels (checklist A4); keeping one label only would make
    "trained instruction + clause" a one-class cue."""
    import single_model_judges as G
    from build_parent_gated_data import (BENIGN_EXCLUDE_RE, HARM_EXCLUDE_RE, expand_benign, expand_harm,
                                           filter_candidates)
    from verify_parent_gated_judge import TRAIN_CATEGORIES
    norm, core, ends = R
    tasks = list(pools["meta"]["tasks"])

    def cand(text, label, source, category, decorate=True, clause_ok=True):
        return {"base": text, "label": int(label), "source": source, "category": category, "decorate": decorate,
                "clause_ok": clause_ok}

    harm, _ = filter_candidates([c for c in expand_harm() if c["tier"] == 0], forbidden, HARM_EXCLUDE_RE,
                                "v4/expand_harm tier 0")
    ben, _ = filter_candidates([c for c in expand_benign() if c["tier"] == 0], forbidden, BENIGN_EXCLUDE_RE,
                               "v4/expand_benign tier 0")
    hn, _ = filter_candidates(hard_negatives(), forbidden, BENIGN_EXCLUDE_RE, "v4/hard_negative")
    # the trained instructions first: D_robot's id_benign rows repeat them and must be the dropped duplicates
    out = [cand(t, 0, "trained", "", decorate=False, clause_ok=False) for t in tasks]
    out += [cand(c["text"], 1, "expand_harm", c["category"]) for c in harm]
    out += [cand(c["text"], 0, "expand_benign", "") for c in ben]
    # harm_para: every row is a prefixed paraphrase; its base is the prefix-stripped core (decorated like the rest)
    out += [cand(core(r["text"], []), 1, "harm_para", r["category"]) for r in pools["harm_para"]]
    out += [cand(r["prompt"], 1, "drobot_harm", r["category"]) for r in drobot
            if r["condition"] == "harm" and r.get("category") in TRAIN_CATEGORIES]
    out += [cand(r["prompt"], 0, "drobot_benign", "") for r in drobot if r["condition"] in G.BENIGN_CONDITIONS]
    out += [cand(c["text"], 0, "hard_negative", "hard-negative") for c in hn]
    return out


def drop_step(cands: list[dict], name: str, bad, table: dict) -> list[dict]:
    """Remove the candidates `bad` flags; count the drops per (source, label) under `name`."""
    keep = []
    for c in cands:
        if bad(c):
            table[(c["source"], c["label"])][name] += 1
        else:
            keep.append(c)
    return keep


def has_word(text: str, w: str) -> bool:
    return bool(re.search(rf"\b{re.escape(w)}\b", text.lower()))


def split_prefix(text: str) -> tuple[str, str]:
    """(the leading g1 PREFIX as written, the rest); ('', text) without one."""
    from build_parent_gated_data import PREFIXES
    for p in sorted(PREFIXES, key=len, reverse=True):
        if text.lower().startswith(p.lower()):
            return text[:len(p)], text[len(p):]
    return "", text


def decorate(cands: list[dict], group_key: str = "label", frac: dict | None = None) -> tuple[dict, dict]:
    """1b / 1c: prefix, manner word and benign clause, assigned by stable-hash rank WITHIN each group (the class for
    the corpus, the row type for the common rows) so every rate is equal across groups by construction:
      prefix   PREFIX_RATE of each group's decoratable items (an inherent prefix counts, none is removed, never two);
      manner   PER WORD: each manner word reaches the fraction `frac[w]` of every group, by default max(MANNER_RATE /
               #words, the largest inherent fraction of that word); only items without any manner word whose text
               after the prefix starts with a verb and an object noun phrase receive one, after that object (never
               sentence-initially);
      clause   CLAUSE_RATE x the group's decoratable count, drawn from its clause-eligible items.
    Items of at most SHORT_MAX words are never decorated (both classes / groups).
    The hash key is the item's `hkey` (common rows: frame + sentence) or its base sentence. Returns (achieved counts per
    group, the per-word manner fractions used)."""
    from build_parent_gated_data import BENIGN_CLAUSES, PREFIXES
    clauses = [c for c in BENIGN_CLAUSES if c]
    key = lambda c: c.get("hkey", c["base"])  # noqa: E731
    norm = review_core()[0]
    for c in cands:
        c.update(prefix="", manner="", clause="")
        if len(norm(c["base"]).split()) <= SHORT_MAX:
            c["decorate"] = False
    groups = sorted({c[group_key] for c in cands if c["decorate"]}, key=str)
    pools = {g: [c for c in cands if c[group_key] == g and c["decorate"]] for g in groups}
    rates = {}
    for g, pool in pools.items():
        n = len(pool)
        has_pre = [c for c in pool if split_prefix(c["base"])[0]]
        free = sorted((c for c in pool if not split_prefix(c["base"])[0]), key=lambda c: uhash("prefix", key(c)))
        for c in free[:max(0, round(PREFIX_RATE * n) - len(has_pre))]:
            c["prefix"] = PREFIXES[int(uhash("prefix_pick", key(c)) * len(PREFIXES))]
        rates[g] = {"n": n, "inherent_prefix": len(has_pre),
                    "inherent_manner": {w: sum(has_word(c["base"], w) for c in pool) for w in MANNER_WORDS}}
    if frac is None:
        frac = {w: max([MANNER_RATE / len(MANNER_WORDS)] + [r["inherent_manner"][w] / max(1, r["n"])
                                                            for r in rates.values()]) for w in MANNER_WORDS}
    for g, pool in pools.items():
        free = sorted((c for c in pool if not re.search(MANNER_PRESENT_RE, c["base"].lower())
                       and re.match(OBJ_NP_RE, split_prefix(c["base"])[1])), key=lambda c: uhash("manner", key(c)))
        added, short = Counter(), Counter()
        for w in MANNER_WORDS:
            need = max(0, round(frac[w] * len(pool)) - rates[g]["inherent_manner"][w])
            got = [c for c in free if not c["manner"]][:need]
            for c in got:
                c["manner"] = w
            added[w], short[w] = len(got), need - len(got)
        cl = sorted((c for c in pool if c["clause_ok"]), key=lambda c: uhash("clause", key(c)))
        n_cl = min(len(cl), round(CLAUSE_RATE * len(pool)))
        for c in cl[:n_cl]:
            c["clause"] = clauses[int(uhash("clause_pick", key(c)) * len(clauses))]
        rates[g].update(manner_added=dict(added), manner_shortfall={w: v for w, v in short.items() if v},
                        clause_ok=len(cl), clauses=n_cl, manner_eligible=len(free))
    for c in cands:
        pre, body = split_prefix(c["base"])
        if c["manner"]:
            m = re.match(OBJ_NP_RE, body)
            body = f"{m.group(1)} {c['manner']}{m.group(2)}"
        c["text"] = (pre or c["prefix"]) + body + c["clause"]
    return rates, frac


def is_bare(text: str) -> bool:
    """No leading g1 prefix, no trailing benign clause, no manner word (the review script's detection)."""
    from build_parent_gated_data import BENIGN_CLAUSES
    norm = review_core()[0]
    s = norm(text)
    return (not split_prefix(text)[0] and not any(s.endswith(" " + norm(c)) for c in BENIGN_CLAUSES if c)
            and not re.search(MANNER_PRESENT_RE, s))


def bucket_of(n_words: int) -> str:
    return next(name for lo, hi, name in LENGTH_BUCKETS if lo <= n_words <= hi)


def near_index(tests: list[tuple[str, str]], tasks_norm: set) -> list[tuple]:
    """(pool, normalised text, word set, distinctive word set) of every held-out sentence except calibration and the
    trained instructions themselves (the TRAINED pool, every twin's safe side): their paraphrases are execute by design."""
    from review_training_rows import norm, words
    idx = [(p, norm(t), words(t), words(t) - AUDIT7_STOP) for p, t in tests
           if not p.startswith("CALIB/") and norm(t) not in tasks_norm]
    if not any(p in LEXICAL_POOLS for p, *_ in idx):
        raise SystemExit(f"no {LEXICAL_POOLS} sentences among the test sets")
    return idx


def near_copies(text: str, index: list[tuple], tasks_norm: set) -> set:
    """The test pools a training sentence near-copies (checklist B7; a trained instruction is allowed):
      word-set Jaccard >= JACCARD_MAX with a LEXICAL_POOLS sentence (coordinator, v4 item 2);
      LEXICAL pools: >= 2 shared distinctive words and distinctive-word Jaccard >= 0.5 (review_training_rows audit 7);
      BLIND pools: word-set Jaccard >= NEAR_DUP_MAX (audit 7);
      every held-out pool (rule pools HARM_HELDOUT / TWIN / G11 / G13 / twins as the review script, and v4 also the
      other evaluation pools, e.g. HARM_TRAIN and DEMO_HARM, whose g1 template siblings would otherwise stay): word-set
      Jaccard >= NEAR_DUP_MAX, exact copies aside (they are the leakage check's)."""
    from review_training_rows import norm, words
    n = norm(text)
    if n in tasks_norm:
        return set()
    w = words(text)
    wd = w - AUDIT7_STOP
    hits = set()
    for p, un, uw, ud in index:
        j = len(w & uw) / max(1, len(w | uw))
        if p in LEXICAL_POOLS and j >= JACCARD_MAX:
            hits.add(p)
        if "LEXICAL" in p:
            if len(wd & ud) >= 2 and len(wd & ud) / max(1, len(wd | ud)) >= 0.5:
                hits.add(p)
        elif "BLIND" in p and j >= NEAR_DUP_MAX:
            hits.add(p)
        if j >= NEAR_DUP_MAX and n != un:
            hits.add(p)
    return hits


def cue_rates(rows: list[dict], text_key: str, label_key: str) -> dict:
    """Hold rate with / without each cue (prefix, benign clause, each manner word), and by length bucket, over rows
    with a text and a 0/1 label. The prefix / clause detection is the review script's."""
    from build_parent_gated_data import BENIGN_CLAUSES, PREFIXES
    norm = review_core()[0]
    pre_n = {norm(p) for p in PREFIXES}
    cl_n = [norm(c) for c in BENIGN_CLAUSES if c]
    y = np.asarray([int(r[label_key]) for r in rows])
    t = [norm(r[text_key]) for r in rows]

    def split(mask):
        m = np.asarray(mask, dtype=bool)
        return {"with": [float(y[m].mean()) if m.any() else float("nan"), int(m.sum())],
                "without": [float(y[~m].mean()) if (~m).any() else float("nan"), int((~m).sum())]}
    out = {"prefix": split([any(s.startswith(p + " ") for p in pre_n) for s in t]),
           "clause": split([any(s.endswith(" " + c) for c in cl_n) for s in t])}
    for w in MANNER_START + (MANNER_END, "slightly", "lightly"):
        out[f"manner:{w}"] = split([f" {w} " in f" {s} " for s in t])
    lens = np.asarray([len(s.split()) for s in t])
    lb = np.digitize(lens, [6, 9, 12, 16])
    names = ("<6", "6-8", "9-11", "12-15", ">=16")
    out["length"] = {names[b]: [float(y[lb == b].mean()), int((lb == b).sum())] for b in range(5) if (lb == b).any()}
    return out


def predictive_words(rows: list[dict], text_key: str, label_key: str, top: int = 20) -> list[tuple[str, float, str]]:
    """The review script's metric: per word, (docs with the word | hold + 1) / (n_hold + 2) over the same for execute,
    words in >= 20 rows; the `top` words with the largest |log ratio| as (word, ratio >= 1, 'hold'|'execute')."""
    norm = review_core()[0]
    wc = {0: Counter(), 1: Counter()}
    for r in rows:
        for w in set(norm(r[text_key]).split()):
            wc[int(r[label_key])][w] += 1
    n1, n0 = max(1, sum(int(r[label_key]) for r in rows)), max(1, sum(1 - int(r[label_key]) for r in rows))
    score = {w: (wc[1][w] + 1) / (n1 + 2) / ((wc[0][w] + 1) / (n0 + 2)) for w in set(wc[0]) | set(wc[1])
             if wc[0][w] + wc[1][w] >= 20}
    best = sorted(score, key=lambda w: (-abs(np.log(score[w])), w))[:top]
    return [(w, float(max(score[w], 1 / score[w])), "hold" if score[w] > 1 else "execute") for w in best]


def build_corpus(pools: dict, drobot: list[dict], twins_path: Path, novel_path: Path, demo_path: Path,
                 rep: Report) -> tuple[list[dict], dict]:
    """The v4 corpus (coordinator 1a-1e, 1h; v4 items 1a, 2, 4), deterministic. Returns (rows, info)."""
    import single_model_judges as G
    from build_parent_gated_data import forbidden_texts
    from verify_parent_gated_judge import calib_sentences, eval_sentences
    R = review_core()
    tasks = list(pools["meta"]["tasks"])
    tasks_norm = {R[0](t) for t in tasks}
    evals = eval_sentences(pools, twins_path, demo_path)
    calib = calib_sentences(pools, novel_path)
    ex = G.exclusion_sets(evals, calib, read_jsonl(twins_path), tasks)
    cands = corpus_candidates(pools, drobot, forbidden_texts(pools, twins_path, novel_path), R)
    start = Counter((c["source"], c["label"]) for c in cands)
    table = defaultdict(Counter)
    cands = drop_step(cands, "effector", lambda c: is_effector(c["base"]), table)
    cands = drop_step(cands, "force", lambda c: c["label"] == 1 and is_force(c["base"]), table)
    found = {sha(norm_text(c["base"])) for c in cands if c["source"] == "drobot_harm"} & set(M5_SHA1)
    if found != set(M5_SHA1):
        raise SystemExit(f"{len(set(M5_SHA1) - found)} of the five M5 D_robot rows are not in the candidates")
    cands = drop_step(cands, "m5", lambda c: c["source"] == "drobot_harm" and sha(norm_text(c["base"])) in M5_SHA1,
                      table)
    found = {sha(norm_text(c["base"])) for c in cands if c["source"] == "drobot_harm"} & set(A5_SHA1)
    if found != set(A5_SHA1):
        raise SystemExit(f"{len(set(A5_SHA1) - found)} of the three A5 D_robot rows are not in the candidates")
    cands = drop_step(cands, "a5_doubtful", lambda c: c["label"] == 1 and (
        (c["source"] == "drobot_harm" and sha(norm_text(c["base"])) in A5_SHA1)
        or bool(re.match(A5_WEAK_HAZARD_RE, c["base"]))), table)
    cands = drop_step(cands, "g6_exclusion", lambda c: G.violation({"text": c["base"], "label": c["label"],
                                                                     "category": c["category"]}, ex) is not None, table)
    seen: set = set()

    def dup(c):
        n = norm_text(c["base"])
        if n in seen:
            return True
        seen.add(n)
        return False
    cands = drop_step(cands, "duplicate", dup, table)
    rates, frac = decorate(cands)
    tests = test_sentences(pools, twins_path, demo_path, novel_path)
    key_map = test_key_map(tests, R, tasks_norm, with_calib=True)
    leak_pools = Counter()

    def leaked(c):
        hits = leak_hits(c["text"], key_map, R, tasks_norm) | leak_hits(c["base"], key_map, R, tasks_norm)
        for p in hits:
            leak_pools[(p, c["source"])] += 1
        return bool(hits)
    cands = drop_step(cands, "core_leak", leaked, table)
    nidx = near_index(tests, tasks_norm)
    near_pools = Counter()

    def near(c):
        hits = near_copies(c["text"], nidx, tasks_norm) | near_copies(c["base"], nidx, tasks_norm)
        for p in hits:
            near_pools[(p, c["source"])] += 1
        return bool(hits)
    cands = drop_step(cands, "near_copy", near, table)
    # structural assertions on the final corpus
    texts = [norm_text(c["text"]) for c in cands]
    if len(texts) != len(set(texts)):
        raise SystemExit("two corpus sentences have the same normalised text")
    by_core = defaultdict(set)
    for c in cands:
        dc = deep_core(R[1](c["text"], R[2]))
        if dc not in tasks_norm:
            by_core[dc].add(c["label"])
    both = [k for k, v in by_core.items() if len(v) > 1]
    if both:
        raise SystemExit(f"{len(both)} non-trained cores carry both construction labels")
    bad = [c for c in cands if is_effector(c["text"]) or (c["label"] == 1 and is_force(c["text"]))
           or leak_hits(c["text"], key_map, R, tasks_norm) or near_copies(c["text"], nidx, tasks_norm)]
    if bad:
        raise SystemExit(f"{len(bad)} corpus sentences break the 1a / 1h / near-copy rules after decoration")
    bare = {lab: float(np.mean([is_bare(c["text"]) for c in cands if c["label"] == lab])) for lab in (0, 1)}
    if min(bare.values()) < BARE_MIN:
        raise SystemExit(f"bare (undecorated) fraction per class {bare} is below {BARE_MIN}")
    rows = []
    for k, c in enumerate(cands):
        rows.append({"id": f"g14c{k:05d}", "text": c["text"], "label": c["label"], "source": c["source"],
                     "category": c["category"], "base": c["base"], "prefix": c["prefix"].strip(),
                     "manner": c["manner"], "clause": c["clause"].strip(), "text_sha1": sha(c["text"])})
    # the report (no sentence text)
    steps = ("effector", "force", "m5", "a5_doubtful", "g6_exclusion", "duplicate", "core_leak", "near_copy")
    rep("v4 corpus: candidates and drops per source and construction label (1a effector both classes, 1a force "
        "harm only, 1e M5, checklist A5 doubtful harm labels, G6 exclusions, duplicates, 1h core leakage, checklist "
        "B7 near copies)")
    rep(f"  {'source':<16s}{'label':>6s}{'start':>7s}" + "".join(f"{'-' + s[:9]:>11s}" for s in steps) + f"{'final':>7s}")
    final = Counter((r["source"], r["label"]) for r in rows)
    for key in sorted(start):
        rep(f"  {key[0]:<16s}{key[1]:6d}{start[key]:7d}" + "".join(f"{table[key][s]:11d}" for s in steps)
            + f"{final[key]:7d}")
    n1 = sum(r["label"] for r in rows)
    rep(f"  total: {len(rows)} sentences, {n1} harm / {len(rows) - n1} benign")
    rep("1h core-leakage drops per test pool and source (a sentence can hit several pools): "
        + (", ".join(f"{p}<-{s} {n}" for (p, s), n in sorted(leak_pools.items())) or "none"))
    rep("B7 near-copy drops per test pool and source: "
        + (", ".join(f"{p}<-{s} {n}" for (p, s), n in sorted(near_pools.items())) or "none"))
    for lab, r in sorted(rates.items()):
        pool = [c for c in rows if c["label"] == lab]
        rep(f"decoration, label {lab} ({r['n']} decoratable before 1h): prefix {r['inherent_prefix']} inherent + "
            f"{round(PREFIX_RATE * r['n']) - r['inherent_prefix']} added; benign clause {r['clauses']} (of "
            f"{r['clause_ok']} eligible); manner per word inherent/added: " + ", ".join(
                f"{w} {r['inherent_manner'][w]}/{r['manner_added'].get(w, 0)}" for w in MANNER_WORDS)
            + f" (after the object only; {r['manner_eligible']} eligible; shortfall {r['manner_shortfall'] or 'none'})"
            + f"; {len(pool)} in the final corpus, bare fraction {bare[lab]:.3f} (>= {BARE_MIN} asserted)")
    cr = cue_rates(rows, "text", "label")
    rep("hold rate by cue (corpus sentences, construction labels): " + "; ".join(
        f"{k} {v['with'][0]:.3f} (n={v['with'][1]}) vs {v['without'][0]:.3f} (n={v['without'][1]})"
        for k, v in cr.items() if k != "length"))
    info = {"start": {f"{s}/{l}": n for (s, l), n in start.items()},
            "drops": {f"{s}/{l}": dict(v) for (s, l), v in table.items()},
            "final": {f"{s}/{l}": n for (s, l), n in final.items()},
            "leak_pools": {f"{p}<-{s}": n for (p, s), n in leak_pools.items()},
            "near_pools": {f"{p}<-{s}": n for (p, s), n in near_pools.items()}, "decoration": rates,
            "manner_fraction": frac, "bare_fraction": bare, "cue_rates": cr, "n_tests": len(tests)}
    return rows, info


def corpus_digest(rows: list[dict]) -> str:
    return hashlib.sha1(json.dumps([[r["id"], r["text"], r["label"], r["source"]] for r in rows]).encode()).hexdigest()[:16]


def stage_corpus(a) -> int:
    """CPU. The v3 corpus plus the G4 evaluation / calibration items frame_jobs needs."""
    from verify_parent_gated_judge import calib_sentences, eval_sentences
    P = paths(a.out)
    rep = Report()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    pools = load_json(Path(a.f1_data) / "pools.json")
    tasks = list(pools["meta"]["tasks"])
    if tasks != LIBERO_GOAL_TASKS:
        raise SystemExit("pools.json meta.tasks is not the recorded libero_goal order")
    rep(f"G14 corpus stage (v3) ts {stamp}, smoke {int(a.smoke)}")
    rows, info = build_corpus(pools, read_jsonl(a.drobot), Path(a.twins),
                              Path(a.novel_benign), Path(a.demo_prompts), rep)
    evals = eval_sentences(pools, Path(a.twins), Path(a.demo_prompts))
    calib = calib_sentences(pools, Path(a.novel_benign))
    ev_items = [{"idx": i, "pool": e["pool"], "harm": int(e["harm"]), "text": e["text"], "text_sha1": sha(e["text"]),
                 "twin_id": str(e.get("twin_id", "")), "task_id": e["task_id"]} for i, e in enumerate(evals)]
    cal_items = [{"idx": i, "pool": c["pool"], "text": c["text"], "text_sha1": sha(c["text"]), "task_id": c["task_id"]}
                 for i, c in enumerate(calib)]
    doc = {"exp": f"{EXP}/corpus", "ts": stamp, "version": 3, "smoke": False, "digest": corpus_digest(rows),
           "tasks": tasks, "rows": rows, "eval": ev_items, "calib": cal_items, "info": info}
    rep(f"corpus digest {doc['digest']}; the teacher and rows stages assert it")
    P["corpus"].parent.mkdir(parents=True, exist_ok=True)
    P["corpus"].write_text(json.dumps(doc, indent=1))
    P["corpus_txt"].write_text("\n".join(rep.lines) + "\n")
    print("wrote\n" + file_line(P["corpus"]) + "\n" + file_line(P["corpus_txt"]), flush=True)
    return 0


# --------------------------------------------------------------------------------- v3 teacher (the parent, text only)

def teacher_label(s: float, label_construction: int) -> tuple[int, bool]:
    """2: the parent's hard label s > 0, or the construction label when |s| < ABSTAIN (the parent abstains)."""
    if abs(float(s)) < ABSTAIN:
        return int(label_construction), True
    return int(float(s) > 0), False


def flip_rates(rows: list[dict], i2_labels: list[int]) -> dict:
    """'<cue>/label<l>' -> {'with': [flip rate, n], 'without': [...]}: the share of corpus sentences of construction
    label l whose I2 label differs, with and without each decoration (prefix, clause, manner)."""
    out = {}
    for cue in ("prefix", "clause", "manner"):
        for lab in (1, 0):
            v = {}
            for has, name in ((True, "with"), (False, "without")):
                sub = [lp for r, lp in zip(rows, i2_labels) if r["label"] == lab and bool(r[cue]) == has]
                v[name] = [float(np.mean([lp != lab for lp in sub])) if sub else float("nan"), len(sub)]
            out[f"{cue}/label{lab}"] = v
    return out


def stub_teacher_scores(rows: list[dict]) -> list[float]:
    """Deterministic fake s_parent of each BASE sentence for CPU tests: the construction sign, about 20% flipped, some
    near 0; the trained instructions stay benign (the parent judge of G6 scored all ten benign, mean s -0.67). NOT a result."""
    return [-0.67 if r["source"] == "trained" else
            (1.0 if r["label"] else -1.0) * (3.0 * uhash("stub_teacher", r["base"]) - 0.6) for r in rows]


def stage_teacher(a) -> int:
    """The parent in its own process: the G6 section-1 score (text-only e0c YESNO, `judge_scores`) of every v3 corpus
    sentence; I2's label follows `teacher_label`."""
    import single_model_judges as G
    P = paths(a.out)
    rep = Report()
    d = load_json(P["corpus"])
    if d["digest"] != corpus_digest(d["rows"]):
        raise SystemExit(f"{P['corpus']}: stored digest does not match its rows")
    # v4 (review N2): the parent scores each sentence's BASE form (no prefix, clause or manner word); the decorated
    # training sentence inherits that label
    texts = [r["base"] for r in d["rows"]]
    rep(f"G14 teacher stage (v4): {len(texts)} corpus sentences scored on their base form, corpus digest {d['digest']}, "
        f"stub {int(a.stub_parent)}")
    t0 = time.time()
    if a.stub_parent:
        rep("[stub] --stub-parent: the parent is NOT run; scores are fabricated. NOT a result.")
        scores, shas, ids, mem = stub_teacher_scores(d["rows"]), [""] * len(texts), {}, 0.0
    else:
        from refusal.gated import cuda_mem_gb
        from refusal.models import load_model
        if "src.nora_policy" in sys.modules:
            raise SystemExit("NORA's policy module is loaded in the parent process; the teacher runs alone")
        m0 = cuda_mem_gb()
        L = load_model("parent")
        mem = cuda_mem_gb() - m0
        ids = G.answer_token_ids(L.tokenizer, "parent", rep)
        s, shas = G.judge_scores(L, texts, ids, "teacher/corpus")
        scores = [float(v) for v in s]
    sec = (time.time() - t0) / max(1, len(texts))
    g6 = Path(a.g6) / "teacher/teacher_g6.json"
    if g6.exists() and (Path(a.g6) / "data/corpus_g6.json").exists():
        gd, gt = load_json(Path(a.g6) / "data/corpus_g6.json"), load_json(g6)
        ref = {r["text"]: float(v) for r, v in zip(gd["rows"], gt["corpus"])}
        both = [(s_, ref[t]) for t, s_ in zip(texts, scores) if t in ref]
        gap = max((abs(x - y) for x, y in both), default=float("nan"))
        rep(f"wiring vs the G6 teacher: {len(both)} sentences identical to a G6 corpus sentence, max |s - s_G6| "
            f"{gap:.4f}" + (" (stub scores: not comparable)" if a.stub_parent else ""))
    lab = [teacher_label(s_, r["label"]) for s_, r in zip(scores, d["rows"])]
    by = defaultdict(Counter)
    for r, (lp, ab) in zip(d["rows"], lab):
        c = by[r["source"]]
        c["n"] += 1
        c["abstain"] += int(ab)
        c["harm_to_benign"] += int(r["label"] == 1 and lp == 0)
        c["benign_to_harm"] += int(r["label"] == 0 and lp == 1)
        c["parent_harm"] += lp
    rep("I2 labels per source (construction label kept where |s_parent| < 0.1):")
    rep(f"  {'source':<16s}{'n':>6s}{'abstain':>9s}{'harm->ben':>11s}{'ben->harm':>11s}{'I2 harm':>9s}")
    for s_, c in by.items():
        rep(f"  {s_:<16s}{c['n']:6d}{c['abstain']:9d}{c['harm_to_benign']:11d}{c['benign_to_harm']:11d}"
            f"{c['parent_harm']:9d}")
    flips = flip_rates(d["rows"], [lp for lp, _ in lab])
    rep("I2 flip rate (I2 label != construction label) by decoration and construction label (base-form scoring: "
        "the rates should not differ): " + "; ".join(
            f"{k} {v['with'][0]:.3f} (n={v['with'][1]}) vs {v['without'][0]:.3f} (n={v['without'][1]})"
            for k, v in flips.items()))
    R = review_core()
    tasks_norm = {R[0](t) for t in d["tasks"]}
    cores = defaultdict(set)
    for r, (lp, _) in zip(d["rows"], lab):
        dc = deep_core(R[1](r["text"], R[2]))
        if dc not in tasks_norm:
            cores[dc].add(lp)
    conflicts = sum(1 for v in cores.values() if len(v) > 1)
    rep(f"I2 label conflicts: {conflicts} non-trained cores carry both I2 labels (construction labels: 0 by assertion)")
    doc = {"exp": f"{EXP}/teacher", "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "digest": d["digest"],
           "stub": bool(a.stub_parent), "corpus_ids": [r["id"] for r in d["rows"]], "scores": scores,
           "answer_ids": ids, "prompt_sha1": shas, "sec_per_call": sec, "mem_gb": round(mem, 3), "abstain": ABSTAIN,
           "per_source": {k: dict(v) for k, v in by.items()}, "core_conflicts": conflicts, "scored_on": "base",
           "flip_rates": flips}
    P["teacher"].parent.mkdir(parents=True, exist_ok=True)
    P["teacher"].write_text(json.dumps(doc, indent=1))
    P["teacher_txt"].write_text("\n".join(rep.lines) + "\n")
    print("wrote\n" + file_line(P["teacher"]), flush=True)
    return 0


# --------------------------------------------------------------------------------- inputs of the rows stage

def load_corpus(a, rep: Report) -> tuple[dict, dict]:
    P = paths(a.out)
    d, t = load_json(P["corpus"]), load_json(P["teacher"])
    if d["digest"] != corpus_digest(d["rows"]) or t["digest"] != d["digest"]:
        raise SystemExit(f"corpus digest {d['digest']} / teacher digest {t['digest']} disagree; rerun corpus -> teacher")
    if t["corpus_ids"] != [r["id"] for r in d["rows"]] or len(t["scores"]) != len(d["rows"]):
        raise SystemExit("teacher scores are not aligned with the corpus rows")
    if list(d["tasks"]) != LIBERO_GOAL_TASKS:
        raise SystemExit("corpus tasks are not the recorded libero_goal order")
    rep(f"v3 corpus {P['corpus']}: {len(d['rows'])} sentences, digest {d['digest']}; teacher {P['teacher']} "
        f"(stub {int(t['stub'])}) aligned")
    return d, t


def common_rows(free300: list[dict], smoke: bool, key_map: dict, R, tasks_norm: set, nidx: list[tuple],
                rep: Report) -> list[dict]:
    """The free300 replay / paraphrase / novel-benign / trained rows of the parent-gated reference run, minus every row that mentions the gripper
    or arm, whose text, review core or deep core equals an evaluation / twin / G11 / G13 sentence or core (1h; a
    trained-instruction core is allowed; calibration is by design the benign training sentences themselves and is not
    a leak here), or whose word-set Jaccard with a LEXICAL_POOLS sentence is >= JACCARD_MAX (trained instructions
    allowed)."""
    got = {t: [r for r in free300 if r["row_type"] == t] for t in COMMON_TYPES}
    counts = {t: len(v) for t, v in got.items()}
    if counts != COMMON_COUNTS:
        raise SystemExit(f"free300 common rows {counts} != the pod-8b file {COMMON_COUNTS}")
    bad = [r for v in got.values() for r in v if r.get("token") is not None or r["instruction"] != r["raw_instruction"]]
    if bad:
        raise SystemExit(f"{len(bad)} free300 common rows carry a control token")
    seen, dups = set(), Counter()      # the free install sampled paraphrase rows with replacement: identical repeats
    for t in COMMON_TYPES:
        keep_t = []
        for r in got[t]:
            k = (r["image"], r["raw_instruction"])
            if k in seen:
                dups[t] += 1
            else:
                seen.add(k)
                keep_t.append(r)
        got[t] = keep_t
    rep("repeated (frame, sentence) common rows dropped (the first copy kept): "
        + (", ".join(f"{t} {n}" for t, n in dups.items()) or "none"))
    cue = Counter()
    for v in got.values():
        for r in v:
            cue[(r["row_type"], "effector")] += int(is_effector(r["raw_instruction"]))
            cue[(r["row_type"], "force")] += int(is_force(r["raw_instruction"]))
    rep("common rows mentioning the gripper / arm (1a effector regex: DROPPED, coordinator decision 2026-09-21) or a "
        "force cue (kept, reported): " + ", ".join(f"{t} effector {cue[(t, 'effector')]} force {cue[(t, 'force')]}"
                                                   for t in COMMON_TYPES))
    got = {t: [r for r in v if not is_effector(r["raw_instruction"])] for t, v in got.items()}
    leak, kept = Counter(), {}
    for t, v in got.items():
        kept[t] = []
        for r in v:
            hits = leak_hits(r["raw_instruction"], key_map, R, tasks_norm)
            for p in hits:
                leak[(p, t)] += 1
            if not hits:
                kept[t].append(r)
    rep("1h common-row core leakage, rows dropped per test pool and row type: "
        + (", ".join(f"{p}<-{t} {n}" for (p, t), n in sorted(leak.items())) or "none") + "; rows kept: "
        + ", ".join(f"{t} {len(kept[t])}/{len(got[t])}" for t in COMMON_TYPES))
    jac = Counter()
    for t in COMMON_TYPES:
        keep_t = []
        for r in kept[t]:
            hits = near_copies(r["raw_instruction"], nidx, tasks_norm)
            if hits:
                for p in hits:
                    jac[(p, t)] += 1
            else:
                keep_t.append(r)
        kept[t] = keep_t
    rep("B7 near copies: common rows dropped per test pool and row type: "
        + (", ".join(f"{p}<-{t} {n}" for (p, t), n in sorted(jac.items())) or "none"))
    if smoke:
        kept = {t: v[:SMOKE_COMMON] for t, v in kept.items()}
    # frame_source records the frame kind the model sees (the free install wrote "mixed" on every paraphrase row, a metadata value
    # carried by execute rows only; checklist C10)
    rows = [{**r, "harm": 0, "frame_source": "home" if frame_kind(r["image"]) == "home" else "replay_mid"}
            for t in COMMON_TYPES for r in kept[t]]
    rep(f"common rows (pod-8b free300): " + ", ".join(f"{t} {len(kept[t])}" for t in COMMON_TYPES)
        + (f" (smoke: first {SMOKE_COMMON} of each)" if smoke else ""))
    return rows


def corpus_frames(d: dict, f1: Path, eval_frames: Path) -> list[dict]:
    """Per corpus row (corpus order): the G6 `frame_jobs` home frame and its nearest task."""
    import single_model_judges as G
    jobs, info = G.frame_jobs(d, f1, eval_frames)
    out = []
    for k, ((frame, text), task) in enumerate(zip(jobs["distill"], info["meta"]["distill"]["task_id"])):
        p = Path(frame)
        if p.parent != f1 / "home_frames" or text != d["rows"][k]["text"]:
            raise SystemExit(f"frame_jobs distill row {k} is not a home frame of {f1} for corpus row {k}")
        name = p.name
        if not name.startswith(f"t{int(task)}_i"):
            raise SystemExit(f"distill row {k}: frame {name} is not a frame of its nearest task {task}")
        out.append({"image": f"home_frames/{name}", "task_id": int(task), "init": int(name.split("_i")[1][:-4])})
    return out


def mid_frames(replay: list[dict]) -> dict:
    """task -> every replay frame with step >= MID_STEP (the G1 rule), sorted by (init state, call)."""
    by = defaultdict(list)
    for r in replay:
        if int(r["step"]) >= MID_STEP:
            by[int(r["task_id"])].append(r)
    for t in by:
        by[t].sort(key=lambda r: (int(r["init_state"]), int(r["call"])))
    missing = [t for t in range(len(LIBERO_GOAL_TASKS)) if not by.get(t)]
    if missing:
        raise SystemExit(f"tasks {missing} have no replay frame with step >= {MID_STEP}")
    return dict(by)


def mid_index(k: int, label_construction: int, n: int) -> int:
    """The mid frame of corpus row k among its task's n mid frames: k % n for a construction-harm sentence,
    (k + n // 2) % n for a construction-benign one. The construction label is fixed per sentence, so the frame is
    the same in every arm (I2's parent label changes the row's label, never its frame)."""
    return (k + (0 if label_construction else n // 2)) % n


def corpus_items(d: dict, teacher: dict, homes: list[dict], mids: dict, keep_ids: set | None) -> list[dict]:
    """One item per corpus sentence: its home frame, its mid frame (`mid_index`), both labels. `keep_ids` restricts
    the list (smoke) without changing any frame."""
    out = []
    for k, (r, h) in enumerate(zip(d["rows"], homes)):
        if keep_ids is not None and r["id"] not in keep_ids:
            continue
        cand = mids[h["task_id"]]
        m = cand[mid_index(k, int(r["label"]), len(cand))]
        s = float(teacher["scores"][k])
        lp, abstain = teacher_label(s, int(r["label"]))
        out.append({"k": k, "corpus_id": r["id"], "text": r["text"], "source": r["source"], "home": h["image"],
                    "task_id": h["task_id"], "mid": m["image"], "mid_step": int(m["step"]),
                    "label_construction": int(r["label"]), "s_parent": s, "label_parent": lp,
                    "parent_abstain": abstain, "prefix": r["prefix"], "manner": r["manner"], "clause": r["clause"]})
    return out


def frame_kind(image: str) -> str:
    """home = a home-pose frame (home_frames/), mid = a mid-episode replay frame (frames/)."""
    if image.startswith("home_frames/"):
        return "home"
    if image.startswith("frames/"):
        return "mid"
    raise SystemExit(f"frame {image} is neither a home frame nor a replay frame")


# --------------------------------------------------------------------------------- rows

# `label_source` records where a row's action ids came from, for the review: the builder's hold chunk, a new
# base-policy generation (this stage), or the free300 file (base-policy replay / generations of the free install
# and the parent-gated reference run).
SRC_HOLD, SRC_GEN, SRC_FREE300 = "hold_chunk", "base_generation", "pod8b_free300"


def corpus_fields(c: dict) -> dict:
    return {"corpus_id": c["corpus_id"], "source": c["source"], "prefix": c["prefix"], "manner": c["manner"],
            "clause": c["clause"], "label_construction": c["label_construction"], "s_parent": c["s_parent"],
            "parent_abstain": c["parent_abstain"]}


def home_row(c: dict, harm: int, label_ids: list[int]) -> dict:
    return {"row_type": "corpus_home", "image": c["home"], "instruction": c["text"], "label_ids": list(label_ids),
            "task_id": c["task_id"], "pool": f"G14_{c['source']}", "token": None, "raw_instruction": c["text"],
            "frame_source": "home", "gripper": "open", "harm": int(harm), "label_source": SRC_HOLD if harm else SRC_GEN,
            "align_key": f"{c['corpus_id']}:home", **corpus_fields(c)}


def mid_row(c: dict, harm: int, holds: dict, grip: dict, gen: dict) -> dict:
    """Hold -> the hold chunk of the frame's gripper (recomputed map); execute -> the base policy's own action on
    that frame."""
    g = grip[c["mid"]]
    ids = holds[g]["gen_ids"] if harm else gen[(c["mid"], c["text"])]
    return {"row_type": "corpus_mid", "image": c["mid"], "instruction": c["text"], "label_ids": list(ids),
            "task_id": c["task_id"], "pool": f"G14_{c['source']}", "token": None, "raw_instruction": c["text"],
            "frame_source": "replay_mid", "gripper": g, "harm": int(harm),
            "label_source": SRC_HOLD if harm else SRC_GEN, "align_key": f"{c['corpus_id']}:mid", **corpus_fields(c)}


def arm_rows(arm: str, common: list[dict], items: list[dict], gen: dict, holds: dict, grip: dict,
             verdict_ids: dict) -> list[dict]:
    """Common rows, then every corpus sentence on its home frame, then every corpus sentence on its mid frame;
    labels per arm (v2 mid rows)."""
    rows = [with_verdict(arm, {**r, "label_source": SRC_FREE300}, verdict_ids) for r in common]
    for c in items:
        lab = arm_label(arm, c)
        ids = holds["open"]["gen_ids"] if lab else gen[(c["home"], c["text"])]
        rows.append(with_verdict(arm, home_row(c, lab, ids), verdict_ids))
    for c in items:
        rows.append(with_verdict(arm, mid_row(c, arm_label(arm, c), holds, grip, gen), verdict_ids))
    return rows


def needed_generations(items: list[dict]) -> list[tuple[str, str]]:
    """(frame, sentence) for the home AND the mid frame of every sentence that is benign under the construction OR
    the parent label: every execute row of every arm."""
    return sorted({(f, c["text"]) for c in items if not c["label_construction"] or not c["label_parent"]
                   for f in (c["home"], c["mid"])})


def check_no_conflict(name: str, rows: list[dict]) -> None:
    """No sentence may be labelled hold and execute in the same file."""
    lab = defaultdict(set)
    for r in rows:
        lab[r["raw_instruction"]].add(int(r["harm"]))
    clash = [sha(t) for t, v in lab.items() if len(v) > 1]
    if clash:
        raise SystemExit(f"{name}: {len(clash)} sentences are labelled both hold and execute (sha1 {clash[:3]})")


def assert_arm_diffs(files: dict, items: list[dict], verdict_ids: dict, holds: dict) -> dict:
    """The four files differ only where the design says. Raises SystemExit on the first violation."""
    i0, i1, i2, i3 = (files[a] for a in ARMS)
    if i3 != i0:
        raise SystemExit("I3 rows differ from I0 rows; the spec makes them identical (the alignment term is in "
                         "training)")
    for name, rows in (("I0", i0), ("I1", i1), ("I2", i2)):
        check_no_conflict(name, rows)
        for r in rows:
            base = r["label_ids"]
            if name in VERDICT_ARMS:
                v = r.get("verdict")
                if v != (STOP if r["harm"] else GO) or base[:len(verdict_ids[v])] != verdict_ids[v]:
                    raise SystemExit(f"{name}: row {r.get('corpus_id', r['image'])} has verdict {v} for harm "
                                     f"{r['harm']} or does not start with its ids")
                base = base[len(verdict_ids[v]):]
            elif "verdict" in r:
                raise SystemExit(f"{name}: a row carries a verdict")
            if r["harm"] and base != holds[r["gripper"]]["gen_ids"]:
                raise SystemExit(f"{name}: a hold row's action ids are not the {r['gripper']} hold chunk")
            if (r["label_source"] == SRC_HOLD) != bool(r["harm"]):
                raise SystemExit(f"{name}: row {r.get('corpus_id', r['image'])} has label source "
                                 f"{r['label_source']} for label {r['harm']}")
    if len(i1) != len(i0):
        raise SystemExit(f"I1 has {len(i1)} rows, I0 {len(i0)}")
    for r0, r1 in zip(i0, i1):
        strip1 = {k: v for k, v in r1.items() if k not in ("label_ids", "verdict")}
        strip0 = {k: v for k, v in r0.items() if k != "label_ids"}
        if strip1 != strip0 or r1["label_ids"] != verdict_ids[r1["verdict"]] + r0["label_ids"]:
            raise SystemExit(f"I1 row {r0.get('corpus_id', r0['image'])} is not the I0 row plus its verdict")
    by = {c["corpus_id"]: c for c in items}
    for a, rows in (("I1", i1), ("I2", i2)):
        for r in rows:
            if "corpus_id" in r and r["harm"] != arm_label(a, by[r["corpus_id"]]):
                raise SystemExit(f"{a}: corpus row {r['corpus_id']} carries label {r['harm']}, the spec label is "
                                 f"{arm_label(a, by[r['corpus_id']])}")
    # v2: I1 and I2 hold the same (sentence, frame) pairs in the same order and differ only in labels
    if len(i2) != len(i1):
        raise SystemExit(f"I2 has {len(i2)} rows, I1 {len(i1)}; they must hold the same (sentence, frame) pairs")
    flips = Counter()
    for r1, r2 in zip(i1, i2):
        s1 = {k: v for k, v in r1.items() if k not in LABEL_FIELDS}
        s2 = {k: v for k, v in r2.items() if k not in LABEL_FIELDS}
        if s1 != s2:
            raise SystemExit(f"I1 and I2 differ beyond the label at row {r1.get('corpus_id', r1['image'])} "
                             f"({r1['row_type']}); they must hold the same (sentence, frame) pairs")
        if r1["harm"] == r2["harm"]:
            if r1 != r2:
                raise SystemExit(f"I2 row {r1.get('corpus_id', r1['image'])} ({r1['row_type']}) differs from I1 "
                                 f"without a label change")
        elif "corpus_id" not in r1:
            raise SystemExit(f"a common row changes label between I1 and I2 ({r1['image']})")
        else:
            flips[r1["row_type"]] += 1
    return {"i3_equals_i0": True, "i1_is_i0_plus_verdict": True, "i1_i2_same_pairs": True,
            "i2_label_flips_home": int(flips["corpus_home"]), "i2_label_flips_mid": int(flips["corpus_mid"])}


def check_corpus_rows(files: dict, items: list[dict], mids: dict, grip: dict, gen: dict, holds: dict,
                      verdict_ids: dict) -> dict:
    """v2 invariants, every arm: each corpus sentence has exactly one home row (its home frame, gripper open) and one
    mid row (a replay frame of its nearest task with step >= MID_STEP, index `mid_index`); every corpus row's gripper
    is the gripper map's; a hold row carries the hold chunk of that gripper, an execute row the base policy's own
    action on its own (frame, sentence); no (frame, sentence) pair repeats; and whenever an arm has both classes,
    home AND mid frames each carry hold AND execute rows (frame kind does not predict the class)."""
    allowed = {t: [r["image"] for r in v] for t, v in mids.items()}
    out = {}
    for arm, rows in files.items():
        corp = [r for r in rows if "corpus_id" in r]
        pairs = [(r["image"], r["raw_instruction"]) for r in corp]
        if len(pairs) != len(set(pairs)):
            raise SystemExit(f"{arm}: a (frame, sentence) pair repeats among the corpus rows")
        per = defaultdict(dict)
        for r in corp:
            if r["row_type"] in per[r["corpus_id"]]:
                raise SystemExit(f"{arm}: sentence {r['corpus_id']} has two {r['row_type']} rows")
            per[r["corpus_id"]][r["row_type"]] = r
        cells = Counter()
        for c in items:
            got = per.get(c["corpus_id"], {})
            if set(got) != {"corpus_home", "corpus_mid"}:
                raise SystemExit(f"{arm}: sentence {c['corpus_id']} has rows {sorted(got)}, needs one home and one mid")
            h, m = got["corpus_home"], got["corpus_mid"]
            cand = allowed[c["task_id"]]
            if h["image"] != c["home"] or h["gripper"] != "open" or frame_kind(h["image"]) != "home":
                raise SystemExit(f"{arm}: home row of {c['corpus_id']} is not its open-gripper home frame")
            if m["image"] != cand[mid_index(c["k"], c["label_construction"], len(cand))] or c["mid_step"] < MID_STEP:
                raise SystemExit(f"{arm}: mid row of {c['corpus_id']} is not its step >= {MID_STEP} replay frame")
            for r in (h, m):
                if r["gripper"] != grip[r["image"]]:
                    raise SystemExit(f"{arm}: {r['row_type']} of {c['corpus_id']} records gripper {r['gripper']}, "
                                     f"the map says {grip[r['image']]}")
                act = action_ids(r, verdict_ids)
                want = holds[r["gripper"]]["gen_ids"] if r["harm"] else gen[(r["image"], r["raw_instruction"])]
                if act != list(want):
                    raise SystemExit(f"{arm}: {r['row_type']} of {c['corpus_id']} does not carry the "
                                     f"{'hold chunk of its gripper' if r['harm'] else 'base action of its frame'}")
                cells[(frame_kind(r["image"]), "hold" if r["harm"] else "execute")] += 1
        labels = {int(r["harm"]) for r in corp}
        if labels == {0, 1} and any(cells[(k, c)] == 0 for k in ("home", "mid") for c in ("hold", "execute")):
            raise SystemExit(f"{arm}: frame kind predicts the class among the corpus rows: {dict(cells)}")
        out[arm] = {f"{k}/{c}": int(n) for (k, c), n in sorted(cells.items())}
    return out


def decorate_common(common: list[dict], frac: dict) -> tuple[list[dict], dict]:
    """v4 1b: the paraphrase / novel-benign common rows get the corpus scheme (prefix, per-word manner fractions `frac`,
    benign clause) per row type; replay and trained rows stay verbatim. Labels are unchanged (the rows stay execute);
    the undecorated sentence is kept in `undecorated`."""
    items = [{"base": r["raw_instruction"], "row_type": r["row_type"], "decorate": r["row_type"] in DECOR_COMMON,
              "clause_ok": True, "hkey": f"{r['image']}|{r['raw_instruction']}"} for r in common]
    rates, _ = decorate(items, group_key="row_type", frac=frac)
    out = []
    for r, it in zip(common, items):
        if it["decorate"]:
            r = {**r, "instruction": it["text"], "raw_instruction": it["text"], "undecorated": it["base"],
                 "prefix": (it["prefix"] or split_prefix(it["base"])[0]).strip(), "manner": it["manner"],
                 "clause": it["clause"].strip()}
        out.append(r)
    return out, rates


def near_hold(ids: list[int], holds: dict) -> bool:
    """1f (the review's near-hold execute rows): tokens 2-5 of the action ids equal those of a hold chunk."""
    return any(list(ids[1:5]) == list(h["gen_ids"][1:5]) for h in holds.values())


def stub_labels(jobs: list[tuple[str, str]], replay: list[dict], holds: dict) -> dict:
    """--stub-generator: each (frame, sentence) gets a real base-policy replay label of the free install chosen by stable hash among
    the replay labels that are not near-hold. Stub rows are NOT a result (the action is not this frame's)."""
    cands = [r["label_ids"] for r in replay if not near_hold(r["label_ids"], holds)]
    print(f"[base-gen] --stub-generator: {len(jobs)} labels drawn from {len(cands)} non-near-hold pod-7 replay labels "
          f"by stable hash (NOT this frame's action)", flush=True)
    return {j: list(cands[int(uhash("stub_gen", j[0] + "|" + j[1]) * len(cands))]) for j in jobs}


def balance_tasks(items: list[dict]) -> tuple[list[dict], dict]:
    """1g: within each nearest task, keep the minority class and the largest stable-hash subset of the majority that
    puts the home-row harm fraction (construction labels) inside TASK_BAND; the ten trained instructions are never
    dropped. A kept sentence keeps both its rows."""
    lo, hi = TASK_BAND
    keep, report = set(), {}
    for t in sorted({c["task_id"] for c in items}):
        cs = [c for c in items if c["task_id"] == t]
        h = [c for c in cs if c["label_construction"]]
        b = [c for c in cs if not c["label_construction"]]
        nh, nb = len(h), len(b)
        key = lambda c: uhash("balance", c["text"])  # noqa: E731
        if nh + nb and nh / (nh + nb) > hi:
            h = sorted(h, key=key)[:int(np.floor(nb * hi / (1 - hi) + 1e-9))]
        elif nh + nb and nh / (nh + nb) < lo:
            prot = [c for c in b if c["source"] == "trained"]
            rest = sorted((c for c in b if c["source"] != "trained"), key=key)
            b = prot + rest[:max(0, int(np.floor(nh * (1 - lo) / lo + 1e-9)) - len(prot))]
        keep |= {c["corpus_id"] for c in h + b}
        report[t] = {"before": [nh, nb], "after": [len(h), len(b)],
                     "harm_fraction": len(h) / max(1, len(h) + len(b))}
    return [c for c in items if c["corpus_id"] in keep], report


def all_row_cues(rows: list[dict]) -> tuple[dict, dict]:
    """v4 1c over the given rows (the caller passes an arm's rows minus the trained instructions): ({bucket: [P(hold),
    n]} for LENGTH_BUCKETS, {cue: {'with': [P(hold), n],
    'without': [...], 'gap': |difference|}} for prefix, benign clause and any manner word). Word counts and cue
    detection are the review script's (norm, leading PREFIX, trailing BENIGN_CLAUSE)."""
    norm = review_core()[0]
    c = cue_rates(rows, "raw_instruction", "harm")
    y = np.asarray([int(r["harm"]) for r in rows])
    b = np.asarray([bucket_of(len(norm(r["raw_instruction"]).split())) for r in rows])
    buckets = {name: [float(y[b == name].mean()) if (b == name).any() else float("nan"), int((b == name).sum())]
               for _, _, name in LENGTH_BUCKETS}
    man = np.asarray([bool(re.search(MANNER_PRESENT_RE, norm(r["raw_instruction"]))) for r in rows])
    cues = {k: c[k] for k in ("prefix", "clause")}
    cues["manner"] = {"with": [float(y[man].mean()) if man.any() else float("nan"), int(man.sum())],
                      "without": [float(y[~man].mean()) if (~man).any() else float("nan"), int((~man).sum())]}
    for v in cues.values():
        v["gap"] = abs(v["with"][0] - v["without"][0]) if v["with"][1] and v["without"][1] else float("nan")
    return buckets, cues


def structural_checks(files: dict, strict: bool) -> dict:
    """3: per arm, closed-gripper holds > 0; no image with >= MIN_IMAGE_ROWS rows carries only holds; the corpus mid
    rows' hold share inside MID_HOLD_BAND; for the construction-label arms, the per-task home-row harm fraction
    inside TASK_BAND and |hold rate prefixed - unprefixed| <= PREFIX_TOL on the corpus home rows; and v4 1c over the
    non-trained-instruction rows of every arm: P(hold) inside BUCKET_BAND in each length bucket with >= BUCKET_MIN_ROWS rows, and
    |P(hold | cue) - P(hold | no cue)| <= DECOR_TOL for prefix, benign clause and manner word. Raises when `strict`."""
    norm = review_core()[0]
    trained = {norm(t) for t in LIBERO_GOAL_TASKS}
    out, bad = {}, []
    for arm, rows in files.items():
        buckets, cues = all_row_cues([r for r in rows if norm(r["raw_instruction"]) not in trained])
        for name, (p, n) in buckets.items():
            if n >= BUCKET_MIN_ROWS and not BUCKET_BAND[0] <= p <= BUCKET_BAND[1]:
                bad.append(f"{arm}: length bucket {name} has P(hold) {p:.3f} over {n} rows, outside {BUCKET_BAND}")
        for k, v in cues.items():
            if v["with"][1] and v["without"][1] and v["gap"] > DECOR_TOL:      # a cue absent on one side: no gap
                bad.append(f"{arm}: P(hold) with / without {k} over all rows differ by {v['gap']:.3f} > {DECOR_TOL}")
        img = defaultdict(list)
        for r in rows:
            img[r["image"]].append(int(r["harm"]))
        big = {i: v for i, v in img.items() if len(v) >= MIN_IMAGE_ROWS}
        mid = [int(r["harm"]) for r in rows if r["row_type"] == "corpus_mid"]
        home = [r for r in rows if r["row_type"] == "corpus_home"]
        tasks = defaultdict(list)
        for r in home:
            tasks[r["task_id"]].append(int(r["harm"]))
        kind = defaultdict(list)
        for r in rows:
            kind[frame_kind(r["image"])].append(int(r["harm"]))
        pre = cue_rates(home, "raw_instruction", "harm")["prefix"] if home else None
        s = {"closed_holds": sum(1 for r in rows if r["harm"] and r["gripper"] == "closed"),
             "images_ge_min": len(big), "hold_only_images": sum(1 for v in big.values() if all(v)),
             "corpus_mid_hold_share": float(np.mean(mid)) if mid else float("nan"),
             "hold_share_by_frame_kind_all_rows": {k: float(np.mean(v)) for k, v in sorted(kind.items())},
             "task_harm_fraction": {int(t): float(np.mean(v)) for t, v in sorted(tasks.items())},
             "prefix_hold_rate": pre,
             "prefix_gap": abs(pre["with"][0] - pre["without"][0]) if pre and pre["with"][1] and pre["without"][1]
             else float("nan"),
             "all_rows_length_buckets": buckets, "all_rows_cues": cues,
             "corpus_bare_fraction": {lab: float(np.mean([is_bare(r["raw_instruction"]) for r in home
                                                          if int(r["label_construction"]) == lab]))
                                      for lab in (0, 1) if any(int(r["label_construction"]) == lab for r in home)}}
        out[arm] = s
        if s["closed_holds"] == 0:
            bad.append(f"{arm}: no closed-gripper hold row")
        if s["hold_only_images"]:
            bad.append(f"{arm}: {s['hold_only_images']} images with >= {MIN_IMAGE_ROWS} rows carry only holds")
        if not MID_HOLD_BAND[0] <= s["corpus_mid_hold_share"] <= MID_HOLD_BAND[1]:
            bad.append(f"{arm}: corpus mid hold share {s['corpus_mid_hold_share']:.3f} outside {MID_HOLD_BAND}")
        if arm not in PARENT_LABEL_ARMS:
            off = {t: f for t, f in s["task_harm_fraction"].items() if not TASK_BAND[0] <= f <= TASK_BAND[1]}
            if off:
                bad.append(f"{arm}: tasks {sorted(off)} have a home-row harm fraction outside {TASK_BAND}")
            if s["prefix_gap"] == s["prefix_gap"] and s["prefix_gap"] > PREFIX_TOL:    # NaN: one side empty
                bad.append(f"{arm}: prefixed vs unprefixed hold rate differ by {s['prefix_gap']:.3f} > {PREFIX_TOL}")
    if bad and strict:
        raise SystemExit("structural invariants fail: " + "; ".join(bad))
    for arm in out:
        out[arm]["violations"] = [b for b in bad if b.startswith(arm + ":")]
    return out


def arm_stats(rows: list[dict]) -> dict:
    c = Counter()
    for r in rows:
        if r["row_type"] in COMMON_TYPES:
            c[r["row_type"]] += 1
        elif r["row_type"] == "corpus_home":
            c["home_hold" if r["harm"] else "home_execute"] += 1
        elif r["row_type"] == "corpus_mid":
            c["mid_hold" if r["harm"] else "mid_execute"] += 1
        else:
            raise SystemExit(f"unknown row type {r['row_type']}")
    v = Counter(r.get("verdict") for r in rows)
    hold = sum(int(r["harm"]) for r in rows)
    corpus_ids = [r for r in rows if r["row_type"] == "corpus_home"]
    mid = [r for r in rows if r["row_type"] == "corpus_mid"]
    return {**{k: int(c[k]) for k in TABLE_COLS}, "total": len(rows), "hold": hold, "execute": len(rows) - hold,
            "open_holds": sum(1 for r in rows if r["harm"] and r["gripper"] == "open"),
            "closed_holds": sum(1 for r in rows if r["harm"] and r["gripper"] == "closed"),
            "mid_rows_gripper": {f"{'hold' if h else 'execute'}/{g}": sum(1 for r in mid if r["harm"] == h and
                                                                          r["gripper"] == g)
                                 for h in (1, 0) for g in ("open", "closed")},
            "stop": int(v[STOP]), "go": int(v[GO]),
            "corpus_harm_sentences": sum(int(r["harm"]) for r in corpus_ids),
            "corpus_benign_sentences": sum(1 - int(r["harm"]) for r in corpus_ids),
            "n_frames": len({r["image"] for r in rows}), "n_sentences": len({r["raw_instruction"] for r in rows})}


def row_table(files: dict, rep: Report) -> dict:
    stats = {arm: arm_stats(rows) for arm, rows in files.items()}
    head = ("replay", "para", "novel", "trained", "home_hold", "home_exec", "mid_hold", "mid_exec")
    rep(f"{'arm':<4s}{'name':<16s}" + "".join(f"{h:>10s}" for h in head)
        + f"{'total':>7s}{'hold':>7s}{'execute':>8s}{'open_holds':>11s}{'closed_holds':>13s}{'[STOP]':>8s}{'[GO]':>7s}")
    for arm, s in stats.items():
        rep(f"{arm:<4s}{ARM_NAME[arm]:<16s}" + "".join(f"{s[k]:10d}" for k in TABLE_COLS)
            + f"{s['total']:7d}{s['hold']:7d}{s['execute']:8d}{s['open_holds']:11d}{s['closed_holds']:13d}"
            + (f"{s['stop']:8d}{s['go']:7d}" if arm in VERDICT_ARMS else f"{'-':>8s}{'-':>7s}"))
    for arm, s in stats.items():
        rep(f"[rows] {arm}: corpus sentences {s['corpus_harm_sentences']} harm / {s['corpus_benign_sentences']} "
            f"benign; mid rows by label/gripper " + ", ".join(f"{k} {v}" for k, v in s["mid_rows_gripper"].items())
            + f"; {s['n_frames']} distinct frames, {s['n_sentences']} distinct sentences")
    return stats


def agreement_table(items: list[dict], rep: Report) -> dict:
    """Parent hard label vs construction label per source AND construction label. (The G6 teacher table compares a
    mixed-label source such as clause_pair with its first row's label only.)"""
    by = defaultdict(list)
    for c in items:
        by[(c["source"], c["label_construction"])].append(c)
    rep(f"I2 label (parent hard label s_parent > 0; construction label where |s_parent| < {ABSTAIN}) vs construction "
        f"label, per source and construction label:")
    rep(f"  {'source':<24s}{'label':>6s}{'rows':>6s}{'abstain':>9s}{'I2_harm':>9s}{'agree':>7s}")
    out = {}
    for (s, lab), cs in sorted(by.items()):
        ph = float(np.mean([c["label_parent"] for c in cs]))
        ab = int(sum(c["parent_abstain"] for c in cs))
        ag = float(np.mean([c["label_parent"] == lab for c in cs]))
        out[f"{s}/{lab}"] = {"n": len(cs), "abstain": ab, "parent_harm": ph, "agree": ag}
        rep(f"  {s:<24s}{lab:6d}{len(cs):6d}{ab:9d}{ph:9.3f}{ag:7.3f}")
    h2b = sum(1 for c in items if c["label_construction"] and not c["label_parent"])
    b2h = sum(1 for c in items if not c["label_construction"] and c["label_parent"])
    ag = float(np.mean([c["label_parent"] == c["label_construction"] for c in items]))
    rep(f"  overall agreement {ag:.3f} over {len(items)} sentences ({sum(c['parent_abstain'] for c in items)} "
        f"abstentions); construction-harm -> I2-benign {h2b}, construction-benign -> I2-harm {b2h}")
    out["overall"] = {"agree": ag, "harm_to_benign": h2b, "benign_to_harm": b2h, "n": len(items),
                      "abstain": int(sum(c["parent_abstain"] for c in items))}
    return out


# --------------------------------------------------------------------------------- review files (before any training)

def leak_table(rows: list[dict], key_map: dict, R, tasks_norm: set) -> Counter:
    """(test pool, 'common' | 'corpus') -> distinct training sentences of the file whose text, review core or deep
    core equals a held-out sentence or core (a sentence that is a trained instruction is allowed)."""
    out = Counter()
    for g, texts in (("common", {r["raw_instruction"] for r in rows if "corpus_id" not in r}),
                     ("corpus", {r["raw_instruction"] for r in rows if "corpus_id" in r})):
        for t in texts:
            for p in leak_hits(t, key_map, R, tasks_norm):
                out[(p, g)] += 1
    return out


def leak_violations(leak: Counter) -> dict:
    """0 is required everywhere except common rows vs calibration: G4 calibrates on the benign TRAINING sentences
    (verify_parent_gated_judge.calib_sentences = BENIGN_PARA_TRAIN + novel benign + trained), which ARE the paraphrase and
    novel-benign common rows; G14 sets no threshold, so that overlap measures nothing."""
    return {f"{p}/{g}": n for (p, g), n in leak.items() if n and not (p.startswith("CALIB/") and g == "common")}


def label_behaviour(policy, ids: list[int], suite: str) -> str:
    """`first_call_token_check.classify` of a label's action ids, decoded as `NoraPolicy.act` decodes a generation."""
    from first_call_token_check import classify
    from refusal.nora_policy import ACTION_MAX, ACTION_MIN
    ft = policy.nora.fast_tokenizer
    act = [t - ACTION_MIN for t in ids if ACTION_MIN <= t <= ACTION_MAX]
    try:
        if not act or len(ft.bpe_tokenizer.decode(act)) != int(ft.time_horizon) * int(ft.action_dim):
            return "invalid"
        norm = np.asarray(ft.decode([act]), dtype=np.float64)
    except Exception:
        return "invalid"
    low, high = (np.asarray(v, dtype=np.float64) for v in policy.nora.libero_keys[suite])
    return classify({"action": (0.5 * (norm + 1.0) * (high - low) + low).reshape(-1, 7), "invalid": False})


def action_ids(r: dict, verdict_ids: dict) -> list[int]:
    return r["label_ids"][len(verdict_ids[r["verdict"]]):] if "verdict" in r else r["label_ids"]


def review(arm: str, rows: list[dict], holds: dict, verdict_ids: dict, key_map: dict, R, tasks_norm: set,
           beh: dict | None, struct: dict | None = None) -> tuple[list[str], dict]:
    """The per-arm review block (the review requirement on every training set, the v3 review items): rows per row type x
    label x frame kind x gripper, first label token / verdict per label, distinct label sequences, label consistency
    by provenance, core leakage, cue and balance statistics, the structural invariants. Raises on a construction
    error; returns (lines, numbers)."""
    L: list[str] = []
    hold_rows = [r for r in rows if r["harm"]]
    exec_rows = [r for r in rows if not r["harm"]]
    L.append("1. rows per row type x label x frame kind x gripper")
    c = Counter((r["row_type"], "hold" if r["harm"] else "execute", r["frame_source"], r["gripper"]) for r in rows)
    L.append(f"  {'row_type':<22s}{'label':<9s}{'frame_source':<14s}{'gripper':<9s}{'rows':>7s}")
    for k in sorted(c):
        L.append(f"  {k[0]:<22s}{k[1]:<9s}{k[2]:<14s}{k[3]:<9s}{c[k]:7d}")
    L.append(f"  total {len(rows)}: hold {len(hold_rows)}, execute {len(exec_rows)}")
    fk = Counter((frame_kind(r["image"]), "hold" if r["harm"] else "execute") for r in rows)
    fkc = Counter((frame_kind(r["image"]), "hold" if r["harm"] else "execute") for r in rows if "corpus_id" in r)
    L.append("  rows by (frame kind, class): " + ", ".join(f"{k}/{c} {fk[(k, c)]}" for k in ("home", "mid")
                                                          for c in ("hold", "execute"))
             + "; corpus rows only: " + ", ".join(f"{k}/{c} {fkc[(k, c)]}" for k in ("home", "mid")
                                                  for c in ("hold", "execute")))
    fkg = Counter((frame_kind(r["image"]), "hold" if r["harm"] else "execute", r["gripper"]) for r in rows)
    L.append("  rows by (frame kind, class, gripper): " + ", ".join(
        f"{k}/{cl}/{g} {fkg[(k, cl, g)]}" for k in ("home", "mid") for cl in ("hold", "execute")
        for g in ("open", "closed")))
    nt = [r for r in rows if frame_kind(r["image"]) == "mid" and R[0](r["raw_instruction"]) not in tasks_norm]
    ntc = Counter(r["row_type"] for r in nt if not r["harm"])
    L.append(f"  non-trained-instruction mid rows: hold {sum(1 for r in nt if r['harm'])} vs execute "
             f"{sum(1 for r in nt if not r['harm'])} (execute by row type: "
             + ", ".join(f"{k} {v}" for k, v in sorted(ntc.items())) + ")")
    mg = Counter((("hold" if r["harm"] else "execute"), r["gripper"]) for r in rows if r["row_type"] == "corpus_mid")
    L.append("  corpus mid rows by class and gripper: " + ", ".join(f"{c}/{g} {mg[(c, g)]}" for c in ("hold", "execute")
                                                                   for g in ("open", "closed")))
    L.append("2. first label token and verdict, per label")
    first = {}
    for cls, sub in (("hold", hold_rows), ("execute", exec_rows)):
        f = Counter(int(r["label_ids"][0]) for r in sub)
        first[cls] = {"distinct": len(f), "top": [[i, n] for i, n in f.most_common(5)]}
        L.append(f"  {cls:<8s} first label token: {len(f)} distinct; top " + ", ".join(f"{i} x{n}" for i, n in
                                                                                       f.most_common(5)))
    verdicts = None
    if arm in VERDICT_ARMS:
        vt = Counter(("hold" if r["harm"] else "execute", r["verdict"]) for r in rows)
        verdicts = {f"{k[0]}/{k[1]}": int(n) for k, n in vt.items()}
        L.append(f"  verdict by label: hold -> {STOP} {vt[('hold', STOP)]}, {GO} {vt[('hold', GO)]}; execute -> "
                 f"{GO} {vt[('execute', GO)]}, {STOP} {vt[('execute', STOP)]}; verdict ids {verdict_ids}")
        for cls, sub in (("hold", hold_rows), ("execute", exec_rows)):
            f = Counter(int(action_ids(r, verdict_ids)[0]) for r in sub)
            L.append(f"  {cls:<8s} first action token after the verdict: {len(f)} distinct; top "
                     + ", ".join(f"{i} x{n}" for i, n in f.most_common(5)))
    distinct = {cls: len({tuple(r["label_ids"]) for r in sub}) for cls, sub in (("hold", hold_rows),
                                                                                 ("execute", exec_rows))}
    L.append(f"3. distinct label_ids sequences: hold {distinct['hold']}, execute {distinct['execute']}")
    hold_set = {tuple(h["gen_ids"]) for h in holds.values()}
    harm_nonhold = sum(1 for r in hold_rows if action_ids(r, verdict_ids) != holds[r["gripper"]]["gen_ids"]
                       or r["label_source"] != SRC_HOLD)
    ben_assigned = sum(1 for r in exec_rows if r["label_source"] == SRC_HOLD)
    ben_same = Counter(f"{r['row_type']}/{r['label_source']}" for r in exec_rows
                       if tuple(action_ids(r, verdict_ids)) in hold_set)
    L.append("4. label consistency")
    L.append(f"  hold-labelled rows whose action ids are not the hold chunk of their gripper: {harm_nonhold} "
             f"(expected 0)")
    L.append(f"  execute-labelled rows assigned a hold chunk by the builder: {ben_assigned} (expected 0)")
    L.append(f"  execute-labelled rows whose base-policy label is identical to a hold chunk: {sum(ben_same.values())}"
             + (" (" + ", ".join(f"{k} {v}" for k, v in sorted(ben_same.items())) + ")" if ben_same else "")
             + " (reported: the base policy itself emitted the hold ids)")
    ben_beh = None
    if beh is None:
        L.append("  execute-labelled rows whose label DECODES to a hold (f1_token_check.classify): not computed "
                 "(no FAST tokenizer on this machine)")
    else:
        bb = Counter(f"{r['row_type']}/{beh[tuple(action_ids(r, verdict_ids))]}" for r in exec_rows)
        ben_beh = {k: int(v) for k, v in sorted(bb.items())}
        n_hold = sum(v for k, v in bb.items() if k.endswith("/hold"))
        L.append(f"  execute-labelled rows whose label DECODES to a hold (|motion| < 0.05, gripper open; "
                 f"f1_token_check.classify): {n_hold} (reported); by row type and class: "
                 + ", ".join(f"{k} {v}" for k, v in sorted(bb.items())))
    leak = leak_table(rows, key_map, R, tasks_norm)
    L.append("5. leakage: distinct training sentences whose text, prefix/clause-stripped core (g14_review_rows.core) or "
             "manner-stripped core equals a held-out sentence or core (a trained instruction is allowed)")
    pools_hit = sorted({p for p, _ in leak})
    for p in pools_hit:
        L.append(f"  {p:<28s} common {leak[(p, 'common')]:5d}   corpus {leak[(p, 'corpus')]:5d}"
                 + ("   by design (calibration = the benign training sentences; G14 sets no threshold)"
                    if p.startswith("CALIB/") else ""))
    bad = leak_violations(leak)
    L.append(f"  leakage violations: {bad if bad else 'none'} (0 required everywhere except common rows vs calibration)")
    home = [r for r in rows if r["row_type"] == "corpus_home"]
    cues = cue_rates(home, "raw_instruction", "harm") if home else {}
    words_ = predictive_words(home, "raw_instruction", "harm") if home else []
    L.append("6. cues and balance, corpus home rows (one per sentence), this arm's labels")
    for k, v in cues.items():
        if k != "length":
            L.append(f"  hold rate with {k:<18s} {v['with'][0]:.3f} (n={v['with'][1]:4d})   without "
                     f"{v['without'][0]:.3f} (n={v['without'][1]:4d})")
    if cues:
        L.append("  hold rate by length bucket (words): " + ", ".join(f"{b} {x[0]:.3f} (n={x[1]})"
                                                                     for b, x in cues["length"].items()))
    tk = defaultdict(list)
    for r in home:
        tk[r["task_id"]].append(int(r["harm"]))
    L.append("  hold rate by nearest task: " + ", ".join(f"t{t} {np.mean(v):.3f} (n={len(v)})" for t, v in sorted(tk.items())))
    L.append("  20 most class-predictive words (g14_review_rows metric, words in >= 20 rows): " + ", ".join(
        f"{w} {ratio:.1f}x->{side}" for w, ratio, side in words_))
    if struct is not None:
        s = struct
        L.append(f"  structural: closed-gripper holds {s['closed_holds']} (> 0 required); images with >= "
                 f"{MIN_IMAGE_ROWS} rows {s['images_ge_min']}, hold-only among them {s['hold_only_images']} (0 required); "
                 f"corpus mid hold share {s['corpus_mid_hold_share']:.3f} (in {MID_HOLD_BAND} required); hold share by "
                 f"frame kind, all rows: " + ", ".join(f"{k} {v:.3f}" for k, v in s["hold_share_by_frame_kind_all_rows"].items())
                 + f"; prefix gap {s['prefix_gap']:.3f}" + (f" (<= {PREFIX_TOL} required)" if arm not in PARENT_LABEL_ARMS
                                                            else " (reported only: parent labels)")
                 + f"; violations: {s['violations'] or 'none'}")
        L.append(f"7. ALL rows of this arm whose text is not a trained instruction (common rows included): P(hold) by "
                 f"length bucket (in {BUCKET_BAND} required "
                 f"where n >= {BUCKET_MIN_ROWS}) and with / without each cue (gap <= {DECOR_TOL} required)")
        L.append("  " + ", ".join(f"{k} {p:.3f} (n={n})" for k, (p, n) in s["all_rows_length_buckets"].items()))
        for k, v in s["all_rows_cues"].items():
            L.append(f"  {k:<8s} with {v['with'][0]:.3f} (n={v['with'][1]})   without {v['without'][0]:.3f} "
                     f"(n={v['without'][1]})   gap {v['gap']:.3f}")
        L.append("  corpus bare fraction (no prefix, clause or manner word) by construction label: " + ", ".join(
            f"label {k} {v:.3f}" for k, v in s["corpus_bare_fraction"].items()))
    if harm_nonhold or ben_assigned or bad:
        raise SystemExit(f"{arm} review: {harm_nonhold} hold rows without the hold chunk, {ben_assigned} execute "
                         f"rows assigned a hold chunk, leakage {bad}")
    return L, {"rows": {"/".join(k): int(n) for k, n in sorted(c.items())},
               "frame_kind_class": {f"{k}/{cl}": int(n) for (k, cl), n in sorted(fk.items())},
               "frame_kind_class_gripper": {f"{k}/{cl}/{g}": int(n) for (k, cl, g), n in sorted(fkg.items())},
               "non_trained_mid": {"hold": sum(1 for r in nt if r["harm"]), "execute": sum(1 for r in nt if not r["harm"])},
               "corpus_mid_gripper": {f"{cl}/{g}": int(n) for (cl, g), n in sorted(mg.items())}, "first_token": first,
               "verdicts": verdicts, "distinct_label_sequences": distinct, "hold_rows_not_hold_chunk": harm_nonhold,
               "execute_rows_assigned_hold": ben_assigned, "execute_rows_equal_hold_chunk": dict(ben_same),
               "execute_rows_label_behaviour": ben_beh, "leakage": {f"{p}/{g}": n for (p, g), n in leak.items()},
               "cue_rates": cues, "predictive_words": words_,
               "task_hold_rate": {int(t): float(np.mean(v)) for t, v in tk.items()}, "structural": struct}


def write_reviews(P: dict, files: dict, holds: dict, verdict_ids: dict, key_map: dict, R, tasks_norm: set,
                  beh: dict | None, struct: dict, header: list[str]) -> dict:
    """<rows>/train_<name>.review.txt next to each row file; returns the numbers for the build summary."""
    out = {}
    for arm in ARMS:
        path = P["file"][arm]
        lines, nums = review(arm, files[arm], holds, verdict_ids, key_map, R, tasks_norm, beh, struct[arm])
        head = [f"G14 training-data review: arm {arm} ({ARM_NAME[arm]}), file {path.name}, {len(files[arm])} rows, "
                f"sha1 {sha1_file(path)}",
                "Sentence texts are not printed here; the row file holds them (field raw_instruction)."] + header
        rp = path.with_name(path.name.replace(".jsonl", ".review.txt"))
        rp.write_text("\n".join(head + [""] + lines) + "\n")
        out[arm] = {"file": str(rp), **nums}
    return out


def nora_mean_words(policy, rows: list[dict], f1: Path, rows_sha1: str, path: Path, rep: Report) -> dict:
    """v4 (review M-e): mu_NORA, the mean over the I3 corpus rows of base NORA's mean-over-instruction-words residual
    (`NoraPolicy.prefill`, position mean_words) at ALIGN_LAYERS; stored with the rows sha1 for teacher_align, which puts
    it next to the parent targets so train_install_lora centres both sides. --stub-generator: seeded Gaussian, NOT a result."""
    corp = [r for r in rows if r.get("align_key")]
    if policy is None:
        rng = np.random.default_rng(0)
        mu = {li: rng.standard_normal(HIDDEN).astype(np.float32) for li in ALIGN_LAYERS}
        rep(f"[mu_nora] --stub-generator: mu_NORA fabricated (seeded Gaussian) for {len(corp)} corpus rows. NOT a result.")
    else:
        from refusal.nora_policy import MEAN_WORDS
        from build_free_install_data import load_png
        acc = {li: np.zeros(HIDDEN, dtype=np.float64) for li in ALIGN_LAYERS}
        t0, cache = time.time(), {}
        for k, r in enumerate(corp):
            if r["image"] not in cache:
                if len(cache) >= 256:
                    cache.clear()
                cache[r["image"]] = load_png(f1 / r["image"])
            h = policy.prefill(cache[r["image"]], r["raw_instruction"], list(ALIGN_LAYERS), [MEAN_WORDS])
            for li in ALIGN_LAYERS:
                acc[li] += np.asarray(h[(li, MEAN_WORDS)], dtype=np.float64)
            if (k + 1) % 500 == 0:
                print(f"[mu_nora] {k + 1}/{len(corp)} prefills, {time.time() - t0:.0f}s", flush=True)
        mu = {li: (acc[li] / max(1, len(corp))).astype(np.float32) for li in ALIGN_LAYERS}
        rep(f"[mu_nora] base NORA mean-words mean over {len(corp)} corpus rows at layers {list(ALIGN_LAYERS)}, "
            f"{time.time() - t0:.0f}s; |mu| " + ", ".join(f"L{li} {np.linalg.norm(mu[li]):.2f}" for li in ALIGN_LAYERS))
    np.savez(path, layers=np.asarray(ALIGN_LAYERS, dtype=np.int32), rows_sha1=np.asarray(rows_sha1),
             n_rows=np.asarray(len(corp)), stub=np.asarray(int(policy is None)),
             **{f"mu_L{li}": mu[li] for li in ALIGN_LAYERS})
    return mu


def verdict_token_ids(policy, rep: Report) -> dict:
    """The ids of "[STOP]" / "[GO]" (no trailing space) in the loaded policy's tokenizer, printed and asserted equal
    to the record of the parent-gated reference run; they must be plain text tokens (not FAST action ids, not <|im_end|>)."""
    from refusal.models import IM_END
    from refusal.nora_policy import ACTION_MAX, ACTION_MIN
    tok = getattr(policy.processor, "tokenizer", policy.processor)
    out = {}
    for v in (STOP, GO):
        ids = [int(i) for i in tok(v, add_special_tokens=False)["input_ids"]]
        pieces = tok.convert_ids_to_tokens(ids)
        rep(f"[verdict] {v!r} -> {len(ids)} tokens {pieces} ids {ids} (NORA tokenizer, no trailing space)")
        if any(ACTION_MIN <= i <= ACTION_MAX or i == IM_END for i in ids) or tok.decode(ids) != v:
            raise SystemExit(f"{v} tokenizes into action / end ids or does not round-trip: {ids}")
        out[v] = ids
    if out != VERDICT_IDS_REF:
        raise SystemExit(f"verdict ids {out} differ from the pod-8b record {VERDICT_IDS_REF}")
    return out


def stage_rows(a) -> int:
    out = Path(a.out)
    P = paths(out)
    P["rows"].mkdir(parents=True, exist_ok=True)
    rep = Report()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    rep(f"G14 rows stage (v3 corpus, v2 mid rows) ts {stamp}, smoke {int(a.smoke)}, stub generator "
        f"{int(a.stub_generator)}")
    f1 = Path(a.f1_data)
    pools = load_json(f1 / "pools.json")
    if list(pools["meta"]["tasks"]) != LIBERO_GOAL_TASKS:
        raise SystemExit("pools.json meta.tasks is not the recorded libero_goal order")
    d, teacher = load_corpus(a, rep)
    free300 = read_jsonl(a.free300)
    replay = read_jsonl(f1 / "replay.jsonl")
    if not free300 or len(replay) != COMMON_COUNTS["replay"]:
        raise SystemExit(f"{a.free300}: {len(free300)} rows; {f1 / 'replay.jsonl'}: {len(replay)} rows (need "
                         f"{COMMON_COUNTS['replay']}); fetch pod-8b g1_full data and the pod-7 f1_full data")
    R = review_core()
    tasks_norm = {R[0](t) for t in d["tasks"]}
    tests = test_sentences(pools, Path(a.twins), Path(a.demo_prompts), Path(a.novel_benign))
    key_map_all = test_key_map(tests, R, tasks_norm, with_calib=True)
    key_map_common = test_key_map(tests, R, tasks_norm, with_calib=False)
    nidx = near_index(tests, tasks_norm)
    common = common_rows(free300, a.smoke, key_map_common, R, tasks_norm, nidx, rep)
    homes = corpus_frames(d, f1, Path(a.eval_frames))
    mids = mid_frames(replay)
    keep = None
    if a.smoke:
        import single_model_judges as G
        keep = {r["id"] for r in G.smoke_rows(d["rows"])}
    items = corpus_items(d, teacher, homes, mids, keep)
    rep(f"corpus sentences: {len(items)}" + (" (smoke: g6 smoke_rows per source, frames from the full corpus)"
                                             if a.smoke else "")
        + f"; home frames {len({c['home'] for c in items})} distinct (G6 frame_jobs); mid frames "
          f"{len({c['mid'] for c in items})} distinct, drawn from the replay frames with step >= {MID_STEP} per task "
        + ", ".join(f"t{t} {len(v)}" for t, v in sorted(mids.items())))
    missing = sorted({x for c in items for x in (c["home"], c["mid"]) if not (f1 / x).exists()}
                     | {r["image"] for r in common if not (f1 / r["image"]).exists()})
    if missing:
        raise SystemExit(f"{len(missing)} frames missing under {f1}, e.g. {missing[:3]}")
    agree = agreement_table(items, rep)

    # -- the base policy: hold chunks, gripper map, verdict ids, new generations --------------------------------
    recorded = {r["image"]: r["gripper"] for r in free300 if r["row_type"] == "replay"}
    if set(recorded) != {r["image"] for r in replay}:
        raise SystemExit("the free300 replay rows do not cover every replay frame")
    policy = None
    if a.stub_generator:
        rep("[stub] --stub-generator: no model on this machine. Verdict ids = the pod-8b record, replay grippers = "
            "the pod-8b free300 record, execute labels = pod-7 replay labels drawn by hash (not this frame's action), "
            "closed hold chunk = the open one. NOT a result.")
        verdict_ids = {k: list(v) for k, v in VERDICT_IDS_REF.items()}
        grip = dict(recorded)
    else:
        from single_model_judges import forbid_parent
        from refusal.nora_policy import NoraPolicy
        forbid_parent()
        policy = NoraPolicy(a.checkpoint, unnorm_key=a.suite, time_horizon=a.time_horizon)
        verdict_ids = verdict_token_ids(policy, rep)
        grip = gripper_map(a, policy, replay, set())
        bad = [img for img, g in recorded.items() if grip[img] != g]
        if bad:
            raise SystemExit(f"{len(bad)} replay frames have a gripper state that differs from the pod-8b record, "
                             f"e.g. {bad[:3]}")
        rep(f"[gripper] {len(grip)} replay frames: the recomputed gripper map equals the pod-8b free300 record")
    for c in items:
        grip.setdefault(c["home"], "open")
    holds = hold_chunks(a, policy)

    def classes(labels) -> dict:
        """tuple(action ids) -> first_call_token_check class (GPU machine) or None (no FAST tokenizer here)."""
        if policy is None:
            return {tuple(x): None for x in labels}
        return {tuple(x): label_behaviour(policy, list(x), a.suite) for x in {tuple(y) for y in labels}}

    # 1f on the common rows: drop execute rows whose label is near-hold (tokens 2-5) or decodes to a hold
    cb = classes([r["label_ids"] for r in common])
    nh_common = Counter()
    kept = []
    for r in common:
        tok, cls = near_hold(r["label_ids"], holds), cb[tuple(r["label_ids"])] == "hold"
        if tok or cls:
            nh_common[(r["row_type"], "tokens_2_5" if tok else "classified_hold")] += 1
        else:
            kept.append(r)
    common = kept
    rep("1f common rows dropped (near-hold: tokens 2-5 of a hold chunk, or classified hold"
        + ("" if policy is not None else "; classification not available on this machine") + "): "
        + (", ".join(f"{t}/{k} {n}" for (t, k), n in sorted(nh_common.items())) or "none"))
    # v4 1b: the corpus decoration scheme on the paraphrase / novel-benign rows (labels unchanged; replay kept in full)
    common, crates = decorate_common(common, d["info"]["manner_fraction"])
    for g, r in crates.items():
        rep(f"v4 common-row decoration, {g}: {r['n']} rows; prefix {r['inherent_prefix']} inherent + "
            f"{max(0, round(PREFIX_RATE * r['n']) - r['inherent_prefix'])} added; clause {r['clauses']}; manner added "
            + ", ".join(f"{w} {n}" for w, n in r["manner_added"].items() if n)
            + f" (shortfall {r['manner_shortfall'] or 'none'}); labels unchanged (execute)")
    def common_bad(r):
        return bool(leak_hits(r["raw_instruction"], key_map_common, R, tasks_norm)
                    or near_copies(r["raw_instruction"], nidx, tasks_norm)
                    or is_effector(r["raw_instruction"]))
    # a decoration can recreate a held-out sentence verbatim (held-out paraphrases carry prefixes): drop those rows
    redo = Counter(r["row_type"] for r in common if common_bad(r))
    common = [r for r in common if not common_bad(r)]
    rep("v4 decorated common rows dropped because the decorated text leaks into a test pool or reaches the lexical "
        "Jaccard bar: " + (", ".join(f"{t} {n}" for t, n in redo.items()) or "none"))
    need = needed_generations(items)
    n_need = Counter(frame_kind(f) for f, _ in need)
    rep(f"[gen] {len(need)} (frame, sentence) pairs need a base-policy generation: the home and the mid frame of every "
        f"sentence benign under the construction or the parent label (home {n_need['home']}, mid {n_need['mid']})")
    gen = stub_labels(need, replay, holds) if policy is None else generate_labels(a, policy, need, f1, "base-gen")
    gb = classes(list(gen.values()))
    need_set = set(need)
    why = {}
    for c in items:
        for f in (c["home"], c["mid"]):
            if (f, c["text"]) not in need_set:
                continue
            ids = gen.get((f, c["text"]))
            reason = ("invalid" if ids is None else "tokens_2_5" if near_hold(ids, holds)
                      else "classified_hold" if gb[tuple(ids)] == "hold" else None)
            if reason and c["corpus_id"] not in why:
                why[c["corpus_id"]] = (reason, frame_kind(f))
    dropped = [c for c in items if c["corpus_id"] in why]
    rep(f"[gen] 1f: {len(dropped)} sentences dropped from ALL four files (a needed base generation was invalid, "
        f"near-hold or classified hold): " + (", ".join(
            f"{src}/{r}/{k} {n}" for (src, r, k), n in sorted(Counter((c['source'],) + why[c['corpus_id']]
                                                                      for c in dropped).items())) or "none"))
    items = [c for c in items if c["corpus_id"] not in why]

    # 1g: per-task balance of the home rows (construction labels); a kept sentence keeps its mid row
    items, bal = balance_tasks(items)
    rep(f"1g per-task balance (home-row harm fraction in {TASK_BAND}, construction labels; trained instructions kept): "
        + ", ".join(f"t{t} {v['before'][0]}h/{v['before'][1]}b -> {v['after'][0]}h/{v['after'][1]}b "
                    f"({v['harm_fraction']:.3f})" for t, v in bal.items()))
    # v4 (review M-c): a common row whose (frame, sentence) is also a corpus row is dropped; the corpus row stays
    pairs = {(f, c["text"]) for c in items for f in (c["home"], c["mid"])}
    n_before = len(common)
    common = [r for r in common if (r["image"], r["raw_instruction"]) not in pairs]
    rep(f"v4 duplicate (frame, sentence) pairs: {n_before - len(common)} common rows dropped because a corpus row has the "
        f"same frame and sentence (the corpus row is kept)")

    # -- the four files ---------------------------------------------------------------------------------------
    files = {arm: arm_rows(arm, common, items, gen, holds, grip, verdict_ids) for arm in ARMS}
    eff = {arm: sum(is_effector(r["raw_instruction"]) for r in rows) for arm, rows in files.items()}
    if any(eff.values()):
        raise SystemExit(f"training rows mention the gripper / arm (1a effector regex): {eff}")
    rep(f"1a effector regex: 0 training rows of any row type match, in every arm (asserted)")
    diffs = assert_arm_diffs(files, items, verdict_ids, holds)
    cells = check_corpus_rows(files, items, mids, grip, gen, holds, verdict_ids)
    rep(f"file differences asserted: I3 == I0 row for row; I1 = I0 + the verdict ids on every row; I2 = I1 with the "
        f"same (sentence, frame) pairs in the same order, labels differing on {diffs['i2_label_flips_home']} home and "
        f"{diffs['i2_label_flips_mid']} mid rows (parent label); common rows identical")
    rep("corpus-row invariants asserted (every arm): one home + one mid row per sentence, mid frame = its step >= "
        f"{MID_STEP} replay frame, gripper = the map's, hold = the hold chunk of that gripper, execute = the base action "
        f"of its own (frame, sentence), no repeated pair, both classes on both frame kinds; corpus rows by (frame kind, "
        f"class): " + "; ".join(f"{arm} " + ", ".join(f"{k} {v}" for k, v in c.items()) for arm, c in cells.items()))
    for arm, rows in files.items():
        prs = [(r["image"], r["raw_instruction"]) for r in rows]
        if len(prs) != len(set(prs)):
            raise SystemExit(f"{arm}: {len(prs) - len(set(prs))} (frame, sentence) pairs repeat in the file")
        lx = sum(1 for t in {r["raw_instruction"] for r in rows} if near_copies(t, nidx, tasks_norm))
        if lx:
            raise SystemExit(f"{arm}: {lx} training sentences near-copy a test sentence (checklist B7)")
    rep(f"asserted in every arm: no (frame, sentence) pair repeats; 0 training sentences near-copy a test sentence "
        f"(B7: Jaccard >= {JACCARD_MAX} vs {'/'.join(LEXICAL_POOLS)}, audit-7 distinctive words vs LEXICAL pools, "
        f"Jaccard >= {NEAR_DUP_MAX} vs BLIND and rule pools; trained instructions allowed)")
    rep("")
    stats = row_table(files, rep)
    struct = structural_checks(files, strict=not a.smoke)
    for arm, s in struct.items():
        rep(f"[struct] {arm}: closed-gripper holds {s['closed_holds']}, hold-only images (>= {MIN_IMAGE_ROWS} rows) "
            f"{s['hold_only_images']} of {s['images_ge_min']}, corpus mid hold share {s['corpus_mid_hold_share']:.3f}, "
            f"prefix gap {s['prefix_gap']:.3f}, task harm fractions " + ", ".join(
                f"t{t} {f:.2f}" for t, f in s["task_harm_fraction"].items())
            + "; non-trained-instruction rows P(hold) by length " + ", ".join(f"{k} {p:.2f} (n={n})" for k, (p, n) in
                                                          s["all_rows_length_buckets"].items())
            + "; cue gaps " + ", ".join(f"{k} {v['gap']:.3f}" for k, v in s["all_rows_cues"].items())
            + f"; violations {s['violations'] or 'none'}" + (" (smoke: reported, not enforced)" if a.smoke and
                                                              s["violations"] else ""))
    n_twins = check_twins(Path(a.twins), {arm: files[arm] for arm in ARMS})
    n_labels, beh = None, None
    if policy is not None:
        n_labels = check_labels(policy, files["I0"], a.time_horizon)
        todo = {tuple(action_ids(r, verdict_ids)) for arm in ARMS for r in files[arm] if not r["harm"]}
        beh = {ids: label_behaviour(policy, list(ids), a.suite) for ids in todo}
        rep(f"[labels] {len(beh)} distinct execute labels decoded and classified: "
            + ", ".join(f"{k} {v}" for k, v in sorted(Counter(beh.values()).items())))
    else:
        rep("[labels] SKIPPED: no FAST tokenizer on this machine (--stub-generator)")
    for arm in ARMS:
        write_jsonl(P["file"][arm], files[arm])
    if sha1_file(P["file"]["I3"]) != sha1_file(P["file"]["I0"]):
        raise SystemExit("the I3 and I0 files are not byte-identical")
    # for experiments/4_parent_taught_verdict/review_training_rows.py --hold-closed (it checks closed-gripper holds against this file)
    (P["rows"] / "hold_closed_g14.json").write_text(json.dumps(
        {"gen_ids": holds["closed"]["gen_ids"], "gripper": "closed", "stub": policy is None}, indent=1))
    nora_mean_words(policy, files["I3"], f1, sha1_file(P["file"]["I3"]), P["nora_mu"], rep)
    steps = TRAIN_EPOCHS * (len(files["I0"]) // TRAIN_BATCH)
    epochs = {arm: steps * TRAIN_BATCH / len(files[arm]) for arm in ARMS}
    rep(f"optimizer steps for every arm = {TRAIN_EPOCHS} epochs of the I0 file = {TRAIN_EPOCHS} x "
        f"({len(files['I0'])} // {TRAIN_BATCH}) = {steps}; epochs per arm: "
        + ", ".join(f"{arm} {e:.3f} ({len(files[arm])} rows)" for arm, e in epochs.items()))
    n_mid_closed = sum(1 for c in items if grip[c["mid"]] == "closed")
    header = [f"stub generator: {int(a.stub_generator)}" + (" -- execute labels are pod-7 replay labels drawn by hash, "
                                                            "closed hold chunk = open; NOT a result" if a.stub_generator
                                                            else " (real base-policy labels)")
              + f"; stub teacher {int(teacher['stub'])}; smoke {int(a.smoke)}; v3 corpus, v2 mid rows (every corpus "
                f"sentence on one replay frame with step >= {MID_STEP})",
              f"hold chunk ids: open {holds['open']['gen_ids']}, closed {holds['closed']['gen_ids']}; verdict ids "
              f"{verdict_ids}",
              f"corpus: v3 {d['digest']}, {len(items)} sentences after 1f ({len(dropped)} dropped) and 1g balance; "
              f"{len(need)} new base generations; mid frames of the kept sentences with a closed gripper: "
              f"{n_mid_closed} of {len(items)}",
              f"I2 label vs construction label: agreement {agree['overall']['agree']:.3f} (before 1f / 1g), "
              f"construction-harm -> I2-benign {agree['overall']['harm_to_benign']}, construction-benign -> I2-harm "
              f"{agree['overall']['benign_to_harm']}, abstentions {agree['overall']['abstain']}",
              f"optimizer steps for every arm: {steps} (epochs " + ", ".join(f"{k} {v:.3f}" for k, v in epochs.items())
              + f"); file differences asserted: {diffs}"]
    reviews = write_reviews(P, files, holds, verdict_ids, key_map_all, R, tasks_norm, beh, struct, header)
    rep("[review] " + "; ".join(f"{arm}: " + Path(v["file"]).name for arm, v in reviews.items()))
    for arm, v in reviews.items():
        rep(f"[review] {arm}: execute rows equal to a hold chunk {sum(v['execute_rows_equal_hold_chunk'].values())}, "
            f"decoding to a hold "
            + (str(sum(n for k, n in v['execute_rows_label_behaviour'].items() if k.endswith('/hold')))
               if v["execute_rows_label_behaviour"] is not None else "not computed")
            + f"; distinct label sequences {v['distinct_label_sequences']}; leakage violations none")
    summary = {"exp": f"{EXP}/rows", "ts": stamp, "smoke": bool(a.smoke), "stub_generator": bool(a.stub_generator),
               "stub_teacher": bool(teacher["stub"]), "corpus_digest": d["digest"], "corpus_version": 3,
               "checkpoint": a.checkpoint, "arms": ARM_NAME, "epochs_per_arm": epochs, "structural": struct,
               "balance": {str(t): v for t, v in bal.items()},
               "near_hold_common": {f"{t}/{k}": n for (t, k), n in nh_common.items()},
               "dropped_sentences": {cid: list(w) for cid, w in why.items()},
               "files": {arm: str(P["file"][arm]) for arm in ARMS},
               "rows_sha1": {arm: sha1_file(P["file"][arm]) for arm in ARMS},
               "verdict_ids": verdict_ids, "hold_open_gen_ids": holds["open"]["gen_ids"],
               "hold_closed_gen_ids": holds["closed"]["gen_ids"], "n_corpus_sentences": len(items),
               "n_new_generations": len(need), "n_labels_kept": len(gen), "n_dropped_sentences": len(dropped),
               "steps_all_arms": steps, "train_batch": TRAIN_BATCH, "train_epochs": TRAIN_EPOCHS,
               "mid_step": MID_STEP, "align_layers": list(ALIGN_LAYERS), "rows": stats, "diffs": diffs,
               "agreement": agree, "n_twins_held_out": n_twins, "n_distinct_labels_i0": n_labels,
               "n_mid_closed": n_mid_closed, "corpus_frame_kind_class": cells, "mid_rows_version": 2,
               "n_new_generations_by_kind": dict(n_need), "review": reviews,
               "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}}
    P["summary"].write_text(json.dumps(summary, indent=1))
    P["table"].write_text("\n".join(rep.lines) + "\n")
    print("wrote\n" + "\n".join(file_line(P["file"][arm]) + "\n" + file_line(Path(reviews[arm]["file"])) for arm in ARMS)
          + "\n" + file_line(P["summary"]) + "\n" + file_line(P["table"]), flush=True)
    return 0


# --------------------------------------------------------------------------------- teacher_align (the parent)

def stub_targets(keys: list[str]) -> dict:
    out = {li: np.zeros((len(keys), HIDDEN), dtype=np.float32) for li in ALIGN_LAYERS}
    for i, k in enumerate(keys):
        rng = np.random.default_rng(int(hashlib.sha1(k.encode()).hexdigest()[:8], 16))
        for li in ALIGN_LAYERS:
            out[li][i] = rng.standard_normal(HIDDEN)
    return out


def parent_targets(L, items: list[dict], f1: Path) -> tuple[dict, np.ndarray, float]:
    """One parent forward per row: its robot prompt (`refusal.models.build_inputs`: the 224x224 frame and the raw
    command, the parent's processor), the block output at each ALIGN layer, mean over the instruction-word span
    (`refusal.patching.span_for_mode(..., "words")`), fp32."""
    import PIL.Image
    import torch
    from refusal.models import build_inputs
    from refusal.patching import span_for_mode
    out = {li: np.zeros((len(items), HIDDEN), dtype=np.float32) for li in ALIGN_LAYERS}
    n_span = np.zeros(len(items), dtype=np.int32)
    span_sha: list[str] = []
    state: dict = {}

    def make_hook(li):
        def hook(_m, _i, o):
            h = o[0] if isinstance(o, tuple) else o
            s0, s1 = state["span"]
            state["got"][li] = h[0, s0:s1, :].float().mean(0).cpu().numpy()
        return hook

    cache: dict = {}
    handles = [L.layers[li].register_forward_hook(make_hook(li)) for li in ALIGN_LAYERS]
    t0 = time.time()
    try:
        with torch.inference_mode():
            for k, r in enumerate(items):
                if r["image"] not in cache:
                    if len(cache) >= 256:
                        cache.clear()
                    cache[r["image"]] = PIL.Image.open(f1 / r["image"]).convert("RGB")
                inputs = build_inputs(L, r["raw_instruction"], cache[r["image"]])
                state["span"] = span_for_mode(inputs["input_ids"], L.tokenizer, "words")
                state["got"] = {}
                L.model(**inputs, use_cache=False)
                if set(state["got"]) != set(ALIGN_LAYERS):
                    raise SystemExit(f"row {k}: hooks fired at {sorted(state['got'])}, need {ALIGN_LAYERS}")
                for li in ALIGN_LAYERS:
                    out[li][k] = state["got"][li]
                s0, s1 = state["span"]
                n_span[k] = s1 - s0
                span_sha.append(span_ids_sha1(inputs["input_ids"][0, s0:s1].tolist()))
                if (k + 1) % 500 == 0:
                    print(f"[align] {k + 1}/{len(items)} parent forwards, {time.time() - t0:.0f}s", flush=True)
    finally:
        for h in handles:
            h.remove()
    sec = (time.time() - t0) / max(1, len(items))
    print(f"[align] {len(items)} parent forwards in {time.time() - t0:.0f}s ({sec:.3f} s each)", flush=True)
    return out, n_span, span_sha, sec


def span_ids_sha1(ids) -> str:
    """Hash of the instruction-word token ids; train_install_lora.AlignLoss compares NORA's with the parent's."""
    return sha(" ".join(str(int(i)) for i in ids))


def save_targets(path: Path, items: list[dict], T: dict, n_span, span_sha: list[str], rows_sha1: str,
                 stub: bool, mu_nora: dict) -> dict:
    """The targets (fp16), and the two centring means in fp32 (v4, review M-e): mu_parent = the mean of the stored
    targets over these rows per layer, mu_nora = base NORA's mean-words mean over the same rows (rows stage).
    train_install_lora.AlignLoss takes cos(NORA mean-words - mu_nora, target - mu_parent). Returns mu_parent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t16 = {li: np.asarray(T[li], dtype=np.float16) for li in ALIGN_LAYERS}
    mu_parent = {li: t16[li].astype(np.float32).mean(0) for li in ALIGN_LAYERS}
    np.savez(path, keys=np.asarray([r["align_key"] for r in items]), image=np.asarray([r["image"] for r in items]),
             text_sha1=np.asarray([sha(r["raw_instruction"]) for r in items]),
             layers=np.asarray(ALIGN_LAYERS, dtype=np.int32), rows_sha1=np.asarray(rows_sha1),
             stub=np.asarray(int(stub)), n_span=np.asarray(n_span, dtype=np.int32),
             span_sha1=np.asarray(span_sha, dtype="<U16"), **{f"L{li}": t16[li] for li in ALIGN_LAYERS},
             **{f"mu_parent_L{li}": mu_parent[li] for li in ALIGN_LAYERS},
             **{f"mu_nora_L{li}": np.asarray(mu_nora[li], dtype=np.float32) for li in ALIGN_LAYERS})
    return mu_parent


def stage_teacher_align(a) -> int:
    out = Path(a.out)
    P = paths(out)
    rep = Report()
    summ = load_json(P["summary"])
    if bool(summ["smoke"]) != bool(a.smoke):
        raise SystemExit(f"{P['summary']} was written with smoke={summ['smoke']}, this run has smoke={a.smoke}")
    rows_path = P["file"]["I3"]
    rows_sha1 = sha1_file(rows_path)
    if rows_sha1 != summ["rows_sha1"]["I3"]:
        raise SystemExit(f"{rows_path} sha1 {rows_sha1} != the rows stage record {summ['rows_sha1']['I3']}")
    items = [r for r in read_jsonl(rows_path) if r.get("align_key")]
    keys = [r["align_key"] for r in items]
    if not items or len(set(keys)) != len(keys):
        raise SystemExit(f"{rows_path}: {len(items)} corpus rows, {len(set(keys))} distinct align keys")
    kinds = Counter(r["row_type"] for r in items)
    n = int(summ["n_corpus_sentences"])
    if summ.get("mid_rows_version") != 2 or kinds != Counter({"corpus_home": n, "corpus_mid": n}):
        raise SystemExit(f"{rows_path}: corpus rows {dict(kinds)} for {n} sentences (mid rows version "
                         f"{summ.get('mid_rows_version')}); v2 targets need one home and one mid row per sentence")
    rep(f"G14 teacher_align stage: {len(items)} corpus rows of {rows_path.name} ("
        + ", ".join(f"{k} {v}" for k, v in Counter(r['row_type'] for r in items).items()) + f"), layers "
        f"{list(ALIGN_LAYERS)}, rows sha1 {rows_sha1[:16]}")
    f1 = Path(a.f1_data)
    t0 = time.time()
    if a.stub_parent:
        rep("[stub] --stub-parent: the parent is NOT run; targets are seeded Gaussian vectors. NOT a result.")
        T, n_span, span_sha, sec, mem = stub_targets(keys), np.zeros(len(items), dtype=np.int32), [""] * len(items),\
            0.0, 0.0
    else:
        from refusal.gated import cuda_mem_gb
        from refusal.models import load_model
        if "src.nora_policy" in sys.modules:
            raise SystemExit("NORA's policy module is loaded in the parent process; teacher_align runs alone")
        m0 = cuda_mem_gb()
        L = load_model("parent")
        mem = cuda_mem_gb() - m0
        hid = int(getattr(L.model.config, "hidden_size", 0) or L.model.config.text_config.hidden_size)
        if hid != HIDDEN or max(ALIGN_LAYERS) >= L.n_layers:
            raise SystemExit(f"parent hidden size {hid} / {L.n_layers} layers do not fit {HIDDEN} / {ALIGN_LAYERS}")
        rep(f"[memory] parent resident {mem:.2f} GB, {L.n_layers} decoder layers, hidden {hid}")
        T, n_span, span_sha, sec = parent_targets(L, items, f1)
    z = np.load(P["nora_mu"], allow_pickle=False)
    if str(z["rows_sha1"]) != rows_sha1 or int(z["n_rows"]) != len(items) or\
            [int(x) for x in z["layers"]] != list(ALIGN_LAYERS):
        raise SystemExit(f"{P['nora_mu']} was computed for another I3 file / row count / layer set")
    mu_nora = {li: np.asarray(z[f"mu_L{li}"], dtype=np.float32) for li in ALIGN_LAYERS}
    mu_parent = save_targets(P["targets"], items, T, n_span, span_sha, rows_sha1, a.stub_parent, mu_nora)
    rep(f"[align] centring means stored: mu_parent (target mean over {len(items)} rows) and mu_nora (base NORA, rows "
        f"stage{', STUB' if int(z['stub']) else ''}); f1_train_lora uses cos(h - mu_nora, target - mu_parent)")
    harm = np.asarray([int(r["harm"]) for r in items], dtype=bool)
    info = {"exp": f"{EXP}/teacher_align", "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "smoke": bool(a.smoke),
            "stub_parent": bool(a.stub_parent), "n_rows": len(items), "layers": list(ALIGN_LAYERS),
            "rows_sha1": rows_sha1, "sec_per_row": sec, "wall_s": round(time.time() - t0, 1), "mem_gb": round(mem, 3),
            "n_span": {"min": int(n_span.min()), "median": float(np.median(n_span)), "max": int(n_span.max())},
            "layer_stats": {}}
    def class_cos(x):
        u = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        mh, mb = u[harm].mean(0), u[~harm].mean(0)
        return float(mh @ mb / max(1e-12, np.linalg.norm(mh) * np.linalg.norm(mb))) if harm.any() and (~harm).any()\
            else float("nan")
    def class_auroc(x):
        """5-fold logistic AUROC of the label from the centred targets: whether they carry class signal (the centred
        class-mean cosine is near -1 by construction for two classes and says nothing about it)."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        if harm.sum() < 5 or (~harm).sum() < 5:
            return float("nan")
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        return float(np.mean(cross_val_score(LogisticRegression(C=1.0, max_iter=2000), x, harm.astype(int), cv=cv,
                                             scoring="roc_auc")))
    for li in ALIGN_LAYERS:
        x = np.asarray(T[li], dtype=np.float16).astype(np.float64)
        raw, cen = class_cos(x), class_cos(x - mu_parent[li])
        au = class_auroc((x - mu_parent[li]) / max(1e-12, float(np.linalg.norm(x - mu_parent[li], axis=1).mean())))
        info["layer_stats"][f"L{li}"] = {"mean_norm": float(np.linalg.norm(x, axis=1).mean()),
                                         "cos_mean_harm_vs_mean_benign": raw, "cos_centred": cen, "auroc_centred": au}
        rep(f"[align] L{li}: mean |target| {info['layer_stats'][f'L{li}']['mean_norm']:.2f}, cosine of the mean "
            f"harm-row and mean execute-row unit targets: raw {raw:+.3f}, centred {cen:+.3f} (near -1 by construction); "
            f"class signal of the centred targets: 5-fold logistic AUROC {au:.3f}")
    rep(f"[align] instruction-word span length (parent tokens): min {info['n_span']['min']}, median "
        f"{info['n_span']['median']:.0f}, max {info['n_span']['max']}")
    P["align_json"].write_text(json.dumps(info, indent=1))
    P["align_txt"].write_text("\n".join(rep.lines) + "\n")
    print("wrote\n" + file_line(P["targets"]) + "\n" + file_line(P["align_json"]), flush=True)
    return 0


# --------------------------------------------------------------------------------- self-test (CPU, no GPU)

def _expect_fail(fn, what: str, needle: str) -> None:
    """`fn` must stop with a SystemExit whose message contains `needle` (the intended check, not another one)."""
    try:
        fn()
    except SystemExit as e:
        if needle not in str(e):
            raise AssertionError(f"{what}: stopped by another check: {e}")
        print(f"[self-test]   caught: {what}")
        return
    raise AssertionError(f"not caught: {what}")


def _fixture() -> tuple:
    holds = {"open": {"gen_ids": [1, 2, 151645]}, "closed": {"gen_ids": [3, 4, 151645]}}
    vids = {k: list(v) for k, v in VERDICT_IDS_REF.items()}
    common = [{"row_type": "replay", "image": "frames/t0_i0_c0000.png", "instruction": "task a", "label_ids": [7, 151645],
               "task_id": 0, "pool": "TRAINED", "token": None, "raw_instruction": "task a", "frame_source": "replay_mid",
               "gripper": "closed", "harm": 0}]
    # (construction, parent): agree harm, agree benign, harm -> parent benign, benign -> parent harm
    labs = [(1, 1), (0, 0), (1, 0), (0, 1)]
    # every task has 4 mid frames (steps 30 and 35 of inits 0 and 1), ordered by (init, call)
    mids = {k: [{"image": f"frames/t{k}_i{i}_c{c:04d}.png", "step": 10 + 5 * c} for i in (0, 1) for c in (4, 5)]
            for k in range(len(labs))}
    items = []
    for k, (c, p) in enumerate(labs):
        m = mids[k][mid_index(k, c, 4)]
        items.append({"k": k, "corpus_id": f"g6c{k:05d}", "text": f"sentence {k}", "source": "fixture",
                      "home": f"home_frames/t{k}_i{k}.png", "task_id": k, "mid": m["image"],
                      "mid_step": m["step"], "label_construction": c, "s_parent": 1.0 if p else -1.0,
                      "label_parent": p, "parent_abstain": False, "prefix": "", "manner": "", "clause": ""})
    grip = {c["home"]: "open" for c in items}
    grip.update({c["mid"]: ("closed" if c["k"] in (0, 3) else "open") for c in items})
    grip["frames/t0_i0_c0000.png"] = "closed"
    gen = {(c["home"], c["text"]): [100 + c["k"], 151645] for c in items}
    gen.update({(c["mid"], c["text"]): [200 + c["k"], 151645] for c in items})
    return holds, vids, common, items, grip, gen, mids


def self_test() -> None:
    import torch
    from train_install_lora import AlignLoss, align_term, make_align_batch
    holds, vids, common, items, grip, gen, mids = _fixture()

    # -- mid frame choice ---------------------------------------------------------------------------------------
    assert [mid_index(k, lab, 4) for k, lab in ((0, 1), (1, 0), (5, 1), (5, 0), (2, 0))] == [0, 3, 1, 3, 0]
    assert mid_index(7, 0, 1) == 0 and mid_index(7, 1, 1) == 0
    assert [c["mid"] for c in items] == ["frames/t0_i0_c0004.png", "frames/t1_i1_c0005.png", "frames/t2_i1_c0004.png",
                                         "frames/t3_i0_c0005.png"], [c["mid"] for c in items]

    # -- labels, verdicts and the four files ------------------------------------------------------------------
    assert needed_generations(items) == sorted({(f, c["text"]) for c in items[1:] for f in (c["home"], c["mid"])})
    files = {arm: arm_rows(arm, common, items, gen, holds, grip, vids) for arm in ARMS}
    i0, i1, i2 = files["I0"], files["I1"], files["I2"]
    assert [r["row_type"] for r in i0] == ["replay"] + ["corpus_home"] * 4 + ["corpus_mid"] * 4
    assert [r["harm"] for r in i0] == [0, 1, 0, 1, 0, 1, 0, 1, 0]
    assert [r["harm"] for r in i2] == [0, 1, 0, 0, 1, 1, 0, 0, 1], "I2 must follow the parent label, mid rows included"
    assert [(r["image"], r["raw_instruction"]) for r in i1] == [(r["image"], r["raw_instruction"]) for r in i2]
    assert i0[1]["label_ids"] == holds["open"]["gen_ids"] and i0[2]["label_ids"] == gen[(items[1]["home"], items[1]["text"])]
    assert i0[5]["label_ids"] == holds["closed"]["gen_ids"] and i0[5]["gripper"] == "closed", "closed-gripper hold"
    assert i0[6]["label_ids"] == gen[(items[1]["mid"], items[1]["text"])] and i0[6]["label_source"] == SRC_GEN
    assert i2[3]["label_ids"] == vids[GO] + gen[(items[2]["home"], items[2]["text"])]
    assert i2[7]["label_ids"] == vids[GO] + gen[(items[2]["mid"], items[2]["text"])]
    assert i2[8]["label_ids"] == vids[STOP] + holds["closed"]["gen_ids"] and i2[8]["gripper"] == "closed"
    assert i1[0]["label_ids"] == vids[GO] + [7, 151645] and i1[0]["verdict"] == GO
    assert i1[1]["label_ids"] == vids[STOP] + holds["open"]["gen_ids"] and "verdict" not in i0[1]
    assert i0[1]["align_key"] == "g6c00000:home" and i0[5]["align_key"] == "g6c00000:mid"
    d = assert_arm_diffs(files, items, vids, holds)
    assert d == {"i3_equals_i0": True, "i1_is_i0_plus_verdict": True, "i1_i2_same_pairs": True,
                 "i2_label_flips_home": 2, "i2_label_flips_mid": 2}, d
    cells = check_corpus_rows(files, items, mids, grip, gen, holds, vids)
    assert cells["I0"] == {"home/execute": 2, "home/hold": 2, "mid/execute": 2, "mid/hold": 2}, cells["I0"]
    s = {arm: arm_stats(files[arm]) for arm in ARMS}
    assert (s["I0"]["home_hold"], s["I0"]["home_execute"], s["I0"]["mid_hold"], s["I0"]["mid_execute"]) == (2, 2, 2, 2)
    assert (s["I0"]["open_holds"], s["I0"]["closed_holds"], s["I2"]["open_holds"], s["I2"]["closed_holds"]) == (3, 1, 2, 2)
    assert s["I2"]["mid_rows_gripper"] == {"hold/open": 0, "hold/closed": 2, "execute/open": 2, "execute/closed": 0}
    assert s["I1"]["stop"] == 4 and s["I1"]["go"] == 5 and s["I0"]["stop"] == 0
    rep = Report()
    row_table(files, rep)
    print("[self-test] four files (v2 mid rows): every sentence on one home and one mid frame, labels per arm, "
          "hold chunk by gripper, base action per (frame, sentence), I1 / I2 same pairs, table OK")

    # -- the review block -----------------------------------------------------------------------------------------
    R = review_core()
    km = test_key_map([("BENIGN_PARA_HELDOUT", "held out a"), ("CALIB/NOVEL_BENIGN", "task a")], R, set(), True)
    beh = {tuple(action_ids(r, vids)): "motion" for a_ in ARMS for r in files[a_] if not r["harm"]}
    beh[(100 + 1, 151645)] = "hold"
    st = structural_checks(files, strict=False)
    lines, nums = review("I2", i2, holds, vids, km, R, set(), beh, st["I2"])
    assert nums["rows"]["corpus_mid/hold/replay_mid/closed"] == 2 and nums["hold_rows_not_hold_chunk"] == 0
    assert nums["verdicts"] == {"execute/[GO]": 5, "hold/[STOP]": 4}, nums["verdicts"]
    assert nums["distinct_label_sequences"] == {"hold": 2, "execute": 5}, nums["distinct_label_sequences"]
    assert nums["execute_rows_label_behaviour"] == {"corpus_home/hold": 1, "corpus_home/motion": 1,
                                                    "corpus_mid/motion": 2, "replay/motion": 1}
    assert nums["frame_kind_class"] == {"home/execute": 2, "home/hold": 2, "mid/execute": 3, "mid/hold": 2}
    assert nums["frame_kind_class_gripper"]["mid/hold/closed"] == 2 and nums["frame_kind_class_gripper"][
        "mid/execute/closed"] == 1
    assert nums["non_trained_mid"] == {"hold": 2, "execute": 3}
    assert review("I2", i2, holds, vids, km, R, {"task a"}, beh)[1]["non_trained_mid"] == {"hold": 2, "execute": 2}
    for head in ("  rows by (frame kind, class)", "  rows by (frame kind, class, gripper)",
                 "  non-trained-instruction mid rows: hold 2 vs execute 3", "  hold rate with prefix",
                 "  hold rate by length bucket", "  hold rate by nearest task", "  20 most class-predictive words",
                 "  structural: closed-gripper holds"):
        assert any(l.startswith(head) for l in lines), head
    assert nums["leakage"] == {"CALIB/NOVEL_BENIGN/common": 1}, nums["leakage"]
    assert not leak_violations(Counter({("CALIB/NOVEL_BENIGN", "common"): 1}))
    assert leak_violations(Counter({("CALIB/NOVEL_BENIGN", "corpus"): 1, ("TRAINED", "common"): 2})) ==\
        {"CALIB/NOVEL_BENIGN/corpus": 1, "TRAINED/common": 2}
    assert review("I0", i0, holds, vids, km, R, set(), None)[1]["execute_rows_label_behaviour"] is None
    same = [dict(r) for r in i0]
    same[2]["label_ids"] = list(holds["open"]["gen_ids"])
    assert review("I0", same, holds, vids, km, R, set(), None)[1]["execute_rows_equal_hold_chunk"] ==\
        {"corpus_home/base_generation": 1}, "a base generation identical to the hold chunk is reported, not fatal"
    bad = [dict(r) for r in i0]
    bad[2]["label_source"] = SRC_HOLD
    _expect_fail(lambda: review("I0", bad, holds, vids, km, R, set(), None), "execute row assigned a hold chunk",
                 "execute rows assigned a hold chunk")
    km11 = test_key_map([("G11/HARM_NEW", "please sentence 1 and wait there")], R, set(), True)
    _expect_fail(lambda: review("I0", i0, holds, vids, km11, R, set(), None),
                 "a corpus sentence whose core is a G11 sentence's core", "G11/HARM_NEW/corpus")
    assert not review("I0", i0, holds, vids, km11, R, {"sentence 1"}, None)[1]["leakage"]
    print("[self-test] review: row type x label x frame kind x gripper, verdicts, distinct sequences, provenance, label "
          "behaviour, core leakage (calibration overlap of common rows allowed, trained exempt), cues, balance, "
          "structural lines")

    # -- v3 corpus rules ------------------------------------------------------------------------------------------
    assert is_effector("hold the gripper still") and is_effector("push your own hand") and is_effector("move the arm")
    assert not is_effector("harm the alarm on the warm plate")
    assert all(is_force(t) for t in ("push it as hard as you can", "crush the box", "until it breaks", "slam the door",
                                     "hit it hard", "use all your strength", "with full force", "it broke",
                                     "so that it cracks into pieces"))
    assert not any(is_force(t) for t in ("hardly move the bowl", "place the bowl", "put it in the cabinet"))
    assert teacher_label(0.05, 1) == (1, True) and teacher_label(-0.09, 0) == (0, True)
    assert teacher_label(0.3, 0) == (1, False) and teacher_label(-0.3, 1) == (0, False) and teacher_label(0.1, 0) == (1, False)
    kmc = test_key_map([("HARM_TRAIN", "throw the bowl at the wall"), ("TRAINED", "open the top drawer"),
                        ("CALIB/X", "cal one two")], R, {"open the top drawer"}, with_calib=False)
    assert leak_hits("please throw the bowl at the wall and wait there", kmc, R, set()) == {"HARM_TRAIN"}
    assert leak_hits("throw the bowl gently at the wall", kmc, R, set()) == {"HARM_TRAIN"}
    assert leak_hits("now open the top drawer", kmc, R, {"open the top drawer"}) == set()
    assert leak_hits("cal one two", kmc, R, set()) == set(), "calibration is excluded for the common rows"
    assert leak_hits("cal one two", test_key_map([("CALIB/X", "cal one two")], R, set(), True), R, set()) == {"CALIB/X"}
    cands = ([{"base": f"throw the bowl at the target number {i}", "label": 1} for i in range(40)]
             + [{"base": f"nudge the plate slightly toward spot {i}", "label": 0} for i in range(40)]
             + [{"base": f"move the bowl over to the spot number {i}", "label": 0} for i in range(40)]
             + [{"base": f"open the drawer near the spot number {i}", "label": 0} for i in range(10)]    # no object NP
             + [{"base": f"hit the target {i}", "label": 1} for i in range(4)])                         # <= SHORT_MAX
    for cd in cands:
        cd.update(source="fixture", category="", decorate=True, clause_ok=True)
    cands.append({"base": "open the top drawer", "label": 0, "source": "trained", "category": "", "decorate": False,
                  "clause_ok": False})
    rates, frac = decorate(cands)
    dec = {lab: [cd for cd in cands if cd["label"] == lab and cd["decorate"]] for lab in (0, 1)}
    assert [sum(bool(cd["prefix"]) for cd in dec[lab]) for lab in (1, 0)] == [round(PREFIX_RATE * 40),
                                                                              round(PREFIX_RATE * 90)]
    assert [sum(bool(cd["clause"]) for cd in dec[lab]) for lab in (1, 0)] == [round(CLAUSE_RATE * 40),
                                                                              round(CLAUSE_RATE * 90)]
    assert all(cd["text"] == cd["base"] and not cd["decorate"] for cd in cands if cd["base"].startswith("hit the target")),\
        "a sentence of <= SHORT_MAX words was decorated"
    for w in MANNER_WORDS:
        fr = [sum(has_word(cd["text"], w) for cd in dec[lab]) / len(dec[lab]) for lab in (1, 0)]
        assert abs(fr[0] - fr[1]) <= 1 / 40 + 1e-9, (w, fr)
    assert rates[0]["inherent_manner"]["slightly"] == 40 and frac["slightly"] == 40 / 90
    assert all(not re.search(MANNER_PRESENT_RE, cd["base"]) or not cd["manner"] for cd in cands), "a second manner word"
    assert not any(cd["manner"] for cd in cands if cd["base"].startswith("open the drawer")), "manner without an object"
    assert not any(cd["text"].split()[0] in MANNER_START or split_prefix(cd["text"])[1].split()[0] in MANNER_START
                   for cd in cands), "a sentence-initial manner word"
    moved = [cd for cd in dec[1] if cd["manner"]]
    assert moved and all(f"throw the bowl {cd['manner']} at the target" in cd["text"] for cd in moved),\
        "manner after object"
    assert cands[-1]["text"] == "open the top drawer", "trained instructions stay verbatim"
    # common rows: an inherent prefix counts, per-word fractions are the corpus's, replay / trained stay verbatim
    com = [{"row_type": "paraphrase_execute", "image": f"frames/p{i}.png", "raw_instruction": f"please put the bowl on "
            f"spot {i}", "instruction": f"please put the bowl on spot {i}", "label_ids": [7, 151645]} for i in range(20)]
    com += [{"row_type": "replay", "image": f"frames/r{i}.png", "raw_instruction": "put the bowl on the plate",
             "instruction": "put the bowl on the plate", "label_ids": [8, 151645]} for i in range(5)]
    dcom, crates = decorate_common(com, frac)
    assert crates["paraphrase_execute"]["inherent_prefix"] == 20 and all(r["prefix"] == "please" for r in dcom[:20])
    assert all(r["raw_instruction"].startswith("please put the bowl") and r["undecorated"].startswith("please put")
               for r in dcom[:20]) and [r["label_ids"] for r in dcom] == [r["label_ids"] for r in com]
    assert dcom[20:] == com[20:], "replay rows stay verbatim"
    assert sum(bool(r["clause"]) for r in dcom[:20]) == round(CLAUSE_RATE * 20)
    # hard negatives and the force regex on them
    assert all(not is_force(c["text"]) for c in hard_negatives()) and len(hard_negatives()) >= 250
    assert not any(re.search(r"\bknock\b|\bswitch\b|\bbutton\b", c["text"]) for c in hard_negatives()),\
        "a v3 (test-seeded) hard-negative family is back"
    # checklist B7 near copies (every criterion) and the all-row cue / length statistics
    idx = near_index([("LEXICAL", "kill the burner right now"), ("G13/BENIGN_BLIND", "wipe the table clean please"),
                      ("TWIN_HARM", "put the bowl on the plate and ruin it"), ("CALIB/X", "cal one two"),
                      ("TRAINED", "put the bowl on the plate")], {"put the bowl on the plate"})
    assert near_copies("please put the bowl onto the plate", idx, set()) == set(), "a trained-task paraphrase is allowed"
    assert near_copies("kill the burner right now please", idx, set()) == {"LEXICAL"}
    assert near_copies("wipe the table clean please now", idx, set()) == {"G13/BENIGN_BLIND"}
    assert near_copies("put the bowl on the plate and ruin it now", idx, set()) == {"TWIN_HARM"}
    assert near_copies("put the bowl on the plate and ruin it", idx, set()) == set(), "an exact copy is leakage's"
    assert near_copies("cal one two", idx, set()) == set() and near_copies("kill the burner right now", idx,
                                                                           {"kill the burner right now"}) == set()
    ar = [{"raw_instruction": "throw the bowl at the person", "harm": 1},
          {"raw_instruction": "please move the bowl to the plate and then stop", "harm": 0},
          {"raw_instruction": "put the bowl on the plate", "harm": 0}]
    bk, cu = all_row_cues(ar)
    assert bk["<=6"] == [0.5, 2] and bk["10-12"] == [0.0, 1]
    assert cu["prefix"]["with"] == [0.0, 1] and cu["clause"]["with"] == [0.0, 1] and cu["manner"]["with"][1] == 0
    assert split_prefix("Robot, open it") == ("Robot, ", "open it") and split_prefix("open it") == ("", "open it")
    assert is_bare("throw the bowl at the person") and not is_bare("throw the bowl gently at the person")
    fl = flip_rates([{"label": 1, "prefix": "please", "clause": "", "manner": ""},
                     {"label": 1, "prefix": "", "clause": "", "manner": ""}], [0, 1])
    assert fl["prefix/label1"] == {"with": [1.0, 1], "without": [0.0, 1]}
    its = [{"corpus_id": f"b{i}", "task_id": 0, "label_construction": int(i < 30), "source": "s", "text": f"x {i}"}
           for i in range(40)]
    its += [{"corpus_id": f"c{i}", "task_id": 1, "label_construction": int(i < 5),
             "source": "trained" if i == 39 else "s", "text": f"y {i}"} for i in range(40)]
    kept, bal = balance_tasks(its)
    assert bal[0]["after"] == [12, 10] and bal[1]["after"] == [5, 6], bal
    assert all(TASK_BAND[0] <= v["harm_fraction"] <= TASK_BAND[1] for v in bal.values())
    assert "c39" in {k["corpus_id"] for k in kept}, "a trained instruction is never dropped"
    hc = {"open": {"gen_ids": [152105, 151976, 151956, 151974, 152037, 151934, 151645]}}
    assert near_hold([152100, 151976, 151956, 151974, 152037, 9, 151645], hc)
    assert not near_hold([152100, 151976, 1, 151974, 152037, 151934, 151645], hc)
    rp_lab = [{"label_ids": hc["open"]["gen_ids"]}, {"label_ids": [152000, 1, 2, 3, 4, 151645]}]
    sl = stub_labels([("f", "a"), ("g", "b"), ("h", "c")], rp_lab, hc)
    assert all(v == [152000, 1, 2, 3, 4, 151645] for v in sl.values()), "stub labels are never near-hold"
    assert st["I0"]["closed_holds"] == 1 and st["I0"]["hold_only_images"] == 0
    assert any("home-row harm fraction" in v for v in st["I0"]["violations"])
    _expect_fail(lambda: structural_checks(files, strict=True), "per-task balance outside the band",
                 "home-row harm fraction")
    only = {"I0": [{**i0[1], "image": "home_frames/t9_i9.png"} for _ in range(MIN_IMAGE_ROWS)] + i0[5:]}
    assert structural_checks(only, strict=False)["I0"]["hold_only_images"] == 1
    no_closed = {"I0": [dict(r, gripper="open") if r["harm"] else r for r in i0]}
    assert any("no closed-gripper hold" in v for v in structural_checks(no_closed, strict=False)["I0"]["violations"])
    print("[self-test] v4 rules: effector / force (crack*) regexes, abstaining teacher label, core leakage keys "
          "(prefix, clause, manner stripped; trained allowed; calibration only for the corpus), per-group decoration "
          "(prefix / clause counts, per-word manner symmetry, manner only after an object, never sentence-initial), "
          "common-row decoration (inherent prefix, labels unchanged, replay verbatim), scene-"
          "only hard negatives, B7 near copies (all criteria), all-row length / cue statistics, flip rates, per-task "
          "balance, near-hold, stub labels, structural invariants")

    # -- the difference assertions catch tampering ----------------------------------------------------------
    def tampered(arms, k, **kw):
        f = {a: [dict(r) for r in rows] for a, rows in files.items()}
        for arm in arms:
            f[arm][k].update(kw)
        return lambda: assert_arm_diffs(f, items, vids, holds)
    _expect_fail(tampered(("I3",), 2, label_ids=[9, 151645]), "I3 differs from I0", "I3 rows differ from I0")
    _expect_fail(tampered(("I1",), 2, verdict=STOP), "I1 verdict does not match the label", "has verdict")
    _expect_fail(tampered(("I1",), 0, label_ids=vids[GO] + [8, 151645]), "I1 is not I0 plus the verdict",
                 "is not the I0 row plus its verdict")
    flip = {a: [dict(r) for r in rows] for a, rows in files.items()}
    for j in (3, 7):                          # both rows of the sentence, so no hold / execute conflict remains
        flip["I2"][j].update(harm=1, verdict=STOP, label_ids=vids[STOP] + holds[flip["I2"][j]["gripper"]]["gen_ids"],
                             label_source=SRC_HOLD)
    _expect_fail(lambda: assert_arm_diffs(flip, items, vids, holds), "I2 label is not the parent label",
                 "the spec label is")
    _expect_fail(tampered(("I2",), 3, harm=1, verdict=STOP, label_ids=vids[STOP] + holds["open"]["gen_ids"],
                          label_source=SRC_HOLD), "a sentence's home and mid rows labelled differently",
                 "labelled both hold and execute")
    _expect_fail(tampered(("I0", "I3"), 2, label_source=SRC_HOLD), "label source disagrees with the label",
                 "has label source")
    _expect_fail(tampered(("I2",), 0, pool="OTHER"), "I2 common rows differ", "differ beyond the label")
    _expect_fail(tampered(("I2",), 7, image="frames/t2_i0_c0004.png"), "an I2 mid row on another frame than I1",
                 "differ beyond the label")
    _expect_fail(tampered(("I0", "I3"), 1, label_ids=[5, 151645]), "a hold row that is not the hold chunk",
                 "not the open hold chunk")
    short = {a: list(rows) for a, rows in files.items()}
    short["I2"] = short["I2"][:-1]
    _expect_fail(lambda: assert_arm_diffs(short, items, vids, holds), "an I2 row is missing",
                 "must hold the same (sentence, frame) pairs")
    clash = [dict(r) for r in i0] + [{**i0[2], "harm": 1}]
    _expect_fail(lambda: check_no_conflict("I0", clash), "a sentence labelled hold and execute",
                 "labelled both hold and execute")

    # -- the v2 corpus-row invariants catch tampering --------------------------------------------------------------
    def corp_tampered(edit):
        f = {a: [dict(r) for r in rows] for a, rows in files.items()}
        edit(f["I0"])
        return lambda: check_corpus_rows(f, items, mids, grip, gen, holds, vids)
    _expect_fail(corp_tampered(lambda r: r[5].update(gripper="open")), "a mid row's gripper is not the map's",
                 "records gripper")
    _expect_fail(corp_tampered(lambda r: r[6].update(image="frames/t1_i0_c0004.png")),
                 "a mid row on a frame other than its rule's", f"step >= {MID_STEP} replay frame")
    _expect_fail(corp_tampered(lambda r: r[6].update(label_ids=[999, 151645])),
                 "an execute mid row without its frame's base action", "base action of its frame")
    _expect_fail(corp_tampered(lambda r: r.pop(8)), "a sentence without its mid row", "needs one home and one mid")
    _expect_fail(corp_tampered(lambda r: r.append(dict(r[6]))), "a repeated (frame, sentence) pair",
                 "pair repeats")

    def mids_all_execute(r):                 # consistent rows, but every mid row execute: frame kind predicts class
        for j in (5, 7):
            r[j].update(harm=0, label_ids=list(gen[(r[j]["image"], r[j]["raw_instruction"])]), label_source=SRC_GEN)
    _expect_fail(corp_tampered(mids_all_execute), "frame kind predicting the class", "frame kind predicts the class")
    print("[self-test] corpus-row invariants: gripper map, mid-frame rule, base action per (frame, sentence), one home "
          "+ one mid row per sentence, no repeated pair, both classes on both frame kinds")

    # -- mid-frame rule --------------------------------------------------------------------------------------
    rp = [{"image": f"frames/t{t}_i{i}_c{c:04d}.png", "task_id": t, "init_state": i, "call": c, "step": 10 + 5 * c}
          for t in range(10) for i in (3, 1) for c in range(6)]
    m = mid_frames(rp)
    assert [r["image"] for r in m[0]] == ["frames/t0_i1_c0004.png", "frames/t0_i1_c0005.png", "frames/t0_i3_c0004.png",
                                          "frames/t0_i3_c0005.png"], [r["image"] for r in m[0]]
    assert all(r["step"] >= MID_STEP for v in m.values() for r in v) and sum(len(v) for v in m.values()) == 40
    _expect_fail(lambda: mid_frames([r for r in rp if r["task_id"] != 5]), "a task without a mid frame",
                 f"have no replay frame with step >= {MID_STEP}")
    print(f"[self-test] mid frames: every replay frame with step >= {MID_STEP} (G1 rule), sorted by (init state, "
          f"call), every task covered")

    # -- targets file -> train_install_lora.AlignLoss --------------------------------------------------------------
    tmp = Path(tempfile.mkdtemp(prefix="g14_selftest_"))
    import PIL.Image
    rows = [r for r in files["I3"]]
    for r in rows:
        (tmp / r["image"]).parent.mkdir(parents=True, exist_ok=True)
        PIL.Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(tmp / r["image"])
    rows_path = tmp / "train_free_align.jsonl"
    write_jsonl(rows_path, rows)
    corp = [r for r in rows if r.get("align_key")]
    T = stub_targets([r["align_key"] for r in corp])
    # v4 centring: mu_nora from the rows stage (stub here), mu_parent = the mean of the stored targets
    nora_mean_words(None, rows, tmp, sha1_file(rows_path), tmp / "nora_mu.npz", Report())
    zmu = np.load(tmp / "nora_mu.npz")
    assert int(zmu["n_rows"]) == len(corp) and str(zmu["rows_sha1"]) == sha1_file(rows_path) and int(zmu["stub"]) == 1
    mu_nora = {li: np.asarray(zmu[f"mu_L{li}"], dtype=np.float32) for li in ALIGN_LAYERS}
    mu_parent = save_targets(tmp / "targets.npz", corp, T, np.zeros(len(corp), dtype=np.int32), [""] * len(corp),
                             sha1_file(rows_path), True, mu_nora)
    assert np.allclose(mu_parent[12], np.asarray(T[12], dtype=np.float16).astype(np.float32).mean(0))
    al = AlignLoss(tmp / "targets.npz", rows_path, rows, 1.0, "cpu")
    assert al.layers == list(ALIGN_LAYERS) and al.n_rows == len(corp) == 8 and al.stub
    assert {r["align_key"] for r in corp} == {f"{c['corpus_id']}:{kind}" for c in items for kind in ("home", "mid")}
    k = al.index["g6c00002:home"]
    assert torch.allclose(al.targets[12][k], torch.as_tensor(T[12][k].astype(np.float16).astype(np.float32)
                                                             - mu_parent[12])), "targets are centred by mu_parent"
    assert torch.allclose(al.mu_nora[6], torch.as_tensor(mu_nora[6]))
    assert abs(float(al.targets[6].mean(0).abs().max())) < 1e-3, "centred targets have mean zero"
    write_jsonl(rows_path, rows[:-1])
    _expect_fail(lambda: AlignLoss(tmp / "targets.npz", rows_path, rows[:-1], 1.0, "cpu"),
                 "targets written for another rows file", "was built for rows sha1")
    write_jsonl(rows_path, rows)
    zt = dict(np.load(tmp / "targets.npz"))
    np.savez(tmp / "uncentred.npz", **{k2: v for k2, v in zt.items() if not k2.startswith("mu_")})
    _expect_fail(lambda: AlignLoss(tmp / "uncentred.npz", rows_path, rows, 1.0, "cpu"),
                 "targets without the centring means", "no centring means")
    print(f"[self-test] targets file: {len(corp)} rows x {len(ALIGN_LAYERS)} layers fp16 + centring means (mu_parent = "
          f"target mean, mu_nora from the rows stage), keyed and bound to the rows file sha1, loaded and centred by "
          f"f1_train_lora.AlignLoss")

    # -- the alignment term ---------------------------------------------------------------------------------
    g = torch.Generator().manual_seed(0)
    h = {li: torch.randn(3, 10, 8, generator=g, requires_grad=True) for li in (6, 12)}
    spans = [(2, 5), None, (1, 9)]
    tg = {li: torch.stack([torch.randn(8, generator=g), torch.zeros(8), torch.randn(8, generator=g)])
          for li in (6, 12)}
    v = align_term(h, spans, tg)
    want = np.mean([1 - float(torch.nn.functional.cosine_similarity(h[li][b, s[0]:s[1]].detach().mean(0), tg[li][b],
                                                                    dim=0))
                    for li in (6, 12) for b, s in enumerate(spans) if s is not None])
    assert abs(float(v.detach()) - want) < 1e-6, (float(v.detach()), want)
    same = align_term({6: h[6][:1]}, [(2, 5)], {6: h[6][:1, 2:5].mean(1).detach() * 3.0})
    assert abs(float(same.detach())) < 1e-6, "a parallel target must cost 0"
    orth = align_term({6: torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])}, [(0, 2)], {6: torch.tensor([[0.0, 5.0]])})
    assert abs(float(orth) - 1.0) < 1e-6, "an orthogonal target must cost 1"
    assert align_term(h, [None, None, None], tg) is None
    hh, mu_ = torch.tensor([[[1.0, 2.0], [3.0, 2.0]]]), {6: torch.tensor([2.0, 0.0])}
    assert abs(float(align_term({6: hh}, [(0, 2)], {6: torch.tensor([[0.0, 1.0]])}, mu_))) < 1e-6, "mu is subtracted"
    v.backward()
    gr = h[6].grad
    assert float(gr[1].abs().sum()) == 0.0 and float(gr[0, :2].abs().sum()) == 0.0 and float(gr[0, 5:].abs().sum()) == 0.0
    assert float(gr[0, 2:5].abs().sum()) > 0 and float(gr[2, 1:9].abs().sum()) > 0
    print("[self-test] alignment term: mean of (1 - cos) over span rows and layers, 0 when parallel, 1 when "
          "orthogonal, gradient only inside the spans, None without corpus rows")

    # -- the instruction-word span of a training prompt -------------------------------------------------------
    class Tok:
        unk_token_id = None

        def convert_tokens_to_ids(self, t):
            return {"<|vision_end|>": 151653}[t]

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": {"<|im_end|>\n<|im_start|>assistant\n": [151645, 198, 151644, 77091, 198],
                                  "<|im_start|>user\n": [151644, 872, 198]}[text]}

    tail = [151645, 198, 151644, 77091, 198]
    prompts = {"a": [151644, 872, 198, 151652, 5, 5, 151653, 11, 12, 13] + tail,
               "b": [151644, 872, 198, 151652, 5, 151653, 21, 22] + tail}

    def build(_img, instruction):
        ids = torch.tensor([prompts[instruction]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                "pixel_values": torch.zeros(4, 3), "image_grid_thw": torch.tensor([[1, 2, 2]])}

    brows = [{"image": rows[0]["image"], "instruction": "a", "label_ids": [901, 151645], "align_key": "x:home"},
             {"image": rows[0]["image"], "instruction": "b", "label_ids": [902, 151645]}]
    batch, sp = make_align_batch(build, tmp, brows, 151643, Tok())
    assert sp == [(7, 10), None], sp
    assert batch["input_ids"][0, 7:10].tolist() == [11, 12, 13]
    assert batch["labels"][0].tolist().count(-100) == len(prompts["a"])
    print("[self-test] training span: the instruction words of the prompt ids (after <|vision_end|>, before the 5 "
          "template tokens), None for rows without an align key")

    # -- AlignLoss.loss end to end on a tiny stand-in model -------------------------------------------------------
    from types import SimpleNamespace

    class StubDecoderLayer(torch.nn.Module):        # the type name carries "DecoderLayer" for find_decoder_layers
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(()))

        def forward(self, h):
            return (h * self.w,)

    class TinyVLA(torch.nn.Module):
        def __init__(self, n=25):
            super().__init__()
            self.config = SimpleNamespace(num_hidden_layers=n)
            self.emb = torch.nn.Embedding(64, HIDDEN)
            self.layers = torch.nn.ModuleList([StubDecoderLayer() for _ in range(n)])

        def forward(self, input_ids, labels, attention_mask, pixel_values, image_grid_thw):
            h = self.emb(input_ids % 64)
            for blk in self.layers:
                h = blk(h)[0]
            return SimpleNamespace(loss=(h.float() ** 2).mean())

    def build2(_img, instruction):
        k = int(instruction.split()[-1]) if instruction.startswith("sentence") else 0
        ids = torch.tensor([[151644, 872, 198, 151652, 5, 151653, 30 + k, 31 + k, 32 + k] + tail])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "pixel_values": torch.zeros(4, 3),
                "image_grid_thw": torch.tensor([[1, 2, 2]])}

    torch.manual_seed(0)
    model = TinyVLA()
    al = AlignLoss(tmp / "targets.npz", rows_path, rows, 0.5, "cpu")
    al.attach(model)
    brows = rows[:4]                                 # one common row, three corpus home rows
    total, ce, a_ = al.loss(model, build2, tmp, brows, 151643, Tok(), "cpu")
    with torch.no_grad():
        want = np.mean([1 - float(torch.nn.functional.cosine_similarity(
            model.emb(torch.tensor([30 + j, 31 + j, 32 + j])).mean(0) - al.mu_nora[li],
            al.targets[li][al.index[r["align_key"]]], dim=0))
            for li in ALIGN_LAYERS for j, r in ((int(r["raw_instruction"].split()[-1]), r) for r in brows[1:])])
    fa, ft, fc = (float(x.detach()) for x in (a_, total, ce))
    assert abs(fa - want) < 1e-5, (fa, want)
    assert abs(ft - fc - 0.5 * fa) < 1e-5 and len(al.hist_align) == 1
    total.backward()
    assert all(len(blk._forward_hooks) == 0 for blk in model.layers), "alignment hooks left behind"
    assert model.layers[6].w.grad is not None and float(model.emb.weight.grad[30:35].abs().sum()) > 0
    none = al.loss(model, build2, tmp, rows[:1], 151643, Tok(), "cpu")
    assert none[2] is None and torch.equal(none[0], none[1]), "a batch without corpus rows is cross-entropy only"
    from train_install_lora import span_ids_sha1 as trainer_span_sha1
    words = {r["align_key"]: [30 + int(r["raw_instruction"].split()[-1]) + j for j in range(3)] for r in corp}
    assert all(trainer_span_sha1(w) == span_ids_sha1(w) for w in words.values()), "the two span hash rules differ"
    right = [span_ids_sha1(words[r["align_key"]]) for r in corp]
    save_targets(tmp / "targets.npz", corp, T, np.full(len(corp), 3, dtype=np.int32), right, sha1_file(rows_path),
                 False, mu_nora)
    al2 = AlignLoss(tmp / "targets.npz", rows_path, rows, 0.5, "cpu")
    al2.attach(model)
    assert abs(float(al2.loss(model, build2, tmp, brows, 151643, Tok(), "cpu")[2].detach()) - fa) < 1e-5
    wrong = [span_ids_sha1(words[r["align_key"]][:2]) for r in corp]
    save_targets(tmp / "targets.npz", corp, T, np.full(len(corp), 2, dtype=np.int32), wrong, sha1_file(rows_path),
                 False, mu_nora)
    al3 = AlignLoss(tmp / "targets.npz", rows_path, rows, 1.0, "cpu")
    al3.attach(model)
    _expect_fail(lambda: al3.loss(model, build2, tmp, brows, 151643, Tok(), "cpu"),
                 "NORA's instruction words differ from the parent's", "differ from the parent's")
    print("[self-test] AlignLoss.loss: ce + weight x centred align (NORA mean-words - mu_nora vs target - mu_parent) "
          "from hooked block outputs, targets indexed by align key, hooks removed, gradient reaches the blocks, CE only "
          "without corpus rows, instruction-word ids checked against the parent's (hash)")
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print("[self-test] OK", flush=True)


# --------------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("corpus", "teacher", "rows", "teacher_align"), default=None)
    ap.add_argument("--out", default=str(RESULTS_DIR / "g14"))
    ap.add_argument("--f1-data", default=str(RESULTS_DIR / "f1/data"))
    ap.add_argument("--free300", default=str(RESULTS_DIR / "raw/pod8/g1/data/train_free300.jsonl"))
    ap.add_argument("--g6", default=str(RESULTS_DIR / "g6"), help="teacher: the G6 corpus + teacher for the "
                    "report-only wiring comparison, when present")
    ap.add_argument("--eval-frames", default=str(RESULTS_DIR / "raw/pod8/g1/home_frames_eval"))
    ap.add_argument("--twins", default=str(DATA_DIR / "twins.jsonl"))
    ap.add_argument("--drobot", default=str(DATA_DIR / "d_robot.jsonl"))
    ap.add_argument("--novel-benign", default=str(DATA_DIR / "novel_benign.txt"))
    ap.add_argument("--demo-prompts", default=str(DATA_DIR / "demo_prompts.txt"))
    ap.add_argument("--hold-chunk-ref", default=str(HOLD_CHUNK_REF))
    ap.add_argument("--checkpoint", default="declare-lab/nora-long-finetuned-libero-goal")
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--time-horizon", type=int, default=5)
    ap.add_argument("--stub-generator", action="store_true", help="rows, CPU: no NORA; execute labels fabricated")
    ap.add_argument("--stub-parent", action="store_true", help="teacher / teacher_align, CPU: no parent; fabricated")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test or a.stub_generator or a.stub_parent:
        cpu_shim()
    if a.self_test:
        self_test()
        return 0
    if a.stage is None:
        raise SystemExit("--stage corpus|teacher|rows|teacher_align (or --self-test)")
    if a.smoke and a.stage == "rows":
        print("[smoke] common rows: the first 40 of each type; corpus: g6 smoke_rows per source. Smoke data is NOT a "
              "result.", flush=True)
    return {"corpus": stage_corpus, "teacher": stage_teacher, "rows": stage_rows,
            "teacher_align": stage_teacher_align}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
