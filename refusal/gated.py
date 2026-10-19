"""The parent-gated policy (the parent-gated reference, Appendix B): the CONDITION is a read-out of harm (the
parent's harm direction, the parent as a judge, or a single-model decider), the RESPONSE is a control token the
LoRA was trained to obey.

Three pieces:

  fit_gate(...)      the parent harm direction, refitted LEAVE-CATEGORY-OUT: diff-in-means
                     over the D_robot harm rows of the three TRAINING categories only versus all benign rows,
                     taken in the PARENT's activations at L27 / t_inst and read in NORA at the same place.
                     Two thresholds, both at the (1 - fpr) quantile: `theta_home` from the home-pose frames of
                     init 0-9 with the benign training sentences, `theta_mid` from the 1,701 replay frames with
                     their trained instructions (scores drift up once the arm moves).

  random_gate(...)   the single-feature control: a random unit vector, thresholds fitted the same way.

  fit_judge_gate(..) the HEADLINE condition is the parent VLM run as a graded 0-10 judge
                     (`parent_harm_judge.GRADED`) on the same 224x224 frame and the raw instruction. One
                     threshold, calibrated at the same (1 - fpr) quantile over the 180 benign TRAINING
                     sentences on the calibration home frames; theta_mid is the same number, because the judge
                     sees the frame and so has no home/mid drift to correct for.

  GatedPolicy        wraps a NoraPolicy. Decides when `instruction != decided_for` (call 0 of an episode, and
                     again whenever the instruction string changes), latches the prefix, and prepends
                     "[STOP] " or "[GO] " to the instruction of EVERY call. The gate's prefill runs through
                     `NoraPolicy.prefill`, i.e. inside the same ablate/steer context as `act()`,
                     so the ablation arms really do silence the gate.

  fit_probe_gate(..) the probe gate: a logistic probe on NORA's OWN activations as the
                     gate, mode `nora_probe`, gate kind `probe`. Score = w @ ((h - mean) / scale) + b on the
                     prefill at (layer, pos); fires on score > threshold, with the same two thresholds, latch and
                     re-decision as the vector gates. No parent model anywhere.

  TextEncoder        the text gate: a small frozen sentence encoder + logistic head as the
                     gate, mode `text_clf`, gate kind `text`, fitted by
                     `experiments/3_refusal_examples/text_classifier_gate.py`. The RAW instruction (no image) is
                     embedded exactly as `text_classifier_gate.embed` does and scored as the probe:
                     w @ ((e - mean) / scale) + b, fires on score > theta. No parent, no second VLM.

  RestoreJudge       the restore gate: the resident policy model is its own judge, mode
                     `restore_judge`, gate kind `restore` (a JSON file from the restore-gate builder, which is not
                     part of this release; the arm ran no episodes). Per decision the rank-32 restore delta of the
                     single-model judges is applied in place, one text-only yes/no forward scores
                     the RAW instruction, and the deployed weights are restored from a CPU snapshot; every step is a
                     `single_model_judges` function. Fires on score > theta. No parent anywhere.

No torch import at module level: fitting, thresholds and the decision logic are numpy-only, so they can be
unit-tested on a laptop. `GatedPolicy` only calls methods on the policy object it is handed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from refusal.paths import add_experiment_dirs  # noqa: E402
add_experiment_dirs()
from refusal.directions import diff_in_means, unit  # noqa: E402

STOP_TOKEN = "[STOP] "
GO_TOKEN = "[GO] "
MODES = ("parent_judge", "parent_read", "parent_only", "random_dir", "oracle", "always_go", "nora_probe",
         "text_clf", "restore_judge")
GATE_LAYER = 27
GATE_POS = "t_inst"
HIDDEN = 2048
# Which gate `kind` each simulator arm is allowed to run with. `oracle` and `always_go` never read the gate,
# so any gate object (or the parent gate) is accepted there; the thresholds are simply unused.
KIND_OF_MODE = {"parent_only": "parent", "random_dir": "random", "parent_read": "parent_read",
                "parent_judge": "judge", "nora_probe": "probe", "text_clf": "text", "restore_judge": "restore"}
VECTOR_MODES = ("parent_only", "random_dir", "parent_read")
PROBE_MODES = ("nora_probe",)        # probe gate: a logistic probe read in NORA's own prefill; never a parent
# A probe gate file: the read-out, its site and thresholds, and the provenance of the fit.
PROBE_FIELDS = ("w", "b", "mean", "scale", "layer", "pos", "theta_home", "theta_mid", "kind")
PROBE_PROVENANCE = ("fit_ids", "fit_categories", "seed", "C")
TEXT_MODES = ("text_clf",)           # text gate: sentence encoder + logistic head on the raw instruction; no parent
# A text gate file, exactly as `experiments/3_refusal_examples/text_classifier_gate.py` writes it.
TEXT_FIELDS = ("kind", "encoder", "pooling", "l2", "mean", "scale", "w", "b", "theta_home", "theta_mid",
               "calib_scores", "eval_scores", "corpus")
TEXT_POOLINGS = ("mean", "cls")      # text_classifier_gate.pool_hidden
TEXT_MAX_LENGTH = 64                 # text_classifier_gate.embed: padding, truncation, max_length 64
TEXT_WIRING_TOL = 1e-3               # run-time scores = the saved scores within 1e-3, no decision flips
RESTORE_MODES = ("restore_judge",)   # restore gate: the deployed model + a rank-32 restore delta judges; no parent
# A restore gate file, as the restore-gate builder (not in this release) writes it from the single-model
# judges' outputs.
RESTORE_FIELDS = ("kind", "rank", "factor_file", "svd_layers", "emb_rows", "lm_head_tied", "answer_ids",
                  "theta_home", "theta_mid", "calib_scores", "eval_scores", "calib_text_sha1", "eval_text_sha1",
                  "addon_mb", "deployed_checksums", "digest")
RESTORE_WIRING_TOL = 0.02            # run-time scores = the saved rank-32 restore-judge scores within 0.02, no flips
RESTORE_WIRING_EVERY = 50            # progress line every this many wiring decisions (seconds each)
JUDGE_UNPARSED_FIRES = True   # a judge answer we cannot parse is treated as STOP: fail safe, never silent


# ----------------------------------------------------------------------------- fitting

def _acts(z, layer: int, pos: str) -> np.ndarray:
    """One (n, d) activation block out of a D_robot activation npz (keys `L{layer}_{pos}`)."""
    key = f"L{int(layer)}_{pos}"
    if key not in (z.files if hasattr(z, "files") else z):
        have = sorted(z.files if hasattr(z, "files") else z)[:6]
        raise SystemExit(f"activation file has no key {key!r}; first keys are {have}")
    a = np.asarray(z[key], dtype=np.float64)
    if a.ndim != 2:
        raise SystemExit(f"{key} has shape {a.shape}, expected (n_rows, hidden)")
    return a


def category_masks(prompts: list[dict], train_categories) -> dict:
    """Leave-category-out row masks over the D_robot prompt rows.

    `train` = harm rows whose category is in `train_categories`; `heldout` = every other harm row;
    `benign` = every non-harm row. The three are asserted disjoint and exhaustive, and the held-out rows are
    asserted absent from the training mask: that assertion is the whole point of the refit."""
    train_categories = tuple(train_categories)
    cond = np.array([r["condition"] for r in prompts])
    cat = np.array([r.get("category") or "" for r in prompts])
    harm = cond == "harm"
    train = harm & np.isin(cat, train_categories)
    heldout = harm & ~np.isin(cat, train_categories)
    benign = ~harm
    if not train.any():
        raise SystemExit(f"no D_robot harm row in the training categories {train_categories}; "
                         f"categories present: {sorted(set(cat[harm].tolist()))}")
    if not heldout.any():
        raise SystemExit(f"no held-out harm category left after removing {train_categories}; the "
                         f"leave-category-out refit would be vacuous")
    if (train & heldout).any() or (train & benign).any() or (heldout & benign).any():
        raise SystemExit("category masks overlap; the D_robot condition/category fields are inconsistent")
    if int(train.sum() + heldout.sum() + benign.sum()) != len(prompts):
        raise SystemExit("category masks do not cover every D_robot row")
    return {"train": train, "heldout": heldout, "benign": benign, "harm": harm,
            "train_categories": train_categories,
            "heldout_categories": tuple(sorted(set(cat[heldout].tolist())))}


def _scores_from(x, u: np.ndarray, what: str) -> np.ndarray:
    """`x` is either an (n, d) activation block (projected here) or an (n,) array of finished scores.

    Note on the parameter names: `benign_frames_scores` / `replay_scores` are activation blocks
    in every real call, because the direction they would have to be projected onto is fitted INSIDE this
    function. Finished 1-D scores are accepted too (unit tests, and any caller that already has them)."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 2:
        if a.shape[1] != u.shape[0]:
            raise SystemExit(f"{what}: hidden size {a.shape[1]} != direction size {u.shape[0]}")
        return a @ u
    if a.ndim == 1:
        return a
    raise SystemExit(f"{what}: shape {a.shape}, expected (n, hidden) activations or (n,) scores")


def _threshold(scores: np.ndarray, fpr: float, what: str) -> float:
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        raise SystemExit(f"{what}: empty calibration set, cannot fit a threshold")
    if not np.all(np.isfinite(s)):
        raise SystemExit(f"{what}: {int((~np.isfinite(s)).sum())} non-finite calibration scores")
    if not 0.0 < fpr < 1.0:
        raise SystemExit(f"fpr must be in (0, 1), got {fpr}")
    return float(np.quantile(s, 1.0 - fpr))


def vector_gate(u: np.ndarray, benign_frames_scores, replay_scores, fpr: float = 0.10, kind: str = "parent",
                layer: int = GATE_LAYER, pos: str = GATE_POS, extra: dict | None = None) -> dict:
    """A gate from an already-chosen direction: unit vector + the two calibrated thresholds."""
    u = unit(np.asarray(u, dtype=np.float64))
    if u.ndim != 1:
        raise SystemExit(f"gate direction has shape {u.shape}, expected a vector")
    s_home = _scores_from(benign_frames_scores, u, "benign_frames_scores")
    s_mid = _scores_from(replay_scores, u, "replay_scores")
    gate = {"u": u, "layer": int(layer), "pos": str(pos), "kind": str(kind),
            "theta_home": _threshold(s_home, fpr, "theta_home"),
            "theta_mid": _threshold(s_mid, fpr, "theta_mid"),
            "fpr": float(fpr), "n_home": int(s_home.size), "n_mid": int(s_mid.size),
            "home_score_mean": float(s_home.mean()), "mid_score_mean": float(s_mid.mean())}
    if extra:
        gate.update(extra)
    return gate


def fit_gate(parent_acts, nora_acts, prompts, train_categories, benign_frames_scores, replay_scores,
             fpr: float = 0.10, seed: int = 0, layer: int = GATE_LAYER, pos: str = GATE_POS) -> dict:
    """The parent-direction gate of the parent-gated reference.

    `parent_acts` / `nora_acts` are the D_robot activation npz files (or plain dicts of the same keys);
    `prompts` are the matching D_robot rows in the SAME order. The direction is diff-in-means in the PARENT's
    activations over the three training harm categories versus all benign rows; it is read in NORA, so both
    transfer AUROCs (training categories, held-out categories) are reported here and printed in stage 0.
    """
    from sklearn.metrics import roc_auc_score
    m = category_masks(prompts, train_categories)
    P = _acts(parent_acts, layer, pos)
    N = _acts(nora_acts, layer, pos)
    if P.shape[0] != len(prompts) or N.shape[0] != len(prompts):
        raise SystemExit(f"activation rows {P.shape[0]} / {N.shape[0]} do not match {len(prompts)} prompt rows")
    if P.shape[1] != N.shape[1]:
        raise SystemExit(f"parent hidden size {P.shape[1]} != NORA hidden size {N.shape[1]}")
    v = diff_in_means(P[m["train"]], P[m["benign"]])
    u = unit(v)
    sn, sp = N @ u, P @ u

    def au(s, pos_mask):
        y = np.concatenate([np.ones(int(pos_mask.sum())), np.zeros(int(m["benign"].sum()))])
        return float(roc_auc_score(y, np.concatenate([s[pos_mask], s[m["benign"]]])))

    extra = {"seed": int(seed), "train_categories": list(m["train_categories"]),
             "heldout_categories": list(m["heldout_categories"]),
             "n_train_harm": int(m["train"].sum()), "n_heldout_harm": int(m["heldout"].sum()),
             "n_benign": int(m["benign"].sum()),
             "auroc_nora_train_cats": au(sn, m["train"]), "auroc_nora_heldout_cats": au(sn, m["heldout"]),
             "auroc_parent_train_cats": au(sp, m["train"]), "auroc_parent_heldout_cats": au(sp, m["heldout"]),
             "drobot_nora_mean_train": float(sn[m["train"]].mean()),
             "drobot_nora_mean_heldout": float(sn[m["heldout"]].mean()),
             "drobot_nora_mean_benign": float(sn[m["benign"]].mean()),
             "drobot_parent_median_harm": float(np.median(sp[m["train"]])),
             "drobot_parent_median_benign": float(np.median(sp[m["benign"]])),
             "dir_norm": float(np.linalg.norm(v))}
    return vector_gate(u, benign_frames_scores, replay_scores, fpr=fpr, kind="parent", layer=layer, pos=pos,
                       extra=extra)


def random_gate(seed: int, benign_frames_scores=None, replay_scores=None, fpr: float = 0.10,
                dim: int = HIDDEN, layer: int = GATE_LAYER, pos: str = GATE_POS) -> dict:
    """The single-feature random-direction control: a fixed random unit vector, the same
    two thresholds fitted on the same calibration sets. The seed alone is not enough: a
    threshold cannot exist without a calibration set, so both are required rather than silently defaulted."""
    if benign_frames_scores is None or replay_scores is None:
        raise SystemExit("random_gate needs the same calibration sets as fit_gate "
                         "(benign_frames_scores, replay_scores); there is no default threshold")
    u = unit(np.random.default_rng(int(seed)).standard_normal(int(dim)))
    return vector_gate(u, benign_frames_scores, replay_scores, fpr=fpr, kind="random", layer=layer, pos=pos,
                       extra={"seed": int(seed), "dim": int(dim)})


def parse_judge(raw: str) -> float | None:
    """The 0-10 graded answer out of the parent's generation (`parent_harm_judge.parse_score`), or None.

    None is never silently turned into 0: `GatedPolicy` treats an unparsable judge answer as a STOP
    (`JUDGE_UNPARSED_FIRES`) and counts it, and stage 0 prints the unparsed rate."""
    from parent_harm_judge import clean, parse_score
    s = parse_score(clean(raw))
    return None if s is None else float(s)


def fit_judge_gate(benign_frames_scores, fpr: float = 0.10, n_cal: int | None = None) -> dict:
    """Addendum B: the parent-judge gate. One calibrated threshold over the benign TRAINING sentences scored
    on the calibration home frames; `theta_home` and `theta_mid` are both that threshold."""
    s = np.asarray(benign_frames_scores, dtype=np.float64).ravel()
    thr = _threshold(s, fpr, "judge threshold")
    return {"kind": "judge", "threshold": float(thr), "theta_home": float(thr), "theta_mid": float(thr),
            "fpr": float(fpr), "n_cal": int(n_cal if n_cal is not None else s.size),
            "cal_score_mean": float(s.mean()), "cal_score_max": float(s.max())}


def probe_arrays(gate: dict) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """(w, b, mean, scale) of a probe gate as float64, checked: equal lengths, finite, scale > 0."""
    w = np.asarray(gate["w"], dtype=np.float64).reshape(-1)
    mean = np.asarray(gate["mean"], dtype=np.float64).reshape(-1)
    scale = np.asarray(gate["scale"], dtype=np.float64).reshape(-1)
    b = float(np.asarray(gate["b"], dtype=np.float64))
    if not (w.shape == mean.shape == scale.shape):
        raise RuntimeError(f"probe gate arrays disagree: w {w.shape}, mean {mean.shape}, scale {scale.shape}")
    if not (np.isfinite(w).all() and np.isfinite(mean).all() and np.isfinite(scale).all() and np.isfinite(b)):
        raise RuntimeError("probe gate has non-finite w / b / mean / scale")
    if not (scale > 0).all():
        raise RuntimeError(f"probe gate has {int((scale <= 0).sum())} non-positive scale entries")
    return w, b, mean, scale


def probe_score(gate: dict, h) -> np.ndarray | float:
    """The probe-gate score: w @ ((h - mean) / scale) + b, for one activation (d,) or a block (n, d).

    This is exactly `StandardScaler` + `LogisticRegression.decision_function`, written out so the run-time gate
    and every offline recomputation use one formula (asserted against sklearn in `fit_probe_gate`)."""
    w, b, mean, scale = probe_arrays(gate)
    x = np.asarray(h, dtype=np.float64)
    if x.shape[-1] != w.shape[0]:
        raise RuntimeError(f"activation of width {x.shape[-1]} against a probe of width {w.shape[0]}")
    s = ((x - mean) / scale) @ w + b
    return float(s) if np.ndim(s) == 0 else np.asarray(s, dtype=np.float64)


def fit_probe_gate(X, y, benign_frames_acts, replay_acts, *, fpr: float, C: float, max_iter: int, seed: int,
                   layer: int, pos: str, extra: dict | None = None) -> dict:
    """The probe gate: StandardScaler + LogisticRegression on (X, y), exported as a probe gate.

    `X` (n, d) are the fit rows' prefill activations, `y` 1 = harm / 0 = benign. The two thresholds come from
    OTHER rows only: `theta_home` = the (1 - fpr) quantile of the scores of `benign_frames_acts` (benign training
    sentences on the calibration home frames) and `theta_mid` of `replay_acts` (replay frames with their trained
    instructions), exactly as `vector_gate` does for the direction gates. The caller asserts that the fit rows and
    the calibration rows are disjoint; this function only fits and exports.

    The exported (w, b, mean, scale) are checked against sklearn's own decision_function on the fit rows, so a
    wrong export can never silently become the run-time gate."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise SystemExit(f"probe fit: X {X.shape} and y {y.shape} do not match")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise SystemExit(f"probe fit needs both classes, got labels {sorted(set(y.tolist()))}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=float(C), max_iter=int(max_iter), random_state=int(seed)).fit(sc.transform(X), y)
    n_conv = sum(1 for c in caught if issubclass(c.category, ConvergenceWarning))
    gate = {"kind": "probe", "w": np.asarray(clf.coef_[0], dtype=np.float64),
            "b": float(clf.intercept_[0]), "mean": np.asarray(sc.mean_, dtype=np.float64),
            "scale": np.asarray(sc.scale_, dtype=np.float64), "layer": int(layer), "pos": str(pos)}
    s_fit = probe_score(gate, X)
    ref = clf.decision_function(sc.transform(X))
    gap = float(np.max(np.abs(s_fit - ref)))
    if gap > 1e-6:
        raise SystemExit(f"probe export disagrees with sklearn decision_function by {gap:.2e} on the fit rows")
    for what, A in (("benign_frames_acts", benign_frames_acts), ("replay_acts", replay_acts)):
        A = np.asarray(A)
        if A.ndim != 2 or A.shape[1] != X.shape[1]:
            raise SystemExit(f"{what}: shape {A.shape}, expected (n, {X.shape[1]}) activations")
    s_home = probe_score(gate, benign_frames_acts)
    s_mid = probe_score(gate, replay_acts)
    gate.update({"theta_home": _threshold(s_home, fpr, "theta_home"),
                 "theta_mid": _threshold(s_mid, fpr, "theta_mid"),
                 "fpr": float(fpr), "n_home": int(s_home.size), "n_mid": int(s_mid.size),
                 "home_score_mean": float(s_home.mean()), "mid_score_mean": float(s_mid.mean()),
                 "C": float(C), "max_iter": int(max_iter), "seed": int(seed), "n_iter": int(clf.n_iter_[0]),
                 "n_convergence_warnings": int(n_conv), "n_fit": int(y.size), "n_fit_harm": int(y.sum()),
                 "n_fit_benign": int((y == 0).sum()), "train_auroc": float(roc_auc_score(y, s_fit)),
                 "export_max_abs_diff": gap})
    if extra:
        gate.update(extra)
    return gate


def cuda_mem_gb() -> float:
    """Allocated CUDA memory in GB, 0.0 without a GPU. Addendum B reports the parent's resident cost."""
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.memory_allocated()) / 1e9
    except Exception:
        pass
    return 0.0


# ----------------------------------------------------------------------------- persistence / reporting

def save_gate(path, gate: dict) -> None:
    """`gate.npz` = {u, layer, pos, theta_home, theta_mid, kind} plus every scalar the fit recorded."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for k, v in gate.items():
        if isinstance(v, np.ndarray):
            flat[k] = v                      # float64 kept: the thresholds were fitted with this exact `u`
        elif isinstance(v, str):
            flat[k] = np.array(v)
        elif isinstance(v, (list, tuple)):
            flat[k] = np.array([str(x) for x in v])
        elif isinstance(v, (bool, int, float, np.integer, np.floating)):
            flat[k] = np.asarray(v, dtype=np.float64 if isinstance(v, float) else np.int64)
        else:
            raise SystemExit(f"gate field {k!r} of type {type(v).__name__} cannot be saved")
    np.savez(path, **flat)


def load_gate(path) -> dict:
    if Path(path).suffix == ".json":        # restore gate: a JSON file
        import json
        gate = json.loads(Path(path).read_text())
        for k in ("calib_scores", "eval_scores"):
            if k in gate:
                gate[k] = np.asarray(gate[k], dtype=np.float64)
    else:
        z = np.load(path, allow_pickle=False)
        gate = {}
        for k in z.files:
            v = z[k]
            if k == "u":
                gate[k] = np.asarray(v, dtype=np.float64)
            elif v.dtype.kind in ("U", "S"):
                gate[k] = str(v) if v.ndim == 0 else [str(x) for x in v]
            elif v.ndim == 0:
                gate[k] = int(v) if v.dtype.kind == "i" else float(v)
            else:
                gate[k] = np.asarray(v)
    if gate.get("kind") == "probe":         # probe gate: the read-out, its thresholds and where the fit came from
        need = PROBE_FIELDS + PROBE_PROVENANCE
    elif gate.get("kind") == "text":        # text gate: the encoder, the head, the threshold and the saved scores
        need = TEXT_FIELDS
    elif gate.get("kind") == "restore":     # restore gate: factor file, rank, threshold, saved scores
        need = RESTORE_FIELDS
    else:
        need = ("threshold", "kind") if gate.get("kind") == "judge" else\
            ("u", "layer", "pos", "theta_home", "theta_mid", "kind")
    for field in need:
        if field not in gate:
            raise SystemExit(f"{path} is not a {gate.get('kind')} gate file: no {field!r}")
    return gate


def gate_line(name: str, gate: dict) -> str:
    """One row of the stage-0 gate table. Judge gates carry no layer / position."""
    layer = gate.get("layer")
    lay = "-" if layer is None else f"L{int(layer)}"
    n_cal = int(gate.get("n_cal", 0))
    return (f"{name:<14s}{gate['kind']:<12s}{lay:<5s}{str(gate.get('pos', '-')):<13s}"
            f"{gate['theta_home']:+11.3f}{gate['theta_mid']:+11.3f}"
            f"{int(gate.get('n_home', n_cal)):8d}{int(gate.get('n_mid', n_cal)):8d}")


def gate_header() -> str:
    return (f"{'gate':<14s}{'kind':<12s}{'layer':<5s}{'pos':<13s}{'theta_home':>11s}{'theta_mid':>11s}"
            f"{'n_home':>8s}{'n_mid':>8s}")


def print_prefix_tokenization(tokenizer) -> dict:
    """Red-team finding 16: "[STOP]" / "[GO]" are plain text, not new vocabulary. Print the subword pieces
    once so the log records what the model actually sees."""
    out = {}
    for name, tok in (("STOP", STOP_TOKEN), ("GO", GO_TOKEN)):
        ids = list(tokenizer(tok, add_special_tokens=False)["input_ids"])
        pieces = tokenizer.convert_ids_to_tokens(ids)
        out[name] = {"text": tok, "ids": [int(i) for i in ids], "pieces": [str(p) for p in pieces]}
        print(f"[gate] prefix {name}: {tok!r} -> {len(ids)} subword pieces {pieces} ids {ids}", flush=True)
    return out


# ----------------------------------------------------------------------------- the parent as a gate reader

def _as_pil(image):
    import PIL.Image
    if image is None or isinstance(image, PIL.Image.Image):
        return image
    return PIL.Image.fromarray(np.asarray(image, dtype=np.uint8))


class ParentReader:
    """The parent VLM as a gate, for both parent-side arms.

    `prefill(...)`     one parent forward, residual stream at the same layer / position as `parent_read`.
    `judge_score(...)` one parent generation with the `parent_harm_judge` graded 0-10 prompt on the same
                       224x224 frame and the RAW instruction; this is the headline `parent_judge` gate.

    One parent call per DECISION, not per policy call. `mem_gb` is the resident cost of the parent measured by
    `load_parent_reader`; it is what the cost comparison of the parent-gated reference reports."""

    def __init__(self, loaded, layers=(GATE_LAYER,), mem_gb: float | None = None):
        from refusal.models import ActivationCache  # late: torch
        self.L = loaded
        self.layers = [int(li) for li in layers]
        self.cache = ActivationCache(loaded, self.layers)
        self.mem_gb = mem_gb
        self.n_judged = 0
        self.n_unparsed = 0
        self.ablate = None   # {"vec": ndarray, "layers": list|None}: test-4 wiring assertion ablates the PARENT's forward

    def _ctx(self):
        import contextlib
        if self.ablate is None:
            return contextlib.nullcontext()
        from refusal.steering import ablate_direction
        layers = self.ablate.get("layers")
        return ablate_direction(self.L, self.ablate["vec"],
                                list(range(len(self.L.layers))) if layers is None else list(layers))

    def prefill(self, image, instruction, layers, positions) -> dict:
        from refusal.models import residuals
        missing = [li for li in layers if int(li) not in self.layers]
        if missing:
            raise RuntimeError(f"ParentReader was built for layers {self.layers}, asked for {missing}")
        with self._ctx():
            acts, _ = residuals(self.L, self.cache, instruction, _as_pil(image))
        idx = {"t_inst": 0, "t_post_inst": 1}
        bad = [p for p in positions if p not in idx]
        if bad:
            raise RuntimeError(f"ParentReader: unknown positions {bad}")
        return {(int(li), p): acts[int(li)][idx[p]].numpy().astype(np.float32)
                for li in layers for p in positions}

    def judge_raw(self, image, instruction: str, max_new_tokens: int = 8) -> str:
        from parent_harm_judge import GRADED
        from refusal.models import generate
        with self._ctx():
            return generate(self.L, GRADED.format(p=instruction), _as_pil(image), max_new_tokens)

    def judge_score(self, image, instruction: str) -> tuple[float | None, str]:
        """(graded 0-10 score or None, the raw generation). The caller decides what None means."""
        raw = self.judge_raw(image, instruction)
        v = parse_judge(raw)
        self.n_judged += 1
        if v is None:
            self.n_unparsed += 1
        return v, raw


class TextEncoder:
    """The frozen sentence encoder of a text gate, resident next to NORA.

    `embed(texts)` is `text_classifier_gate.embed` moved to a device: the same tokenizer call (padding,
    truncation, max_length 64), the same `pool_hidden` (imported from that file: mean pooling over the attention
    mask, or CLS) and the same L2 normalisation when `l2`, in fp32, eval mode, under `torch.inference_mode()`.
    `mem_mb` is the encoder's resident cost: parameter bytes / 1e6."""

    def __init__(self, repo: str, pooling: str, l2: bool, device: str):
        import torch
        from transformers import AutoModel, AutoTokenizer
        if pooling not in TEXT_POOLINGS:
            raise RuntimeError(f"text gate pooling {pooling!r} is not one of {TEXT_POOLINGS}")
        self.repo, self.pooling, self.l2, self.device = str(repo), str(pooling), bool(l2), str(device)
        self.tok = AutoTokenizer.from_pretrained(self.repo)
        # no dtype kwarg: transformers 4.50 calls it torch_dtype, 5.x dtype; cast explicitly instead
        self.model = AutoModel.from_pretrained(self.repo).to(device=self.device, dtype=torch.float32).eval()
        self.dim = int(self.model.config.hidden_size)
        self.n_params = int(sum(p.numel() for p in self.model.parameters()))
        self.mem_mb = float(sum(p.numel() * p.element_size() for p in self.model.parameters())) / 1e6

    def embed(self, texts) -> np.ndarray:
        import torch
        from text_classifier_gate import pool_hidden
        with torch.inference_mode():
            b = self.tok(list(texts), padding=True, truncation=True, max_length=TEXT_MAX_LENGTH,
                         return_tensors="pt").to(self.device)
            e = pool_hidden(self.model(**b), b["attention_mask"], self.pooling)
            if self.l2:
                e = torch.nn.functional.normalize(e, dim=-1)
            return e.float().cpu().numpy()


def load_text_encoder(gate: dict, device: str) -> TextEncoder:
    """The encoder a text gate names (`encoder`, `pooling`, `l2`), loaded once and shared by every text_clf arm."""
    import time
    t0 = time.perf_counter()
    enc = TextEncoder(gate["encoder"], gate["pooling"], bool(int(gate["l2"])), device)
    print(f"[gate] text encoder {enc.repo} ({enc.pooling} pooling, l2 {int(enc.l2)}) on {enc.device}, fp32: "
          f"{enc.n_params / 1e6:.1f} M parameters, {enc.mem_mb:.1f} MB resident, dim {enc.dim}, loaded in "
          f"{time.perf_counter() - t0:.1f}s", flush=True)
    return enc


class _NoPolicy:
    """Stands in for NoraPolicy in the text-gate wiring check: a text gate never reads the policy."""

    def __getattr__(self, name):
        raise RuntimeError(f"the text-gate wiring check touched policy.{name}")


def _wiring_scores(gp, gate: dict, texts: dict, tol: float, what: str, every: int | None = None) -> tuple:
    """The wiring loop shared by the text and restore gates: every sentence of `texts` = {"eval": [...],
    "calib": [...]} goes through `gp._decide` (a fresh decision at call 0 each, i.e. against theta_home) and its
    score is compared with the saved one (`eval_scores` / `calib_scores`). Prints per set the max |d|, the fire
    counts and the flipped decisions (set, index, the two scores; never the text); `every` adds a progress line.
    Returns (summary dict, list of failures, per-decision latencies, per-decision judge-forward times)."""
    out, bad = {"tol": float(tol), "theta": gp.theta_home}, []
    lat, judge = [], []
    for name in ("eval", "calib"):
        saved = np.asarray(gate[f"{name}_scores"], dtype=np.float64).reshape(-1)
        ts = list(texts[name])
        if len(ts) != saved.size:
            raise RuntimeError(f"{what} wiring check: {len(ts)} {name} sentences against {saved.size} saved scores")
        got, fired = np.zeros(saved.size), np.zeros(saved.size, dtype=bool)
        for k, t in enumerate(ts):
            gp.reset()
            gp._decide(None, t, None)
            got[k], fired[k] = gp.last_score, gp.fired
            lat.append(gp.last_latency)
            if gp.last_judge_s is not None:
                judge.append(gp.last_judge_s)
            if every and (k + 1) % every == 0:
                print(f"[wiring] {name} {k + 1}/{saved.size} decided, max |d| so far "
                      f"{float(np.abs(got[:k + 1] - saved[:k + 1]).max()):.2e}, {float(np.mean(lat)):.2f} s per "
                      f"decision", flush=True)
        d = np.abs(got - saved)
        flips = np.flatnonzero(fired != (saved > gp.theta_home))
        out[name] = {"n": int(saved.size), "max_abs_d": float(d.max()), "mean_abs_d": float(d.mean()),
                     "n_fire_saved": int((saved > gp.theta_home).sum()), "n_fire_now": int(fired.sum()),
                     "n_flipped": int(flips.size), "flipped_idx": [int(i) for i in flips]}
        print(f"[wiring] {name:<5s} {saved.size:4d} sentences: max |d| {d.max():.2e} (mean {d.mean():.2e}), "
              f"fires saved {out[name]['n_fire_saved']} / now {out[name]['n_fire_now']}, decisions flipped "
              f"{flips.size}", flush=True)
        for i in flips:
            print(f"[wiring]   flipped {name}[{int(i)}]: saved {saved[i]:+.6f}, now {got[i]:+.6f}, theta "
                  f"{gp.theta_home:+.6f}", flush=True)
        if d.max() > tol:
            bad.append(f"{name}: max |d| {d.max():.2e} > {tol:g}")
        if flips.size:
            bad.append(f"{name}: {flips.size} fire decisions differ from the saved ones")
    gp.reset()
    return out, bad, lat, judge


def text_wiring_check(gate: dict, text_encoder, texts: dict, tol: float = TEXT_WIRING_TOL) -> dict:
    """Text-gate wiring check, asserted before any episode: every sentence of
    `texts` = {"eval": [...], "calib": [...]} goes through `GatedPolicy._decide` (mode text_clf, a fresh decision
    at call 0 each, i.e. against theta_home) and its score is compared with the score `text_classifier_gate` saved
    (`eval_scores` / `calib_scores`).

    Raises RuntimeError when max |d| > tol or any fire decision differs from the saved one (saved score > theta).
    Prints both numbers and, for every differing decision, the set, index and the two scores (never the text)."""
    gp = GatedPolicy(_NoPolicy(), gate, "text_clf", text_encoder=text_encoder)
    out, bad, lat, _ = _wiring_scores(gp, gate, texts, tol, "text")
    out["latency_s_mean"] = float(np.mean(lat))
    out["mem_mb"] = float(text_encoder.mem_mb)
    if bad:
        raise RuntimeError("text-gate wiring check FAILED: " + "; ".join(bad))
    print(f"[wiring] PASS: max |d| {max(out['eval']['max_abs_d'], out['calib']['max_abs_d']):.2e} <= {tol:g}, "
          f"0 decisions flipped; {out['latency_s_mean'] * 1e3:.1f} ms per decision on {text_encoder.device}",
          flush=True)
    return out


class RestoreJudge:
    """The restore-gate judge, built once per run on the RESIDENT policy and
    shared by every restore_judge arm (as `TextEncoder` is for text_clf).

    A decision is exactly the first-call path of the single-model judges, every step a `single_model_judges` function:
    `apply_restore(tg, fac, rank, device, all layers, emb_rows)` on the policy's own weights, `judge_scores` on
    YESNO(raw instruction) with no image (s = logsumexp(yes, Yes) - logsumexp(no, No) at the last prompt
    position), then `restore_originals(tg, orig)` from the CPU snapshot taken once here. `sums0` are the
    three-matrix `checksums` of the deployed weights at construction; `assert_deployed` compares against them.

    `fac` keeps only the rank-`rank` columns of the factor file (the file holds up to 128): what stays resident is
    exactly the single-model judges' `addon_mb` at this rank (asserted). `snapshot_mb` is the CPU copy of every
    tensor a decision
    writes (the whole language model and the embedding rows below `emb_rows`)."""

    def __init__(self, policy, gate: dict):
        import time
        import torch
        from probe_gate_first_call import Report
        from single_model_judges import (ANSWER_NO, ANSWER_YES, PUSH_RANK, addon_mb,
                                                        answer_token_ids, checksums, load_factors, restore_targets,
                                                        snapshot_originals)
        from refusal.models import Loaded
        t0 = time.perf_counter()
        self.policy, self.model, self.device = policy, policy.model, policy.device
        self.rank, self.factor_file = int(gate["rank"]), str(gate["factor_file"])
        self.n_rows = int(gate["emb_rows"])
        if not 0 < self.rank <= PUSH_RANK:
            raise RuntimeError(f"restore rank {self.rank} is outside 1..{PUSH_RANK} (the factor file's columns)")
        tok = policy.processor.tokenizer
        self.L = Loaded("deployed", "vlm", policy.model, policy.processor, tok, policy.layers, policy.device)
        self.ids = answer_token_ids(tok, "deployed", Report())
        want_ids = {str(k): int(v) for k, v in dict(gate["answer_ids"]).items()}
        if self.ids != want_ids:
            raise RuntimeError(f"answer token ids {self.ids} differ from the ones G6 scored with {want_ids}")
        self.tg = restore_targets(policy.model, policy.layers)
        self.layers = [int(i) for i in gate["svd_layers"]]
        if self.layers != list(range(self.tg["n_layers"])):
            raise RuntimeError(f"the gate's restore layers {self.layers[:3]}.. ({len(self.layers)}) are not all "
                               f"{self.tg['n_layers']} decoder layers of the resident model")
        if bool(self.tg["tied"]) != bool(gate["lm_head_tied"]):
            raise RuntimeError(f"lm_head tied {self.tg['tied']} here, {gate['lm_head_tied']} in G6")
        head_ids = [self.ids[w] for w in ANSWER_YES + ANSWER_NO]
        self.orig = snapshot_originals(self.tg, self.n_rows, head_ids)
        self.snapshot_mb = float(sum(v.numel() * v.element_size() for v in self.orig.values()
                                     if isinstance(v, torch.Tensor))) / 1e6
        self.sums0 = checksums(self.tg)
        self.same_as_g6 = self.sums0 == dict(gate["deployed_checksums"])
        full = load_factors({"fac_push": Path(self.factor_file)}, self.rank)
        self.fac = {}
        for k, v in full.items():
            if k.startswith("L") and k.endswith(".US"):
                if v.shape[1] < self.rank:
                    raise RuntimeError(f"{k} has {v.shape[1]} columns, fewer than rank {self.rank}")
                self.fac[k] = v[:, :self.rank].contiguous()
            elif k.startswith("L") and k.endswith(".Vh"):
                self.fac[k] = v[:self.rank].contiguous()
            elif not k.startswith("emb.") or k.startswith(f"emb.r{self.rank}."):
                self.fac[k] = v                          # small.*, head.* and this rank's embedding factors
        del full
        self.addon_mb = float(addon_mb(self.tg, self.rank, self.n_rows))
        resident = float(sum(v.numel() * v.element_size() for k, v in self.fac.items() if k != "head.ids")) / 1e6
        if abs(resident - self.addon_mb) > 1e-6 or abs(self.addon_mb - float(gate["addon_mb"])) > 1e-6:
            raise RuntimeError(f"rank-{self.rank} factors resident {resident:.6f} MB, add-on {self.addon_mb:.6f} MB "
                               f"here, {float(gate['addon_mb']):.6f} MB in G6")
        self.n_scored = 0
        print(f"[gate] restore judge: rank-{self.rank} delta of {self.factor_file} on the resident model "
              f"({len(self.layers)} layers, embedding rows < {self.n_rows}); add-on {self.addon_mb:.2f} MB (G6 "
              f"addon_mb), CPU snapshot {self.snapshot_mb:.1f} MB of {len(self.orig)} entries; deployed checksums "
              f"{'IDENTICAL to' if self.same_as_g6 else 'DIFFERENT from'} G6's ({', '.join(self.sums0)}); built in "
              f"{time.perf_counter() - t0:.1f}s", flush=True)

    def assert_deployed(self, where: str) -> None:
        """The three-matrix checksum equals the construction-time value, else the action path would see the delta."""
        from single_model_judges import checksums
        now = checksums(self.tg)
        if now != self.sums0:
            bad = [k for k in now if now[k] != self.sums0[k]]
            raise RuntimeError(f"restore gate: deployed weights differ from their construction-time checksums {where} "
                               f"({bad})")

    def score(self, instruction: str) -> tuple[float, float]:
        """(judge score of the RAW instruction with the delta applied, seconds of the judge forward). The weights
        are restored even when the apply or the forward raises."""
        import contextlib
        import io
        import time
        import torch
        from single_model_judges import apply_restore, judge_scores, restore_originals
        try:
            apply_restore(self.tg, self.fac, self.rank, self.device, self.layers, self.n_rows)
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):     # judge_scores prints a timing line per call
                s, _ = judge_scores(self.L, [instruction], self.ids, "g10/judge")
            judge_s = time.perf_counter() - t0
        finally:
            restore_originals(self.tg, self.orig)
            if str(self.device).startswith("cuda"):
                torch.cuda.synchronize()
        self.n_scored += 1
        return float(s[0]), judge_s


def restore_wiring_check(gate: dict, restore_judge, policy, texts: dict, tol: float = RESTORE_WIRING_TOL) -> dict:
    """Restore-gate wiring check, asserted before any episode: the 193 evaluation and 180 calibration sentences
    go through `GatedPolicy._decide` (mode restore_judge: apply, judge, restore, checksum) and their scores are
    compared with the saved rank-32 restore-judge scores. The sentences must be the single-model judges', in their
    order (sha1 of each text, asserted).

    Raises RuntimeError when max |d| > tol or any fire decision differs from the saved one (saved score > theta).
    Prints max |d|, the flips and the mean latency (apply + judge + restore) and judge-forward time."""
    from probe_gate_first_call import sha
    for name in ("eval", "calib"):
        want, got = [str(x) for x in gate[f"{name}_text_sha1"]], [sha(t) for t in texts[name]]
        if got != want:
            n_diff = sum(a != b for a, b in zip(got, want)) + abs(len(got) - len(want))
            raise RuntimeError(f"restore wiring check: the {len(got)} {name} sentences are not G6's {len(want)} in "
                               f"G6 order ({n_diff} differ)")
    gp = GatedPolicy(policy, gate, "restore_judge", restore_judge=restore_judge)
    out, bad, lat, judge = _wiring_scores(gp, gate, texts, tol, "restore", every=RESTORE_WIRING_EVERY)
    out.update({"latency_s_mean": float(np.mean(lat)), "judge_s_mean": float(np.mean(judge)),
                "addon_mb": restore_judge.addon_mb, "snapshot_mb": restore_judge.snapshot_mb,
                "deployed_checksums": restore_judge.sums0, "deployed_same_as_g6": bool(restore_judge.same_as_g6),
                "n_flipped": out["eval"]["n_flipped"] + out["calib"]["n_flipped"],
                "max_abs_d": max(out["eval"]["max_abs_d"], out["calib"]["max_abs_d"])})
    print(f"[wiring] restore: max |d| {out['max_abs_d']:.2e} (tol {tol:g}), {out['n_flipped']} decisions flipped, "
          f"{out['latency_s_mean']:.2f} s per decision (apply + judge + restore; judge forward "
          f"{out['judge_s_mean']:.3f} s) on {restore_judge.device}", flush=True)
    if bad:
        raise RuntimeError("restore-gate wiring check FAILED: " + "; ".join(bad))
    print(f"[wiring] PASS: max |d| {out['max_abs_d']:.2e} <= {tol:g}, 0 decisions flipped", flush=True)
    return out


def load_parent_reader(layers=(GATE_LAYER,), device=None):
    """Load the parent VLM and measure what it costs to keep it resident next to NORA.

    `torch.cuda.memory_allocated()` is read immediately before and after `load_model("parent")`, so `mem_gb`
    is the parent's own delta rather than the whole process's footprint."""
    from refusal.models import load_model
    before = cuda_mem_gb()
    L = load_model("parent", device)
    after = cuda_mem_gb()
    reader = ParentReader(L, layers=layers, mem_gb=round(after - before, 3))
    print(f"[gate] parent VLM resident: {reader.mem_gb:.2f} GB (allocated {before:.2f} -> {after:.2f} GB)",
          flush=True)
    return reader


# ----------------------------------------------------------------------------- the gated policy

class GatedPolicy:
    """NoraPolicy + a control-token prefix decided by the gate (the parent-gated policy).

    `mode`
      parent_judge the parent VLM run as a graded 0-10 judge on the same frame and the RAW instruction, at
                   the threshold calibrated in stage 0 (HEADLINE arm). One parent generation per decision.
      parent_read  the leave-category-out harm direction read in the PARENT's own activations (secondary).
                   One parent prefill per decision.
      parent_only  the same direction read in NORA's activations (drift comparison). One NORA prefill.
      random_dir   a random unit vector read in NORA, same calibration (control)
      oracle       token from the ground-truth `harm_label` (ceiling; gives P(hold | STOP))
      always_go    the gate is off, every call is "[GO] " (the LoRA's own behaviour)
      nora_probe   probe gate: a logistic probe (gate kind `probe`) read in NORA's own prefill of the RAW instruction.
                   One NORA prefill per decision, no parent (passing one is an error, as for parent_only).
      text_clf     text gate: a small frozen sentence encoder + logistic head (gate kind `text`) on the RAW instruction,
                   no image. One encoder forward per decision on `text_encoder` (a `TextEncoder`, loaded once by
                   the caller); no parent (passing one is an error). theta_home = theta_mid (asserted).
      restore_judge restore gate: the policy's OWN model with the rank-32 restore delta applied in place judges the RAW
                   instruction, text only (gate kind `restore`; `restore_judge` = a `RestoreJudge` built once on
                   this policy by the caller). Per decision: apply, one forward, restore, checksum asserted; the
                   checksum is asserted again before every `act`, so the action forward never sees the delta. No
                   parent (passing one is an error). theta_home = theta_mid (asserted).

    A decision happens when `instruction != decided_for`: at call 0 of an episode against `theta_home`, and at
    every later instruction change against `theta_mid` (the moving-arm threshold; for the judge gate the two
    are the same number, because the judge sees the frame). Firing LATCHES: once the episode is in "[STOP] "
    it stays there, so a harm -> benign swap cannot un-refuse. The prefix is latched too, i.e. it is prepended
    to every call until the next decision.

    Cost accounting: every decision records `latency_s`; `parent_mem_gb` (the parent's resident
    delta, from `load_parent_reader`) and `total_mem_gb` (allocated with both models up) are read once at
    construction; `text_mem_mb` is the text encoder's parameter memory, None for every other mode; the
    restore judge carries its add-on and CPU snapshot size, and each of its decisions records `judge_s`.
    """

    def __init__(self, policy, gate, mode: str, parent_policy=None, text_encoder=None, restore_judge=None):
        if mode not in MODES:
            raise RuntimeError(f"unknown gate mode {mode!r}; have {MODES}")
        is_judge = gate.get("kind") == "judge"
        is_probe = gate.get("kind") == "probe"
        is_text = gate.get("kind") == "text"
        is_restore = gate.get("kind") == "restore"
        if is_probe:
            need = PROBE_FIELDS + PROBE_PROVENANCE
        elif is_text:
            need = TEXT_FIELDS
        elif is_restore:
            need = RESTORE_FIELDS
        else:
            need = ("threshold", "kind") if is_judge else ("u", "layer", "pos", "theta_home", "theta_mid", "kind")
        for field in need:
            if field not in gate:
                raise RuntimeError(f"gate has no {field!r}; load it with src.gated.load_gate")
        want = KIND_OF_MODE.get(mode)
        if want is not None and gate["kind"] != want:
            raise RuntimeError(f"mode {mode!r} needs a gate of kind {want!r}, got {gate['kind']!r}")
        needs_parent = mode in ("parent_read", "parent_judge")
        if needs_parent and parent_policy is None:
            raise RuntimeError(f"mode {mode!r} needs `parent_policy` (src.gated.load_parent_reader())")
        if not needs_parent and parent_policy is not None:
            raise RuntimeError(f"mode {mode!r} was given a parent_policy; it would never be used")
        if mode == "parent_judge" and not hasattr(parent_policy, "judge_score"):
            raise RuntimeError("mode 'parent_judge' needs a parent with .judge_score(image, instruction)")
        if mode in TEXT_MODES and text_encoder is None:
            raise RuntimeError(f"mode {mode!r} needs `text_encoder` (src.gated.load_text_encoder(gate, device))")
        if mode not in TEXT_MODES and text_encoder is not None:
            raise RuntimeError(f"mode {mode!r} was given a text_encoder; it would never be used")
        if mode in RESTORE_MODES and restore_judge is None:
            raise RuntimeError(f"mode {mode!r} needs `restore_judge` (src.gated.RestoreJudge(policy, gate))")
        if mode not in RESTORE_MODES and restore_judge is not None:
            raise RuntimeError(f"mode {mode!r} was given a restore_judge; it would never be used")
        self.p = policy
        self.gate = gate
        self.mode = mode
        self.parent = parent_policy
        self.is_judge = bool(is_judge)
        self.is_probe = bool(is_probe)
        self.is_text = bool(is_text)
        self.probe = None
        self.text = None
        self.text_mem_mb = None
        self.restore = None
        if is_judge:
            self.u = None
            self.layer, self.pos = None, None
            self.theta_home = self.theta_mid = float(gate["threshold"])
        elif is_probe:
            self.u = None
            w, b, mean, scale = probe_arrays(gate)      # a malformed probe fails at construction, not mid-episode
            self.probe = {"w": w, "b": b, "mean": mean, "scale": scale}
            self.layer, self.pos = int(gate["layer"]), str(gate["pos"])
            self.theta_home, self.theta_mid = float(gate["theta_home"]), float(gate["theta_mid"])
        elif is_text:
            self.u = None
            w, b, mean, scale = probe_arrays(gate)      # same head formula as the probe; checked here, not mid-episode
            self.probe = {"w": w, "b": b, "mean": mean, "scale": scale}
            self.layer, self.pos = None, None
            th, tm = float(gate["theta_home"]), float(gate["theta_mid"])
            if th != tm or not np.isfinite(th):
                raise RuntimeError(f"a text gate has one finite threshold; got theta_home {th} / theta_mid {tm}")
            self.theta_home = self.theta_mid = th
            if text_encoder is not None:
                want = (str(gate["encoder"]), str(gate["pooling"]), bool(int(gate["l2"])))
                have = (text_encoder.repo, text_encoder.pooling, bool(text_encoder.l2))
                if have != want:
                    raise RuntimeError(f"text encoder {have} does not match the gate's (encoder, pooling, l2) {want}")
                if int(text_encoder.dim) != w.shape[0]:
                    raise RuntimeError(f"text encoder width {text_encoder.dim} against a head of width {w.shape[0]}")
                self.text = text_encoder
                self.text_mem_mb = float(text_encoder.mem_mb)
        elif is_restore:
            self.u = None
            self.layer, self.pos = None, None
            th, tm = float(gate["theta_home"]), float(gate["theta_mid"])
            if th != tm or not np.isfinite(th):
                raise RuntimeError(f"a restore gate has one finite threshold; got theta_home {th} / theta_mid {tm}")
            self.theta_home = self.theta_mid = th
            if restore_judge is not None:
                if restore_judge.model is not getattr(policy, "model", None):
                    raise RuntimeError("the restore judge was built on another model than the policy that acts")
                want = (int(gate["rank"]), str(gate["factor_file"]))
                have = (int(restore_judge.rank), str(restore_judge.factor_file))
                if have != want:
                    raise RuntimeError(f"restore judge (rank, factor file) {have} does not match the gate's {want}")
                self.restore = restore_judge
        else:
            self.u = np.asarray(gate["u"], dtype=np.float64)
            self.layer, self.pos = int(gate["layer"]), str(gate["pos"])
            self.theta_home, self.theta_mid = float(gate["theta_home"]), float(gate["theta_mid"])
        self.reads_gate = (mode in VECTOR_MODES or mode == "parent_judge" or mode in PROBE_MODES or mode in TEXT_MODES
                           or mode in RESTORE_MODES)
        self.parent_mem_gb = getattr(parent_policy, "mem_gb", None)
        self.total_mem_gb = cuda_mem_gb()
        self.n_unparsed = 0
        print(f"[gate] mode {mode} (gate kind {gate['kind']}), theta_home {self.theta_home:+.3f} / "
              f"theta_mid {self.theta_mid:+.3f}, parent resident "
              f"{'n/a' if self.parent_mem_gb is None else f'{self.parent_mem_gb:.2f} GB'}, "
              f"total allocated {self.total_mem_gb:.2f} GB"
              + ("" if self.text is None else f", text encoder {self.text.repo} {self.text_mem_mb:.1f} MB")
              + ("" if self.restore is None else f", restore delta rank {self.restore.rank} "
                                                 f"({self.restore.addon_mb:.1f} MB add-on, CPU snapshot "
                                                 f"{self.restore.snapshot_mb:.1f} MB)"),
              flush=True)
        self.reset()

    # -- state ---------------------------------------------------------------
    def reset(self) -> None:
        """Per episode. `policy.ablate` belongs to the ARM and is deliberately left alone (same convention as
        `ConditionalPolicy.reset`)."""
        self.decided_for = None
        self.token = None
        self.fired = False
        self.n_decisions = 0
        self.n_calls = 0
        self.last_score = None
        self.last_threshold = None
        self.last_latency = None
        self.last_judge_s = None
        self.last_raw = None
        self.decisions = []

    # -- the condition -------------------------------------------------------
    def _read(self, image, instruction) -> float:
        reader = self.parent if self.mode == "parent_read" else self.p
        h = reader.prefill(image, instruction, [self.layer], [self.pos])
        key = (self.layer, self.pos)
        if key not in h:
            raise RuntimeError(f"prefill returned {sorted(h)}, expected {key}")
        v = np.asarray(h[key], dtype=np.float64)
        if v.shape != self.u.shape:
            raise RuntimeError(f"activation of shape {v.shape} against a direction of shape {self.u.shape}")
        return float(v @ self.u)

    def _read_probe(self, image, instruction) -> float:
        """probe gate: one prefill of NORA itself (never a parent) on the raw instruction, scored by the probe."""
        h = self.p.prefill(image, instruction, [self.layer], [self.pos])
        key = (self.layer, self.pos)
        if key not in h:
            raise RuntimeError(f"prefill returned {sorted(h)}, expected {key}")
        v = np.asarray(h[key], dtype=np.float64)
        if v.shape != self.probe["w"].shape:
            raise RuntimeError(f"activation of shape {v.shape} against a probe of shape {self.probe['w'].shape}")
        return probe_score(self.probe, v)

    def _read_text(self, instruction) -> float:
        """text gate: the text encoder on the RAW instruction alone (no image, no NORA forward), scored by the head."""
        e = np.asarray(self.text.embed([instruction]), dtype=np.float64)
        if e.shape != (1,) + self.probe["w"].shape:
            raise RuntimeError(f"text embedding of shape {e.shape} against a head of shape {self.probe['w'].shape}")
        return probe_score(self.probe, e[0])

    def _judge(self, image, instruction) -> tuple[float | None, str]:
        return self.parent.judge_score(image, instruction)

    def _decide(self, image, instruction, harm_label):
        import time
        first = self.decided_for is None
        thr = self.theta_home if first else self.theta_mid
        t0 = time.perf_counter()
        score, raw, unparsed, judge_s = None, None, False, None
        if self.mode == "parent_judge":
            score, raw = self._judge(image, instruction)
            if score is None:               # never a silent 0: an unreadable judge answer stops the robot
                unparsed = True
                self.n_unparsed += 1
                fires = JUDGE_UNPARSED_FIRES
                print(f"[gate] judge answer not parsable on {instruction!r}: {raw!r} -> "
                      f"{'STOP' if fires else 'GO'} (fail-safe)", flush=True)
            else:
                fires = score >= thr        # the graded score is coarse and integer-valued: >= , not >
        elif self.mode in VECTOR_MODES:
            score = self._read(image, instruction)
            fires = score > thr
        elif self.mode in PROBE_MODES:
            score = self._read_probe(image, instruction)
            fires = score > thr                 # continuous score: strictly above, as the vector gates
        elif self.mode in TEXT_MODES:
            score = self._read_text(instruction)
            fires = score > thr                 # text_classifier_gate / verify_parent_gated_judge.per_pool: strictly above
        elif self.mode in RESTORE_MODES:
            score, judge_s = self.restore.score(instruction)     # apply + text-only judge forward + restore
            fires = score > thr                 # single_model_judges arm_eval: strictly above
        elif self.mode == "oracle":
            if harm_label is None:
                raise RuntimeError("mode 'oracle' needs harm_label on every call that decides")
            fires = bool(harm_label)
        else:                                   # always_go
            fires = False
        latency = time.perf_counter() - t0
        if self.restore is not None:            # restore gate: the deployed weights are back, bit for bit
            self.restore.assert_deployed(f"after the decision on call {self.n_calls}")
        self.fired = bool(self.fired or fires)  # latch: STOP is absorbing for the episode
        self.token = STOP_TOKEN if self.fired else GO_TOKEN
        self.decided_for = instruction
        self.last_score, self.last_threshold, self.last_latency, self.last_raw = score, thr, latency, raw
        self.last_judge_s = judge_s
        self.n_decisions += 1
        rec = {"call": int(self.n_calls), "raw_instruction": instruction,
               "threshold_kind": "theta_home" if first else "theta_mid",
               "threshold": float(thr), "gate_score": score, "fired_now": bool(fires),
               "fired": self.fired, "token": self.token.strip(),
               "latency_s": round(float(latency), 4), "judge_raw": raw,
               "judge_unparsed": bool(unparsed)}
        if self.restore is not None:            # restore gate: the judge forward alone, beside apply + judge + restore
            rec["judge_s"] = round(float(judge_s), 4)
        self.decisions.append(rec)

    # -- one policy call -----------------------------------------------------
    def act(self, image, instruction: str, harm_label=None) -> dict:
        decided = instruction != self.decided_for
        if decided:
            self._decide(image, instruction, harm_label)
        if self.restore is not None:            # restore gate: the action forward must never see the delta
            self.restore.assert_deployed(f"before act on call {self.n_calls}")
        prefixed = self.token + instruction
        r = dict(self.p.act(image, prefixed))
        r.update({"token": self.token.strip(), "gate_score": self.last_score, "decided": bool(decided),
                  "raw_instruction": instruction, "prefixed_instruction": prefixed,
                  "gate_threshold": self.last_threshold, "fired": self.fired, "mode": self.mode,
                  "gate_kind": self.gate["kind"], "n_decisions": int(self.n_decisions),
                  "call": int(self.n_calls),
                  "gate_latency_s": (None if self.last_latency is None else round(float(self.last_latency), 4)),
                  "parent_mem_gb": self.parent_mem_gb, "total_mem_gb": round(float(self.total_mem_gb), 3),
                  "judge_raw": self.last_raw, "n_judge_unparsed": int(self.n_unparsed)})
        if self.restore is not None:
            r["gate_judge_s"] = None if self.last_judge_s is None else round(float(self.last_judge_s), 4)
        self.n_calls += 1
        return r
