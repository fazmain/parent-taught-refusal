"""Conditional activation steering for the NORA VLA policy (CAST, Lee et al. 2024).
Zero training anywhere in this file.

Three pieces:

  fit_triggers(...)      the CONDITION side. Three triggers, each = (a vector or a probe, a layer, a prompt
                         position, a threshold), fitted only on the D_robot activations:
                           parent_dir  parent diff-in-means harm-benign at L27 t_inst, read on NORA L27 t_inst
                           probe       the logistic probe on NORA L21 t_post_inst (monitor_halt_demo)
                           random_dir  a fixed random unit vector, same layer/position as parent_dir
                         The two vector triggers get a threshold at the (1-fpr) quantile of the 46 benign
                         D_robot rows, i.e. a 5% false-positive rate by construction.

  ConditionalPolicy      wraps a NoraPolicy. Forward hooks read the prefill residual stream at the trigger
                         positions on every policy call; ALL triggers are scored and logged whatever the arm
                         is, but only the arm's trigger decides, and only within the first `decide_calls`
                         calls of an episode (the scores drift upward once the arm moves). Once fired
                         the RESPONSE stays on for the rest of the episode.

  build_hold_chunk(...)  the "stop" behaviour target: the FAST token ids of a zero-motion action chunk,
                         gripper open (the `forced_hold` response of conditional steering and the free install) or
                         closed (the parent-gated reference's halt label on a mid-carry frame).

No torch import at module level: the hooks only call methods on the tensors the model hands them, so trigger
fitting runs on a laptop (`conditional_steering.py --fit-only`) while the policy parts need a GPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from refusal.paths import add_experiment_dirs  # noqa: E402
add_experiment_dirs()
from refusal.directions import cosine, diff_in_means, unit  # noqa: E402

BENIGN_CONDITIONS = ("id_benign", "ood_benign", "lexical")   # benign_all of the D_robot prompt file
CONDITIONS = ("harm",) + BENIGN_CONDITIONS
POSITIONS = ("t_inst", "t_post_inst")


# ----------------------------------------------------------------------------- triggers (condition side)

def load_drobot(nora_acts_path, parent_acts_path, prompts_path):
    """The D_robot activations for NORA and the parent, with the prompt rows they were computed from.
    The id arrays must match the prompt file row for row (same check as monitor_halt_demo.train_probe)."""
    prompts = [json.loads(l) for l in Path(prompts_path).read_text().splitlines() if l.strip()]
    ids = [r["id"] for r in prompts]
    loaded = []
    for p in (nora_acts_path, parent_acts_path):
        z = np.load(p)
        if [str(i) for i in z["ids"]] != ids:
            raise SystemExit(f"activation ids in {p} do not match {prompts_path} row order")
        loaded.append(z)
    cond = np.array([r["condition"] for r in prompts])
    unexpected = set(cond.tolist()) - set(CONDITIONS)
    if unexpected:
        raise SystemExit(f"unexpected conditions {sorted(unexpected)} in {prompts_path}")
    return loaded[0], loaded[1], cond


def fit_triggers(nora_acts_path, parent_acts_path, prompts_path, dir_layer=27, probe_layer=21,
                 fpr=0.05, seed=0, rand_seed=0):
    """Fit the three triggers. Returns (triggers, fit) where `fit` holds only JSON-serialisable scalars.

    Every score is read on NORA activations (that is where the trigger will live at run time); only the
    parent_dir VECTOR comes from the parent model.
    """
    from sklearn.metrics import roc_auc_score
    from monitor_halt_demo import train_probe  # sklearn only, no torch

    nora, parent, cond = load_drobot(nora_acts_path, parent_acts_path, prompts_path)
    harm = cond == "harm"
    benign = ~harm
    P = parent[f"L{dir_layer}_t_inst"].astype(np.float64)
    N = nora[f"L{dir_layer}_t_inst"].astype(np.float64)
    X = nora[f"L{probe_layer}_t_post_inst"].astype(np.float64)
    if P.shape[1] != N.shape[1]:
        raise SystemExit(f"parent hidden size {P.shape[1]} != NORA hidden size {N.shape[1]}")

    parent_vec = diff_in_means(P[harm], P[benign])
    rand_vec = unit(np.random.default_rng(rand_seed).standard_normal(N.shape[1]))
    probe, probe_stats = train_probe(nora_acts_path, prompts_path, probe_layer, fpr, seed)
    probe_w = np.asarray(probe["w"], dtype=np.float64)
    probe_mean = np.asarray(probe["mean"], dtype=np.float64)
    probe_scale = np.asarray(probe["scale"], dtype=np.float64)

    triggers = {
        "parent_dir": {"kind": "vec", "vec": parent_vec, "u": unit(parent_vec).astype(np.float64),
                       "layer": int(dir_layer), "pos": "t_inst"},
        "probe": {"kind": "probe", "w": probe_w, "b": float(probe["b"]), "mean": probe_mean,
                  "scale": probe_scale, "layer": int(probe_layer), "pos": "t_post_inst",
                  "threshold": float(probe["threshold"])},
        "random_dir": {"kind": "vec", "vec": rand_vec, "u": unit(rand_vec).astype(np.float64),
                       "layer": int(dir_layer), "pos": "t_inst"},
    }
    scores = {"parent_dir": N @ triggers["parent_dir"]["u"],
              "random_dir": N @ triggers["random_dir"]["u"],
              "probe": ((X - probe_mean) / probe_scale) @ probe_w + float(probe["b"])}
    for name in ("parent_dir", "random_dir"):  # 95th percentile of the benign D_robot rows = fpr by construction
        triggers[name]["threshold"] = float(np.quantile(scores[name][benign], 1.0 - fpr))

    fit = {"n": int(len(cond)), "n_harm": int(harm.sum()), "n_benign": int(benign.sum()),
           "fpr": float(fpr), "dir_layer": int(dir_layer), "probe_layer": int(probe_layer),
           "seed": int(seed), "rand_seed": int(rand_seed),
           "probe_train_auroc": float(probe_stats["train_auroc"]),
           "cos_parent_dir_probe": cosine(parent_vec, probe_w / probe_scale),
           "nora_acts": str(nora_acts_path), "parent_acts": str(parent_acts_path),
           "prompts": str(prompts_path), "triggers": {}}
    for name, t in triggers.items():
        s, thr = scores[name], t["threshold"]
        fit["triggers"][name] = {"kind": t["kind"], "layer": t["layer"], "pos": t["pos"],
                                 "threshold": float(thr),
                                 "drobot_auroc": float(roc_auc_score(harm.astype(int), s)),
                                 "drobot_catch_rate": float((s[harm] >= thr).mean()),
                                 "drobot_false_alarm_rate": float((s[benign] >= thr).mean()),
                                 "benign_score_mean": float(s[benign].mean()),
                                 "harm_score_mean": float(s[harm].mean())}
    return triggers, fit


def trigger_table(fit) -> str:
    """The trigger table, printed at fit time and again in the run log."""
    lines = [f"[triggers] {fit['n']} D_robot rows ({fit['n_harm']} harm / {fit['n_benign']} benign), "
             f"target FPR {fit['fpr']:.2f}, dir layer {fit['dir_layer']}, probe layer {fit['probe_layer']}",
             f"{'trigger':<12s}{'layer':>6s}  {'pos':<13s}{'auroc':>8s}{'threshold':>11s}"
             f"{'catch':>8s}{'false_alarm':>13s}{'mean_benign':>13s}{'mean_harm':>11s}"]
    for name, t in fit["triggers"].items():
        lines.append(f"{name:<12s}{t['layer']:6d}  {t['pos']:<13s}{t['drobot_auroc']:8.3f}"
                     f"{t['threshold']:+11.3f}{t['drobot_catch_rate']:8.2f}"
                     f"{t['drobot_false_alarm_rate']:13.2f}{t['benign_score_mean']:+13.3f}"
                     f"{t['harm_score_mean']:+11.3f}")
    lines.append(f"cos(parent_dir, probe direction w/scale) = {fit['cos_parent_dir_probe']:+.4f}")
    return "\n".join(lines)


def save_triggers(path, triggers) -> None:
    """Archive the fitted vectors / probe parameters next to the run."""
    flat = {}
    for name, t in triggers.items():
        for k, v in t.items():
            if isinstance(v, np.ndarray):
                flat[f"{name}__{k}"] = v.astype(np.float32)
            elif isinstance(v, (int, float)):
                flat[f"{name}__{k}"] = np.asarray(v, dtype=np.float64)
    np.savez(path, **flat)


# ----------------------------------------------------------------------------- prompt positions

def instruction_offsets(policy, instruction: str) -> dict:
    """Negative indexes (counted from the END of the prompt) of t_inst and t_post_inst for NORA's chat template.

    `refusal.models.positions` gives absolute indexes for one tokenised prompt, but the prefill length changes with
    the instruction and with the image, while a forward hook only sees the activation tensor. Everything after
    the instruction is the fixed template tail "<|im_end|>\\n<|im_start|>assistant\\n", so the offset from the
    end is instruction-independent; it is recomputed and compared for every new instruction, never assumed.
    The image block sits before the instruction, so a text-only tokenisation gives the same offsets and costs
    no image preprocessing.
    """
    from refusal.models import IM_END, positions  # late: refusal.models imports torch / transformers
    tok = getattr(policy.processor, "tokenizer", policy.processor)
    msgs = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
    text = policy.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = list(tok(text, add_special_tokens=False)["input_ids"])
    pos = positions(np.asarray([ids]))
    if pos["t_inst"] + 1 >= len(ids) or ids[pos["t_inst"] + 1] != IM_END:
        got = ids[pos["t_inst"] + 1] if pos["t_inst"] + 1 < len(ids) else None
        raise RuntimeError(f"token after t_inst is {got}, expected <|im_end|> ({IM_END}); template mismatch")
    return {"t_inst": int(pos["t_inst"] - len(ids)), "t_post_inst": -1}


# ----------------------------------------------------------------------------- ConditionalPolicy

class ConditionalPolicy:
    """NoraPolicy + trigger hooks + a response that switches on when the trigger fires.

    `triggers`  {name: {"kind": "vec"|"probe", "layer": int, "pos": "t_inst"|"t_post_inst",
                        "threshold": float, + "u" (vec) or "w"/"b"/"mean"/"scale" (probe)}}
    `trigger`   the arm's trigger name, or None (log only / unconditional response).
    `response`  None, or one of
                  {"kind": "add",        "layer": int, "vec": ndarray, "alpha": float}
                  {"kind": "add_multi",  "specs": [{"layer","vec","alpha"}, ...]}
                  {"kind": "forced_hold","gen_ids": [int]}
                With `trigger=None` and a response, the response is active from call 0 (stage 2 / unconditional
                arms). Ablation is NOT handled here: the experiment sets `policy.ablate` for the whole arm.

    On the call that fires, the action has already been generated from the unsteered forward pass, so the call
    is re-run once with the response active and that second result is the one that reaches the environment.
    The scores reported for the call are always the ones the decision was made on (the clean pass).
    """

    def __init__(self, policy, triggers, trigger=None, response=None, decide_calls=1):
        if trigger is not None and trigger not in triggers:
            raise RuntimeError(f"unknown trigger {trigger!r}; have {sorted(triggers)}")
        if response is not None and response.get("kind") not in ("add", "add_multi", "forced_hold"):
            raise RuntimeError(f"unknown response kind {response.get('kind')!r}")
        for name, t in triggers.items():
            if t["pos"] not in POSITIONS:
                raise RuntimeError(f"trigger {name}: unknown position {t['pos']!r}")
            if "threshold" not in t:
                raise RuntimeError(f"trigger {name}: no threshold")
        self.p, self.triggers, self.trigger, self.response = policy, triggers, trigger, response
        self.decide_calls = int(decide_calls)
        self.uncond = trigger is None and response is not None
        self.needed = sorted({(int(t["layer"]), t["pos"]) for t in triggers.values()})
        self.off = None                 # resolved on the first instruction
        self.ref_instruction = None
        self._offsets = {}
        self._h = {}
        self.handles = [policy.layers[li].register_forward_hook(self._hook(li))
                        for li in sorted({li for li, _ in self.needed})]
        self.reset()

    # -- hooks ---------------------------------------------------------------
    def _hook(self, li):
        """Read-only hook, registered once at construction. The steering / ablation hooks are registered per
        call inside NoraPolicy.act, i.e. AFTER this one, and forward hooks run in registration order: at the
        layer a response edits, the score logged for that call is therefore the value BEFORE the edit (the
        natural read-out), while layers above it see the edited stream. Scores are logged, not fed back, so
        this only matters when reading the drift columns."""
        def fn(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] <= 1:          # decode step: one token, no prompt positions (see MonitoredPolicy)
                return
            if self.off is None:
                raise RuntimeError("trigger hook fired before the prompt positions were resolved")
            for lj, pos in self.needed:
                if lj != li:
                    continue
                idx = self.off[pos]
                if h.shape[1] < -idx:
                    raise RuntimeError(f"prefill length {h.shape[1]} shorter than offset {idx} at layer {li}")
                self._h[(li, pos)] = h[0, idx, :].detach().float().cpu().numpy()
        return fn

    def _resolve(self, instruction):
        if instruction in self._offsets:
            return
        off = instruction_offsets(self.p, instruction)
        if self.off is None:
            self.off, self.ref_instruction = off, instruction
            print(f"[cond] prompt positions resolved on {instruction!r}: t_inst at offset {off['t_inst']}, "
                  f"t_post_inst at offset {off['t_post_inst']}", flush=True)
        elif off != self.off:
            raise RuntimeError(f"instruction {instruction!r} gives offsets {off} but {self.ref_instruction!r} "
                               f"gave {self.off}; chat template mismatch")
        self._offsets[instruction] = off

    def _require_hooks(self):
        missing = [k for k in self.needed if k not in self._h]
        if missing:
            raise RuntimeError(f"trigger hooks never fired for {missing} on a prefill forward")

    # -- scoring -------------------------------------------------------------
    def _score(self, name, h):
        t = self.triggers[name]
        x = np.asarray(h, dtype=np.float64)
        if t["kind"] == "probe":
            return float(t["w"] @ ((x - t["mean"]) / t["scale"]) + t["b"])
        return float(x @ t["u"])

    # -- response ------------------------------------------------------------
    def _set_response(self, active):
        self.p.steer = None
        self.p.forced_gen_ids = None
        if not active or self.response is None:
            return
        k = self.response["kind"]
        if k == "add":
            self.p.steer = {"layer": int(self.response["layer"]), "vec": self.response["vec"],
                            "alpha": float(self.response["alpha"])}
        elif k == "add_multi":
            self.p.steer = list(self.response["specs"])
        else:
            self.p.forced_gen_ids = list(self.response["gen_ids"])

    def reset(self):
        """Per episode. Clears the fired state and the steer/forced hooks; `policy.ablate` belongs to the arm
        and is only cleared in close(), because the experiment sets it once per arm."""
        self.fired, self.fired_step, self.fired_score = False, None, None
        self.n_calls = 0
        self.last_scores = None
        self.last_h = {}
        self._h = {}
        self._set_response(self.uncond)

    def act(self, image, instruction):
        self._resolve(instruction)
        i = self.n_calls
        self.n_calls += 1
        active = self.response is not None and (self.uncond or self.fired)
        self._set_response(active)
        self._h = {}
        r = self.p.act(image, instruction)
        self._require_hooks()
        self.last_h = dict(self._h)
        scores = {name: self._score(name, self._h[(t["layer"], t["pos"])]) for name, t in self.triggers.items()}
        rerun = False
        if self.trigger is not None and not self.fired and i < self.decide_calls:
            if scores[self.trigger] >= self.triggers[self.trigger]["threshold"]:
                self.fired, self.fired_step, self.fired_score = True, i, scores[self.trigger]
                if self.response is not None:  # re-run this call with the response on, so nothing moves first
                    self._set_response(True)
                    active, rerun = True, True
                    self._h = {}
                    r = self.p.act(image, instruction)
                    self._require_hooks()
        self.last_scores = scores
        r.update({"scores": scores, "fired": self.fired, "response_active": active, "rerun": rerun})
        return r

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.p.steer = None
        self.p.forced_gen_ids = None
        self.p.ablate = None


# ----------------------------------------------------------------------------- hold chunk (behaviour target)

def _encoded_ids(enc) -> list[int]:
    """FAST processor __call__ output -> a flat list of action-token indexes.

    ASSUMPTION the GPU smoke run must confirm: `fast_tokenizer(chunk[None])` returns one token sequence per batch
    element, either as a list of lists of ints, as an integer array, or inside a BatchFeature under
    "input_ids". The remote code (physical-intelligence/fast, processing_action_tokenizer.py) may not be cached
    locally, so the shape is checked here at run time instead of being read off the source. If this assertion
    fires, print `type(enc)` / `enc` and adapt this function only.
    """
    if hasattr(enc, "keys") and "input_ids" in enc:
        enc = enc["input_ids"]
    seq = enc
    if len(seq) == 1 and not isinstance(seq[0], (int, np.integer)):
        seq = seq[0]
    if isinstance(seq, (int, np.integer)):
        raise SystemExit(f"FAST tokenizer returned a scalar {seq!r}, expected a token sequence")
    arr = np.asarray(seq).ravel()
    if arr.size == 0:
        raise SystemExit(f"FAST tokenizer returned an empty token sequence: {enc!r}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise SystemExit(f"FAST tokenizer returned dtype {arr.dtype}, expected integers: {str(enc)[:200]}")
    return [int(i) for i in arr.tolist()]


GRIPPER_STATES = ("open", "closed")


def gripper_normalised(low6: float, high6: float, want: str) -> tuple[float, float, float]:
    """(normalised, raw, env) gripper value whose env command is open (-1) or closed (+1).

    Pure arithmetic over the suite's normalisation stats and `normalize_gripper`, so it is unit-testable on
    a CPU. The env convention is NORA's: `normalize_gripper` maps raw -> sign(2x-1), inverts it, then snaps
    >= 0 to +1, i.e. -1 = open, +1 = closed. Never hard-coded: both candidates are evaluated and exactly one
    must match, otherwise this is fatal.
    """
    from refusal.libero_env import normalize_gripper
    if want not in GRIPPER_STATES:
        raise SystemExit(f"gripper must be one of {GRIPPER_STATES}, got {want!r}")
    span = float(high6) - float(low6)
    if span <= 0:
        raise SystemExit(f"degenerate gripper normalisation stats: low {low6} high {high6}")
    hits = []
    for g in (1.0, -1.0):
        raw = 0.5 * (g + 1.0) * span + float(low6)
        env = float(normalize_gripper(np.concatenate([np.zeros(6), [raw]]))[-1])
        if (env < 0) == (want == "open"):
            hits.append((float(g), float(raw), env))
    if len(hits) != 1:
        raise SystemExit(f"cannot pick a gripper-{want} normalised value from low {low6} high {high6}: "
                         f"candidates {hits}")
    return hits[0]


def build_hold_chunk(policy, unnorm_key=None, motion_eps=0.05, gripper: str = "open") -> dict:
    """The zero-motion action chunk as FAST token ids, with a decode round trip.

    `gripper="open"` is the scripted hold chunk of conditional steering (unchanged default, so every existing
    caller and the `results/raw/pod6/c1/hold_chunk.json` equality assertion keep working). `gripper="closed"` is
    the variant of the parent-gated reference: a halt on a mid-carry frame must not drop the object, so the halt
    label keeps the gripper as the base policy commanded it at that frame.

    Raw env-space action [0,0,0,0,0,0,g] repeated time_horizon times. The normalised value that decodes to raw
    motion 0 is n_j = 2*(0 - low_j)/(high_j - low_j) - 1. Any failure is fatal: a silent fallback here would
    make the halt behaviour meaningless.
    """
    from refusal.models import IM_END
    from refusal.nora_policy import ACTION_MAX, ACTION_MIN
    from refusal.libero_env import normalize_gripper

    key = unnorm_key or policy.unnorm_key
    low, high = (np.asarray(v, dtype=np.float64) for v in policy.nora.libero_keys[key])
    span = high - low
    if np.any(span <= 0):
        raise SystemExit(f"degenerate normalisation stats for {key}: low={low.tolist()} high={high.tolist()}")
    horizon = int(policy.nora.fast_tokenizer.time_horizon)
    n_motion = 2.0 * (0.0 - low[:6]) / span[:6] - 1.0
    if np.abs(n_motion).max() > 1.0:
        raise SystemExit(f"zero motion is outside the normalised range for {key}: {n_motion.tolist()}")

    g_norm, g_raw, g_env = gripper_normalised(low[6], high[6], gripper)

    chunk = np.tile(np.concatenate([n_motion, [g_norm]]), (horizon, 1)).astype(np.float32)
    enc = policy.nora.fast_tokenizer(chunk[None])   # (1, horizon, 7) -> token ids; see _encoded_ids
    ids = _encoded_ids(enc)
    if min(ids) < 0 or max(ids) > ACTION_MAX - ACTION_MIN:
        raise SystemExit(f"FAST ids {min(ids)}..{max(ids)} outside the action-token range "
                         f"0..{ACTION_MAX - ACTION_MIN}")

    norm = np.asarray(policy.nora.fast_tokenizer.decode([ids]), dtype=np.float64).reshape(-1, 7)
    if norm.shape[0] != horizon:
        raise SystemExit(f"round trip decoded {norm.shape[0]} steps, expected {horizon}")
    action = 0.5 * (norm + 1.0) * span + low
    env_action = normalize_gripper(action)
    err = float(np.abs(env_action[:, :6]).max())
    if not err < motion_eps:
        raise SystemExit(f"hold chunk round trip max |motion| {err:.4f} >= {motion_eps}")
    want_open = gripper == "open"
    if not np.all((env_action[:, 6] < 0) == want_open):
        raise SystemExit(f"hold chunk round trip does not keep the gripper {gripper}: "
                         f"{env_action[:, 6].tolist()}")

    im_end = getattr(policy.processor, "tokenizer", policy.processor).convert_tokens_to_ids("<|im_end|>")
    if int(im_end) != IM_END:
        raise SystemExit(f"tokenizer maps <|im_end|> to {im_end}, expected {IM_END}")
    info = {"unnorm_key": key, "time_horizon": horizon, "gripper": gripper,
            "normalised_step": [float(x) for x in chunk[0].tolist()],
            "gripper_normalised": g_norm, "gripper_raw": g_raw, "gripper_env": g_env,
            "action_token_ids": ids, "n_tokens": len(ids),
            "gen_ids": [int(i) + ACTION_MIN for i in ids] + [int(IM_END)],
            "roundtrip_max_abs_motion": err,
            "roundtrip_gripper_env": [float(x) for x in env_action[:, 6].tolist()],
            "roundtrip_first_step": [float(x) for x in env_action[0].tolist()]}
    print(f"[hold:{gripper}] {key} horizon {horizon}: {len(ids)} FAST ids {ids} -> gen_ids {info['gen_ids']}\n"
          f"[hold:{gripper}] round trip max |motion| {err:.5f} m/rad, gripper env "
          f"{info['roundtrip_gripper_env']} (normalised {g_norm:+.1f}, raw {g_raw:.3f})", flush=True)
    return info
