"""Activation patching (causal tracing, Meng et al. 2022) for the VLA policy.

Additive steering failed as a positive control (single-layer and multi-layer): no strength of a (B-A) task
direction makes NORA do task B. Patching is the stronger intervention: instead of adding a direction we
COPY the exact residual-stream vectors at the instruction-token positions from a source forward pass
(same image, instruction B) into the target forward pass (same image, instruction A), at chosen layers,
on every policy call of a closed-loop rollout. If the robot then does B while hearing A, we know which
layers read the words.

  instruction_span(input_ids, tokenizer)  positions of the instruction words + the 5 post-instruction
                                          template tokens ("<|im_end|>\\n<|im_start|>assistant\\n")
  span_for_mode(...)                      that span, or the words only, or the template tokens only
  capture_residuals(...)                  one forward pass, full block output at each layer (the source
                                          may be a different policy: cross-model patching)
  patch_residuals(...)                    context manager, overwrites those positions on the prefill

Only the prefill forward (sequence length > 1) is patched; decode steps read the patched values back out
of the KV cache, so touching them again would double-apply the edit.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from refusal.models import IM_END  # noqa: E402

VISION_END = "<|vision_end|>"      # id 151653 in the Qwen2.5 vocab; looked up on the tokenizer, never hardcoded
USER_HEADER = "<|im_start|>user\n"  # fallback anchor for text-only prompts (no vision block)
POST_INST = "<|im_end|>\n<|im_start|>assistant\n"
N_POST_INST = 5                    # tokens in POST_INST; verified against the tokenizer on every call


def _tokenizer(policy):
    """NORA's `policy.processor` is an AutoProcessor (tokenizer inside); the smoke policy passes a tokenizer."""
    return getattr(policy.processor, "tokenizer", policy.processor)


def _ids_of(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _last_subsequence_end(ids: list[int], sub: list[int]) -> int | None:
    """Index just past the last occurrence of `sub` in `ids`, or None."""
    for i in range(len(ids) - len(sub), -1, -1):
        if ids[i:i + len(sub)] == sub:
            return i + len(sub)
    return None


def instruction_span(input_ids, tokenizer) -> tuple[int, int]:
    """(start, end) half-open range of the instruction tokens plus the post-instruction template tokens.

    start = one past the last <|vision_end|> (the image block ends there), end = len(prompt). For a
    text-only prompt there is no vision block; we then fall back to one past the last "<|im_start|>user\\n",
    which is the same boundary (first token of the user turn's text). The prompt is required to end with
    the 5 POST_INST tokens; anything else is a template mismatch and raises.
    """
    ids = input_ids[0].tolist() if hasattr(input_ids, "dim") and input_ids.dim() == 2 else list(input_ids)
    if IM_END not in ids:
        raise RuntimeError("no <|im_end|> in prompt; template mismatch")
    ve = tokenizer.convert_tokens_to_ids(VISION_END)
    if ve is None or ve == getattr(tokenizer, "unk_token_id", None):
        raise RuntimeError(f"tokenizer has no {VISION_END} token")
    if ve in ids:
        start = len(ids) - 1 - ids[::-1].index(ve) + 1
    else:
        start = _last_subsequence_end(ids, _ids_of(tokenizer, USER_HEADER))
        if start is None:
            raise RuntimeError(f"no {VISION_END} and no {USER_HEADER!r} in prompt; template mismatch")
    tail = _ids_of(tokenizer, POST_INST)
    if len(tail) != N_POST_INST or ids[-N_POST_INST:] != tail:
        raise RuntimeError(f"prompt does not end with {POST_INST!r} ({N_POST_INST} tokens); "
                           f"got {ids[-N_POST_INST:]}, expected {tail}")
    if start >= len(ids):
        raise RuntimeError(f"empty instruction span (start={start}, len={len(ids)})")
    return start, len(ids)


def span_for_mode(input_ids, tokenizer, mode: str) -> tuple[int, int]:
    """"instruction" = the whole span; "words" = the instruction words only (span minus the 5 template
    tokens); "postinst" = only the 5 post-instruction template positions.

    "words" is the layer-informative single-layer intervention: the template positions keep their own
    A-derived states at layers <= l, so the target instruction can still reach the action tokens.
    """
    start, end = instruction_span(input_ids, tokenizer)
    if mode == "instruction":
        return start, end
    if mode == "words":
        if start >= end - N_POST_INST:
            raise RuntimeError(f"no instruction words in span ({start}, {end})")
        return start, end - N_POST_INST
    if mode == "postinst":
        return end - N_POST_INST, end
    raise ValueError(f"unknown span_mode {mode!r}")


def _hidden_size(policy) -> int:
    cfg = policy.model.config
    h = getattr(cfg, "hidden_size", None)
    if h is None and getattr(cfg, "text_config", None) is not None:
        h = cfg.text_config.hidden_size
    if h is None:
        raise RuntimeError(f"cannot read hidden_size from {type(cfg).__name__}")
    return int(h)


def check_source_policy(policy, source_policy) -> None:
    """Cross-model patching (parent VLM -> VLA): the residual stream must have the same width, and the
    layer indices must mean the same thing, so the block count has to match too. Vocabularies may differ."""
    h_t, h_s = _hidden_size(policy), _hidden_size(source_policy)
    if h_t != h_s:
        raise RuntimeError(f"cross-model patch needs equal hidden sizes: target {h_t}, source {h_s}")
    if len(policy.layers) != len(source_policy.layers):
        raise RuntimeError(f"cross-model patch needs equal layer counts: target {len(policy.layers)}, "
                           f"source {len(source_policy.layers)}")


@torch.inference_mode()
def capture_residuals(policy, image, instruction: str, layers: list[int]):
    """One forward pass (no generation). Returns ({layer: (seq, hidden) tensor on device}, input_ids).

    The FULL block output is stored at every position so the caller can slice whatever span it wants;
    dtype and device are the model's, so the tensors go straight back into a forward pass.
    """
    inputs = policy._inputs(image, instruction)
    acts: dict[int, torch.Tensor] = {}

    def make_hook(li):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            acts[li] = h[0].detach().clone()
        return hook

    handles = [policy.layers[li].register_forward_hook(make_hook(li)) for li in layers]
    try:
        policy.model(**inputs)
    finally:
        for hd in handles:
            hd.remove()
    missing = [li for li in layers if li not in acts]
    if missing:
        raise RuntimeError(f"no activation captured at layers {missing}; hooks never fired")
    return acts, inputs["input_ids"]


@contextlib.contextmanager
def patch_residuals(policy, layers: list[int], source_acts: dict[int, torch.Tensor],
                    source_span: tuple[int, int], target_span: tuple[int, int]):
    """Overwrite the target span of each layer's block output with the source span's vectors.

    Spans of different length (the two instructions tokenize to different lengths) are aligned from the
    END, so the 5 template tokens and the last instruction tokens line up; min(len) positions are patched.
    Only prefill forwards (sequence length > 1) are touched: the decode steps read the patched keys and
    values out of the KV cache. Hooks are removed on exit; a hook that never fired is an error.
    """
    s0, s1 = source_span
    t0, t1 = target_span
    n = min(s1 - s0, t1 - t0)
    if n <= 0:
        raise RuntimeError(f"empty patch span: source {source_span}, target {target_span}")
    s_start, t_start = s1 - n, t1 - n
    fired = {li: False for li in layers}

    def make_hook(li):
        src = source_acts[li]
        if src.dim() != 2 or src.shape[0] < s1:
            raise RuntimeError(f"source activation at layer {li} has shape {tuple(src.shape)}, "
                               f"need (seq >= {s1}, hidden)")

        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] <= 1:  # decode step: the KV cache already carries the patched values
                return out
            if h.shape[1] < t1:
                raise RuntimeError(f"prefill length {h.shape[1]} shorter than target span end {t1} "
                                   f"at layer {li}; prompt/template mismatch")
            fired[li] = True
            h = h.clone()
            h[0, t_start:t_start + n, :] = src[s_start:s_start + n, :].to(device=h.device, dtype=h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook

    handles = [policy.layers[li].register_forward_hook(make_hook(li)) for li in layers]
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()
    never = [li for li in layers if not fired[li]]
    if never:
        raise RuntimeError(f"patch hooks never fired at layers {never}; no prefill forward happened")
