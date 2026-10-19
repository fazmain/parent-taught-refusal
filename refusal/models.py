"""Model loading and residual-stream hooks for the parent VLM, NORA, and a tiny text model for local smoke tests.

Layer resolution is generic (searches for the decoder-layer ModuleList) because the attribute path differs
between transformers 4.50 (`model.model.layers`) and later versions (`model.model.language_model.layers`).
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
import torch.nn as nn
import transformers

MODELS = {
    "parent": {"repo": "Qwen/Qwen2.5-VL-3B-Instruct", "kind": "vlm"},
    "nora": {"repo": "declare-lab/nora", "kind": "vlm"},
    "nora_lg": {"repo": "declare-lab/nora-long-finetuned-libero-goal", "kind": "vlm"},
    "smoke": {"repo": "Qwen/Qwen2.5-0.5B-Instruct", "kind": "text"},
}
IM_END = 151645  # <|im_end|> in the Qwen2.5 vocab (same id in NORA's extended vocab)


@dataclass
class Loaded:
    name: str
    kind: str
    model: nn.Module
    processor: object  # AutoProcessor (vlm) or AutoTokenizer (text)
    tokenizer: object
    layers: list
    device: str

    @property
    def n_layers(self) -> int:
        return len(self.layers)


def _dtype_kwarg(dtype):
    major = int(transformers.__version__.split(".")[0])
    return {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}


def find_decoder_layers(model: nn.Module) -> list:
    cfg = model.config
    n = getattr(cfg, "num_hidden_layers", None)
    if n is None and getattr(cfg, "text_config", None) is not None:
        n = cfg.text_config.num_hidden_layers
    hits = [(name, m) for name, m in model.named_modules()
            if isinstance(m, nn.ModuleList) and len(m) == n and "DecoderLayer" in type(m[0]).__name__]
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one decoder ModuleList of length {n}, found {[h[0] for h in hits]}")
    print(f"[models] decoder layers at `{hits[0][0]}` ({n} layers)")
    return list(hits[0][1])


def load_model(name: str, device: str | None = None, dtype=torch.bfloat16) -> Loaded:
    spec = MODELS[name]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if spec["kind"] == "vlm":
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(spec["repo"], attn_implementation="sdpa", **_dtype_kwarg(dtype))
        processor = AutoProcessor.from_pretrained(spec["repo"])
        tokenizer = processor.tokenizer
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(spec["repo"], **_dtype_kwarg(dtype))
        tokenizer = AutoTokenizer.from_pretrained(spec["repo"])
        processor = tokenizer
    model.to(device).eval()
    model.generation_config.do_sample = False  # NORA ships do_sample=True, temperature=0.1
    return Loaded(name, spec["kind"], model, processor, tokenizer, find_decoder_layers(model), device)


def build_inputs(L: Loaded, text: str, image=None) -> dict:
    """Chat-formatted inputs with the generation prompt appended (assistant turn open), batch size 1."""
    if L.kind == "text" or image is None:
        content = text if L.kind == "text" else [{"type": "text", "text": text}]
        msgs = [{"role": "user", "content": content}]
        prompt = L.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = L.processor(text=[prompt], return_tensors="pt") if L.kind == "vlm" else L.tokenizer(prompt, return_tensors="pt")
    else:
        from qwen_vl_utils import process_vision_info
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image, "resized_height": 224, "resized_width": 224},
            {"type": "text", "text": text}]}]
        prompt = L.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = L.processor(text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    return {k: v.to(L.device) for k, v in inputs.items()}


def positions(input_ids: torch.Tensor) -> dict:
    """t_inst = last instruction token (just before the user turn's <|im_end|>); t_post_inst = last prompt token."""
    ids = input_ids[0].tolist()
    ends = [i for i, t in enumerate(ids) if t == IM_END]
    if not ends:
        raise RuntimeError("no <|im_end|> in prompt; template mismatch")
    return {"t_inst": ends[-1] - 1, "t_post_inst": len(ids) - 1}


class ActivationCache:
    """Forward hooks on every decoder block; stores the residual stream at the requested positions (float32, CPU)."""

    def __init__(self, L: Loaded, layers: list[int] | None = None):
        self.L = L
        self.layer_ids = layers if layers is not None else list(range(L.n_layers))
        self.pos: list[int] = []
        self.acts: dict[int, torch.Tensor] = {}
        self._handles = []

    def _hook(self, li):
        def fn(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            self.acts[li] = h[0, self.pos, :].detach().float().cpu()
        return fn

    @contextlib.contextmanager
    def capture(self, pos: list[int]):
        self.pos, self.acts = pos, {}
        self._handles = [self.L.layers[li].register_forward_hook(self._hook(li)) for li in self.layer_ids]
        try:
            yield self
        finally:
            for h in self._handles:
                h.remove()
            self._handles = []


@torch.inference_mode()
def residuals(L: Loaded, cache: ActivationCache, text: str, image=None) -> tuple[dict, dict]:
    """One forward pass. Returns ({layer: (2, d) tensor for [t_inst, t_post_inst]}, positions)."""
    inputs = build_inputs(L, text, image)
    pos = positions(inputs["input_ids"])
    with cache.capture([pos["t_inst"], pos["t_post_inst"]]):
        L.model(**inputs)
    return cache.acts, pos


@torch.inference_mode()
def generate(L: Loaded, text: str, image=None, max_new_tokens: int = 48) -> str:
    inputs = build_inputs(L, text, image)
    out = L.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return L.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
