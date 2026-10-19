"""NORA as a LIBERO policy with the raw generated token ids exposed (needed to count text tokens / invalid decodes,
and later to attach steering hooks). Uses the vendored `third_party.nora.nora_utils.Nora` for loading, prompt format,
FAST decoding and the LIBERO un-normalisation stats; the inference path below mirrors `Nora.inference` line for line."""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import numpy as np
import PIL.Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from refusal.models import find_decoder_layers  # noqa: E402
from refusal.patching import capture_residuals, check_source_policy, patch_residuals, span_for_mode  # noqa: E402
from refusal.steering import ablate_direction, add_direction, add_directions  # noqa: E402
from third_party.nora.nora_utils import Nora  # noqa: E402

ACTION_MIN, ACTION_MAX = Nora._ACTION_TOKEN_MIN, Nora._ACTION_TOKEN_MAX
MEAN_WORDS = "mean_words"   # mean-pooled probe prefill position: the mean residual over the instruction words


def _merge_adapter(model, adapter: str):
    """Fold a peft LoRA adapter into the base weights (used by every LoRA install).

    `merge_and_unload()` returns the plain Qwen2_5_VLForConditionalGeneration with the LoRA deltas written into
    the frozen weights, so every hook / steering / ablation / patching path below keeps working unchanged: no
    peft wrapper sits between `find_decoder_layers` and the decoder blocks, and the module identities the hooks
    attach to are the base model's own."""
    from peft import PeftModel  # late: peft is only needed when an adapter is loaded
    path = Path(adapter)
    if not (path / "adapter_config.json").exists():
        raise SystemExit(f"{path} is not a peft adapter directory: no adapter_config.json")
    wrapped = PeftModel.from_pretrained(model, str(path), is_trainable=False)
    n_merged = sum(1 for n, _ in wrapped.named_modules() if n.endswith("lora_A.default"))
    if n_merged == 0:
        raise SystemExit(f"adapter {path} loaded but contains no LoRA modules")
    merged = wrapped.merge_and_unload()
    merged.eval()
    print(f"[nora] merged LoRA adapter {path} ({n_merged} modules) into the base weights", flush=True)
    return merged


class NoraPolicy:
    def __init__(self, checkpoint: str, unnorm_key: str = "libero_goal", time_horizon: int = 5, device: str | None = None,
                 adapter: str | None = None):
        self.nora = Nora(model_path=checkpoint, device=device)
        if adapter is not None:
            self.nora.model = _merge_adapter(self.nora.model, adapter)
        self.nora.fast_tokenizer.time_horizon = time_horizon
        self.nora.model.generation_config.do_sample = False
        self.unnorm_key = unnorm_key
        self.model = self.nora.model            # hooks attach here
        self.processor = self.nora.processor
        self.device = self.nora.device
        self.layers = find_decoder_layers(self.model)  # `add_direction(self, ...)` needs `.layers`
        self.steer = None                       # {"layer": int, "vec": np.ndarray, "alpha": float}, a list of those, or None
        self.patch = None                       # {"layers": [int], "source_instruction": str,
                                                #  "span_mode": "instruction"|"words"|"postinst",
                                                #  "source_policy": NoraPolicy (optional, cross-model)} or None
        self.ablate = None                      # {"vec": np.ndarray, "layers": [int] or None (= all)} or None
        self.forced_gen_ids = None              # list[int]: skip generation and decode these ids instead

    def _inputs(self, image: np.ndarray, instruction: str) -> dict:
        """NORA's prompt: one user turn (224x224 image + instruction), assistant turn opened."""
        from qwen_vl_utils import process_vision_info
        img = PIL.Image.fromarray(image)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img, "resized_height": 224, "resized_width": 224},
            {"type": "text", "text": instruction}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _patch_ctx(self, image: np.ndarray, inputs: dict):
        """Activation patching: capture the source instruction's residuals on the SAME image (one extra forward
        pass per policy call), then overwrite the target prompt's span with them while generating.
        With "source_policy" the capture runs on ANOTHER policy (e.g. the parent VLM): cross-model patching.
        Each side's span is computed with its own tokenizer, since the vocabularies differ."""
        layers = list(self.patch["layers"])
        mode = self.patch.get("span_mode", "instruction")
        source_policy = self.patch.get("source_policy") or self
        if source_policy is not self:
            check_source_policy(self, source_policy)
        source_acts, source_ids = capture_residuals(source_policy, image, self.patch["source_instruction"], layers)
        return patch_residuals(self, layers, source_acts,
                               span_for_mode(source_ids, source_policy.processor.tokenizer, mode),
                               span_for_mode(inputs["input_ids"], self.processor.tokenizer, mode))

    def _ctx(self, image: np.ndarray, inputs: dict):
        """The intervention context for one forward pass: patch, ablate, steer, or nothing.

        Factored out of `act()` so that `prefill()` can enter exactly the same context (a gate read
        that bypasses the ablation would make the wiring assertion vacuous)."""
        on = [n for n, v in (("steer", self.steer), ("patch", self.patch), ("ablate", self.ablate)) if v is not None]
        if len(on) > 1:
            raise RuntimeError(f"steer, patch and ablate are mutually exclusive; set at most one, got {on}")
        if self.patch is not None:
            return self._patch_ctx(image, inputs)
        if self.ablate is not None:
            layers = self.ablate.get("layers")
            return ablate_direction(self, self.ablate["vec"],
                                    list(range(len(self.layers))) if layers is None else list(layers))
        if self.steer is None:
            return contextlib.nullcontext()
        if isinstance(self.steer, dict):
            return add_direction(self, self.steer["layer"], self.steer["vec"], self.steer["alpha"])
        return add_directions(self, self.steer)

    @torch.inference_mode()
    def act(self, image: np.ndarray, instruction: str, max_new_tokens: int = 64) -> dict:
        """Returns {'action': (T,7) un-normalised or None, 'gen_ids': list[int], 'n_text': int, 'invalid': bool}.

        With `forced_gen_ids` set, nothing is sampled: one plain forward pass is run (so hooks that read the
        prompt still fire and the trigger scores are still logged) and those ids are decoded as if generated."""
        inputs = self._inputs(image, instruction)
        ctx = self._ctx(image, inputs)
        with ctx:
            if self.forced_gen_ids is None:
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                gen = out[0, inputs["input_ids"].shape[1]:]
            else:
                self.model(**inputs)  # prefill only: the read-out hooks fire, no token is sampled
                gen = torch.as_tensor(self.forced_gen_ids, dtype=torch.long, device=inputs["input_ids"].device)
        is_act = (gen >= ACTION_MIN) & (gen <= ACTION_MAX)
        specials = set(self.processor.tokenizer.all_special_ids)
        gen_list = gen.tolist()
        n_text = sum(1 for t in gen_list if t not in specials and not (ACTION_MIN <= t <= ACTION_MAX))
        act_ids = (gen[is_act] - ACTION_MIN).tolist()
        if not act_ids:
            return {"action": None, "gen_ids": gen_list, "n_text": n_text, "invalid": True}
        # The FAST processor swallows a malformed token string: it prints "Error decoding tokens" and returns an
        # all-zero DCT block, which un-normalises to a near-zero chunk (the "fake freeze" seen in the steering diagnostics). Reproduce its check
        # here so such a chunk is reported as invalid instead of as a hold.
        ft = self.nora.fast_tokenizer
        try:
            n_coeff = len(ft.bpe_tokenizer.decode(act_ids))
        except Exception:
            return {"action": None, "gen_ids": gen_list, "n_text": n_text, "invalid": True}
        if n_coeff != int(ft.time_horizon) * int(ft.action_dim):
            return {"action": None, "gen_ids": gen_list, "n_text": n_text, "invalid": True}
        try:
            norm = np.asarray(ft.decode([act_ids]))  # (1, T, 7) in [-1, 1]
        except Exception:
            return {"action": None, "gen_ids": gen_list, "n_text": n_text, "invalid": True}
        low, high = self.nora.libero_keys[self.unnorm_key]
        action = 0.5 * (norm + 1) * (high - low) + low
        return {"action": action.reshape(-1, 7), "gen_ids": gen_list, "n_text": n_text, "invalid": False}

    @torch.inference_mode()
    def prefill(self, image: np.ndarray, instruction: str, layers: list[int],
                positions: list[str]) -> dict[tuple[int, str], np.ndarray]:
        """Residual stream at the requested PROMPT positions from one forward pass, no generation.

        Two differences from `residuals_at`, both required by the parent-gated reference:
          * the forward runs inside `self._ctx(...)`, i.e. the same ablate / steer / patch context `act()`
            uses, so the gate sees what the policy sees;
          * positions are named ("t_inst", "t_post_inst") and resolved with `instruction_offsets`, which
            re-derives the negative offsets for every new instruction and asserts the template tail, instead
            of hard-coding the last prompt token.

        The read hooks are registered INSIDE the context, i.e. after the intervention hooks, so at the read
        layer the value returned is the POST-intervention residual (forward hooks run in registration order).
        """
        from refusal.conditional import instruction_offsets  # late: avoids a refusal.conditional <-> refusal.nora_policy cycle
        positions = list(positions)
        layers = [int(li) for li in layers]
        off = instruction_offsets(self, instruction)
        unknown = [p for p in positions if p not in off and p != MEAN_WORDS]
        if unknown:
            raise RuntimeError(f"unknown prompt positions {unknown}; have {sorted(off) + [MEAN_WORDS]}")
        bad = [li for li in layers if not 0 <= li < len(self.layers)]
        if bad:
            raise RuntimeError(f"layers {bad} outside 0..{len(self.layers) - 1}")
        inputs = self._inputs(image, instruction)
        got: dict[tuple[int, str], np.ndarray] = {}
        span = None
        if MEAN_WORDS in positions:      # mean-pooled probe: the mean over the instruction words, ending at t_inst
            span = span_for_mode(inputs["input_ids"], self.processor.tokenizer, "words")

        def make_hook(li):
            def hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                if h.shape[1] <= 1:
                    raise RuntimeError(f"prefill saw a decode step at layer {li}; no token may be generated here")
                for p in positions:
                    if p == MEAN_WORDS:
                        got[(li, p)] = h[0, span[0]:span[1], :].float().mean(0).cpu().numpy()
                        continue
                    idx = off[p]
                    if h.shape[1] < -idx:
                        raise RuntimeError(f"prefill length {h.shape[1]} shorter than offset {idx} at layer {li}")
                    got[(li, p)] = h[0, idx, :].detach().float().cpu().numpy()
            return hook

        with self._ctx(image, inputs):
            handles = [self.layers[li].register_forward_hook(make_hook(li)) for li in layers]
            try:
                self.model(**inputs)
            finally:
                for hd in handles:
                    hd.remove()
        missing = [(li, p) for li in layers for p in positions if (li, p) not in got]
        if missing:
            raise RuntimeError(f"no activation captured at {missing}; the prefill hooks never fired")
        return got

    @torch.inference_mode()
    def residuals_at(self, image: np.ndarray, instruction: str, layers: list[int]) -> dict[int, np.ndarray]:
        """Residual stream after each decoder block in `layers` at the last prompt token (t_post_inst), from ONE
        forward pass (no generation). Values are float32, shape (hidden,)."""
        inputs = self._inputs(image, instruction)
        got: dict[int, np.ndarray] = {}

        def make_hook(li):
            def hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                got[li] = h[0, -1, :].detach().float().cpu().numpy()
            return hook

        handles = [self.layers[li].register_forward_hook(make_hook(li)) for li in layers]
        try:
            self.model(**inputs)
        finally:
            for hd in handles:
                hd.remove()
        missing = [li for li in layers if li not in got]
        if missing:
            raise RuntimeError(f"no activation captured at layers {missing}; hooks never fired")
        return got

    def residual_at(self, image: np.ndarray, instruction: str, layer: int) -> np.ndarray:
        """Single-layer residuals_at."""
        return self.residuals_at(image, instruction, [layer])[layer]
