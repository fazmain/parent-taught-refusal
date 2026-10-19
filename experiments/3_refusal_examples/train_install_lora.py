#!/usr/bin/env python3
"""LoRA training for every install (the free install of Section 4.1 and every install of Section 5 and
Appendix A). Installs a halt behaviour into NORA.

One row of `train_install.jsonl` / `train_control.jsonl` is (image, instruction, label ids). The prompt is built
by the exact path `NoraPolicy._inputs` uses at inference (chat template, image resized to 224x224, assistant turn
opened), the label ids are appended, and the loss is taken on the appended ids only: labels are -100 over the
prompt and over the right padding. Install and control differ only in the halt rows of the file, never in a
hyper-parameter.

  python3 experiments/3_refusal_examples/train_install_lora.py --self-test                                     # collator only, no model
  python3 experiments/3_refusal_examples/train_install_lora.py --rows results/f1/data/train_install.jsonl \
      --out results/f1/adapters/preflight --preflight                                  # 24 rows, 30 steps
  python3 experiments/3_refusal_examples/train_install_lora.py --rows results/f1/data/train_install.jsonl \
      --out results/f1/adapters/install                                                # the real run (GPU)

Round 1 of the install search adds two options and changes nothing without them:
`--steps N` fixes the optimizer step count (every round-1 arm gets 2 epochs of the I0 file), and `--align-targets
<npz>` (arm I3) adds `--align-weight` x mean over the target layers of (1 - cos(mean over the instruction words of
NORA+LoRA's block output - mu_nora, the parent's precomputed target - mu_parent)) on the rows that carry an
`align_key` (both centring means are stored in the targets file); other rows get the action cross-entropy only. The
targets come from `build_training_rows.py --stage teacher_align`; no parent is loaded.

Round 2 (pre-registered) adds two more, again inert without them:
`--align-rows-ref <rows>` (arm I4) lets the targets bound to the I3 rows file serve another rows file whose rows with an
`align_key` carry the same keys, frames and sentences (asserted row by row against the bound file), and
`--restore-base <restore_base_g15.json>` (arm I6) merges the rank-32 parent-weight delta of
`train_verdict_alignment.py --stage restore` into the base weights before the LoRA is wrapped (`train_verdict_alignment.restore_base`); the
record, with the checksums of the merged base, goes into train_meta.json, and stage 0 must reproduce it.

Round 5 (pre-registered) adds no training behaviour at all. train_meta.json now
always carries `rows_sha1` (the sha1 of --rows), and `--rows-derived-from <approved rows>` records the path and sha1
of the approved file a derived row file was built from (arm I9 doubles the replay rows of the approved I2 file), so
an adapter can be tied to reviewed bytes it was not trained on directly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from refusal.paths import RESULTS_DIR, add_experiment_dirs  # noqa: E402
add_experiment_dirs()

TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
IGNORE = -100
N_TRAINABLE_RANGE = (1e7, 1e8)   # sanity band around the ~30 M expected for r=16 on a 3B language model


# --------------------------------------------------------------------------------- prompt / batch construction

def inputs_builder(processor, device="cpu"):
    """`NoraPolicy._inputs` bound to a processor, without loading a second 3 B model.

    The method touches only `self.processor` and `self.device`, so an uninitialised instance is enough. Using
    the method itself (rather than a copy of its body) is what guarantees that the training prompt and the
    inference prompt are the same string and the same token ids; `check_prompt_path` asserts it once anyway."""
    from refusal.nora_policy import NoraPolicy
    shim = NoraPolicy.__new__(NoraPolicy)
    shim.processor = processor
    shim.device = device
    return shim._inputs


def encode_sample(prompt: dict, label_ids: list[int]):
    """One training sample: prompt ids + label ids, labels masked to IGNORE over the prompt.

    `prompt` is a `NoraPolicy._inputs` result (batch of one). Pure torch, so the masking can be unit-tested on a
    fabricated tokenizer output with no model and no processor."""
    import torch
    ids = prompt["input_ids"][0].to(torch.long)
    lab = torch.as_tensor(label_ids, dtype=torch.long, device=ids.device)
    if lab.numel() == 0:
        raise SystemExit("empty label_ids")
    return {"input_ids": torch.cat([ids, lab]),
            "labels": torch.cat([torch.full((ids.numel(),), IGNORE, dtype=torch.long, device=ids.device), lab]),
            "n_prompt": int(ids.numel()),
            "pixel_values": prompt["pixel_values"],
            "image_grid_thw": prompt["image_grid_thw"]}


def collate(samples: list[dict], pad_id: int) -> dict:
    """Right-padded batch. Padding is masked out of both the attention mask and the labels."""
    import torch
    n = max(s["input_ids"].numel() for s in samples)
    input_ids, labels, attn = [], [], []
    for s in samples:
        k = n - s["input_ids"].numel()
        input_ids.append(torch.cat([s["input_ids"], torch.full((k,), pad_id, dtype=torch.long)]))
        labels.append(torch.cat([s["labels"], torch.full((k,), IGNORE, dtype=torch.long)]))
        attn.append(torch.cat([torch.ones(s["input_ids"].numel(), dtype=torch.long),
                               torch.zeros(k, dtype=torch.long)]))
    return {"input_ids": torch.stack(input_ids), "labels": torch.stack(labels),
            "attention_mask": torch.stack(attn),
            "pixel_values": torch.cat([s["pixel_values"] for s in samples], dim=0),
            "image_grid_thw": torch.cat([s["image_grid_thw"] for s in samples], dim=0)}


def load_image(data_dir: Path, rel: str) -> np.ndarray:
    import PIL.Image
    p = data_dir / rel
    if not p.exists():
        raise SystemExit(f"training row points at a missing frame: {p}")
    return np.asarray(PIL.Image.open(p).convert("RGB"), dtype=np.uint8)


def make_batch(build, data_dir: Path, rows: list[dict], pad_id: int) -> dict:
    return collate([encode_sample(build(load_image(data_dir, r["image"]), r["instruction"]), r["label_ids"])
                    for r in rows], pad_id)


def check_prompt_path(build, data_dir: Path, row: dict, pad_id: int) -> None:
    """Assert once that the prompt part of a batch is exactly what `NoraPolicy._inputs` produces at inference."""
    import torch
    ref = build(load_image(data_dir, row["image"]), row["instruction"])
    batch = make_batch(build, data_dir, [row], pad_id)
    n = ref["input_ids"].shape[1]
    if not torch.equal(batch["input_ids"][0, :n].cpu(), ref["input_ids"][0].cpu()):
        raise SystemExit("the training prompt tokens differ from NoraPolicy._inputs for the same image and "
                         "instruction; the two prompt paths have diverged")
    if batch["labels"][0, :n].ne(IGNORE).any():
        raise SystemExit("labels are not masked over the whole prompt")
    if batch["input_ids"][0, n:].tolist() != list(row["label_ids"]):
        raise SystemExit("the appended ids are not the row's label ids")
    if not torch.equal(batch["pixel_values"].cpu(), ref["pixel_values"].cpu()):
        raise SystemExit("pixel_values differ between the training batch and NoraPolicy._inputs")
    import hashlib
    print(f"[check] prompt path matches NoraPolicy._inputs: {n} prompt tokens + {len(row['label_ids'])} label "
          f"tokens on {row['image']!r} / {row['row_type']} instruction sha1 "
          f"{hashlib.sha1(row['instruction'].encode('utf-8')).hexdigest()[:16]}", flush=True)


# --------------------------------------------------------------------------------- I3: alignment term

ALIGN_SPAN_MODE = "words"    # the span the parent targets were averaged over (refusal.patching.span_for_mode)


def make_align_batch(build, data_dir: Path, rows: list[dict], pad_id: int, tokenizer):
    """`make_batch` plus, per row, the instruction-word span of its prompt (None for a row without `align_key`).
    The span is read on the prompt ids alone; the label ids are appended after the prompt and the padding goes to
    the right, so the same indices address the batch."""
    from refusal.patching import span_for_mode
    samples, spans = [], []
    for r in rows:
        prompt = build(load_image(data_dir, r["image"]), r["instruction"])
        spans.append(span_for_mode(prompt["input_ids"], tokenizer, ALIGN_SPAN_MODE) if r.get("align_key") else None)
        samples.append(encode_sample(prompt, r["label_ids"]))
    return collate(samples, pad_id), spans


def align_term(hidden: dict, spans: list, targets: dict, mu: dict | None = None):
    """Mean over the rows that have a span and over the layers of 1 - cos(mean over the span of hidden[layer][row]
    - mu[layer], targets[layer][row]), in fp32 (`mu` None: nothing subtracted); None when no row has a span."""
    import torch
    import torch.nn.functional as F
    terms = []
    for li in sorted(hidden):
        for b, sp in enumerate(spans):
            if sp is None:
                continue
            m = hidden[li][b, sp[0]:sp[1], :].float().mean(0)
            if mu is not None:
                m = m - mu[li].float()
            terms.append(1.0 - F.cosine_similarity(m, targets[li][b].float(), dim=0))
    return torch.stack(terms).mean() if terms else None


def span_ids_sha1(ids) -> str:
    """sha1[:16] of the space-joined token ids (the rule build_training_rows.span_ids_sha1 stores for the parent)."""
    import hashlib
    return hashlib.sha1(" ".join(str(int(i)) for i in ids).encode("utf-8")).hexdigest()[:16]


ALIGN_REF_FIELDS = ("image", "instruction", "raw_instruction", "row_type")


def same_rows_as_ref(rows: list[dict], ref_rows: list[dict], rows_path, ref_path) -> None:
    """I4 (round 2): the rows with an `align_key` must be, key for key, the rows with an `align_key` of the file the
    targets are bound to: the same keys, each once, on the same frame and sentence (`ALIGN_REF_FIELDS`). Labels may
    differ; the alignment term reads the prompt only."""
    mine = [r for r in rows if r.get("align_key")]
    ref = [r for r in ref_rows if r.get("align_key")]
    by_mine = {r["align_key"]: r for r in mine}
    by_ref = {r["align_key"]: r for r in ref}
    if len(by_mine) != len(mine) or len(by_ref) != len(ref):
        raise SystemExit(f"duplicate align_key: {rows_path} {len(mine)} rows / {len(by_mine)} keys, {ref_path} "
                         f"{len(ref)} rows / {len(by_ref)} keys")
    if set(by_mine) != set(by_ref):
        raise SystemExit(f"{rows_path} and {ref_path} carry different align_key sets ({len(set(by_mine) - set(by_ref))} "
                         f"keys only in the first, {len(set(by_ref) - set(by_mine))} only in the second)")
    bad = [k for k in by_mine if any(by_mine[k][f] != by_ref[k][f] for f in ALIGN_REF_FIELDS)]
    if bad:
        raise SystemExit(f"{len(bad)} rows of {rows_path} differ from {ref_path} in frame or sentence, e.g. {bad[0]}")
    print(f"[align] the {len(mine)} rows with an align_key in {rows_path} match {ref_path} key for key "
          f"({', '.join(ALIGN_REF_FIELDS)}); the targets are bound to the latter", flush=True)


class AlignLoss:
    """The I3 auxiliary term. Targets are bound to the rows file (sha1), keyed by `align_key`, and checked against
    each row's frame and sentence hash; the instruction-word token ids NORA sees must be the parent's (hash, unless
    the targets are stubs). Hooks on the decoder blocks read the block outputs of the forward that gives the CE.
    CENTRED (round-1 data v4): cos(NORA mean-words - mu_nora, target - mu_parent), both means stored in the targets file (the
    raw targets' class means have cosine 0.99, so an uncentred cosine mostly pulls toward a shared direction)."""

    def __init__(self, path, rows_path, rows: list[dict], weight: float, device, ref_path=None):
        import hashlib
        import torch
        z = np.load(path, allow_pickle=False)
        bound = Path(ref_path) if ref_path else Path(rows_path)
        got = hashlib.sha1(bound.read_bytes()).hexdigest()
        if str(z["rows_sha1"]) != got:
            raise SystemExit(f"{path} was built for rows sha1 {str(z['rows_sha1'])[:16]}, {bound} is {got[:16]}")
        if ref_path:
            same_rows_as_ref(rows, read_rows(bound), rows_path, bound)
        keys = [str(k) for k in z["keys"]]
        self.index = {k: i for i, k in enumerate(keys)}
        by_key = {r["align_key"]: r for r in rows if r.get("align_key")}
        if len(self.index) != len(keys) or set(by_key) != set(keys):
            raise SystemExit(f"{len(keys)} targets ({len(self.index)} distinct) for {len(by_key)} rows with an "
                             f"align_key; the sets differ")
        for k, i in self.index.items():
            r = by_key[k]
            sh = hashlib.sha1(r["raw_instruction"].encode("utf-8")).hexdigest()[:16]
            if str(z["image"][i]) != r["image"] or str(z["text_sha1"][i]) != sh:
                raise SystemExit(f"target {k} was computed on another frame or sentence")
        self.layers = [int(li) for li in z["layers"]]
        missing = [f"{k}_L{li}" for li in self.layers for k in ("mu_parent", "mu_nora") if f"{k}_L{li}" not in z.files]
        if missing:
            raise SystemExit(f"{path} has no centring means {missing[:2]}; rebuild it with g14_build_rows.py v4")
        self.mu_nora = {li: torch.as_tensor(z[f"mu_nora_L{li}"].astype(np.float32), device=device) for li in self.layers}
        self.targets = {li: torch.as_tensor(z[f"L{li}"].astype(np.float32) - z[f"mu_parent_L{li}"].astype(np.float32),
                                            device=device) for li in self.layers}
        self.n_span = np.asarray(z["n_span"], dtype=np.int64)
        self.span_sha1 = [str(s) for s in z["span_sha1"]]
        self.stub = bool(int(z["stub"]))
        self.weight = float(weight)
        self.n_rows = len(keys)
        self.path = str(path)
        self.blocks = None
        self.hist_ce, self.hist_align = [], []

    def attach(self, model) -> None:
        from refusal.models import find_decoder_layers
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.blocks = find_decoder_layers(base)
        bad = [li for li in self.layers if not 0 <= li < len(self.blocks)]
        if bad:
            raise SystemExit(f"alignment layers {bad} outside 0..{len(self.blocks) - 1}")

    def loss(self, model, build, data_dir: Path, rows: list[dict], pad_id: int, tokenizer, device):
        batch, spans = make_align_batch(build, data_dir, rows, pad_id, tokenizer)
        idx = []
        for b, (r, sp) in enumerate(zip(rows, spans)):
            if sp is None:
                idx.append(0)
                continue
            i = self.index[r["align_key"]]
            if not self.stub and span_ids_sha1(batch["input_ids"][b, sp[0]:sp[1]].tolist()) != self.span_sha1[i]:
                raise SystemExit(f"row {r['align_key']}: NORA's instruction words ({sp[1] - sp[0]} tokens) differ from "
                                 f"the parent's ({int(self.n_span[i])} tokens, span sha1 {self.span_sha1[i]})")
            idx.append(i)
        batch = {k: v.to(device) for k, v in batch.items()}
        got = {}

        def make_hook(li):
            def hook(_m, _i, out):
                got[li] = out[0] if isinstance(out, tuple) else out
            return hook

        handles = [self.blocks[li].register_forward_hook(make_hook(li)) for li in self.layers]
        try:
            out = model(**batch)
        finally:
            for hd in handles:
                hd.remove()
        if set(got) != set(self.layers):
            raise SystemExit(f"alignment hooks fired at {sorted(got)}, need {self.layers}")
        tg = {li: self.targets[li][idx] for li in self.layers}
        al = align_term(got, spans, tg, self.mu_nora)
        ce = out.loss
        self.hist_ce.append(float(ce.detach().float()))
        if al is not None:
            self.hist_align.append(float(al.detach().float()))
        return (ce if al is None else ce + self.weight * al), ce, al


# --------------------------------------------------------------------------------- model

def lora_targets(model) -> tuple[list[str], list[str], int]:
    """Explicit module names so the vision tower and the merger can be shown to be untouched."""
    import torch.nn as nn
    from refusal.models import find_decoder_layers
    n_layers = len(find_decoder_layers(model))
    lin = [n for n, m in model.named_modules()
           if isinstance(m, nn.Linear) and n.split(".")[-1] in TARGET_SUFFIXES]
    vision = [n for n in lin if n.startswith("visual") or ".visual." in n]
    targets = [n for n in lin if n not in set(vision)]
    expect = n_layers * len(TARGET_SUFFIXES)
    if len(targets) != expect:
        raise SystemExit(f"expected {expect} LoRA targets ({n_layers} decoder layers x {len(TARGET_SUFFIXES)} "
                         f"projections), found {len(targets)}: {targets[:8]}")
    return targets, vision, n_layers


def build_model(a):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(a.checkpoint, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.checkpoint, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to(device)
    model.generation_config.do_sample = False
    restore = None
    if a.restore_base:
        from train_verdict_alignment import restore_base       # I6 (round 2): before the LoRA is wrapped
        from refusal.models import find_decoder_layers
        restore = restore_base(model, find_decoder_layers(model), a.restore_base, a.checkpoint, device)
    targets, vision, n_layers = lora_targets(model)
    print(f"[lora] {len(targets)} target modules over {n_layers} decoder layers; {len(vision)} vision-tower "
          f"matches excluded; merger and vision tower stay frozen", flush=True)
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none",
                                             task_type="CAUSAL_LM", target_modules=targets))
    frozen_but_trainable = [n for n, p in model.named_parameters() if p.requires_grad and "visual" in n]
    if frozen_but_trainable:
        raise SystemExit(f"vision parameters are trainable: {frozen_but_trainable[:5]}")
    n_trainable = 0
    for _, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()          # fp32 LoRA parameters over a bf16 base: peft casts per forward
            n_trainable += p.numel()
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable {n_trainable / 1e6:.1f} M / {n_total / 1e6:.1f} M parameters "
          f"({100 * n_trainable / n_total:.2f}%)", flush=True)
    if not N_TRAINABLE_RANGE[0] <= n_trainable <= N_TRAINABLE_RANGE[1]:
        raise SystemExit(f"{n_trainable} trainable parameters is outside the expected band {N_TRAINABLE_RANGE}")
    info = {"n_targets": len(targets), "n_layers": n_layers, "n_trainable": int(n_trainable), "n_total": int(n_total)}
    if restore is not None:
        info["base_restore"] = restore
    return model, processor, device, info


# --------------------------------------------------------------------------------- training

def read_rows(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"{path} is empty")
    return rows


def sha1_file(path) -> str:
    import hashlib
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def preflight_rows(rows: list[dict], n: int, seed: int) -> list[dict]:
    """A stratified sample, so the preflight loss moves on both the halt rows and the execute rows."""
    rng = np.random.default_rng(seed)
    by_type = {}
    for r in rows:
        by_type.setdefault(r["row_type"], []).append(r)
    per = max(1, n // len(by_type))
    out = []
    for t in sorted(by_type):
        idx = rng.choice(len(by_type[t]), size=min(per, len(by_type[t])), replace=False)
        out += [by_type[t][int(i)] for i in sorted(idx)]
    return out[:n]


def train(a, model, build, data_dir: Path, rows: list[dict], pad_id: int, device: str, steps: int | None,
          align: AlignLoss | None = None, tokenizer=None):
    import torch
    from transformers import get_cosine_schedule_with_warmup
    log_path = Path(a.out) / "train_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    per_epoch = max(1, len(rows) // a.batch)
    total = steps if steps is not None else per_epoch * a.epochs
    warmup = min(a.warmup, max(1, total // 10))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, warmup, total)
    rng = np.random.default_rng(a.seed)
    model.train()
    print(f"[train] {len(rows)} rows, batch {a.batch}, {per_epoch} steps/epoch, {total} steps total, "
          f"lr {a.lr} cosine with {warmup} warmup steps, seed {a.seed}", flush=True)
    losses, window, t0 = [], [], time.time()
    ce_window, align_window = [], []
    order, cursor = rng.permutation(len(rows)), 0
    with log_path.open("a") as log:
        for step in range(total):
            if cursor + a.batch > len(order):
                order, cursor = rng.permutation(len(rows)), 0
            batch_rows = [rows[int(i)] for i in order[cursor:cursor + a.batch]]
            cursor += a.batch
            if align is None:
                batch = make_batch(build, data_dir, batch_rows, pad_id)
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss
            else:
                loss, ce, al = align.loss(model, build, data_dir, batch_rows, pad_id, tokenizer, device)
                ce_window.append(float(ce.detach().float()))
                if al is not None:
                    align_window.append(float(al.detach().float()))
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            v = float(loss.detach().float())
            if not np.isfinite(v):
                raise SystemExit(f"loss is {v} at step {step}; training diverged")
            losses.append(v)
            window.append(v)
            if (step + 1) % a.log_every == 0 or step == total - 1:
                row = {"step": step + 1, "total": total, "epoch": round((step + 1) / per_epoch, 3),
                       "loss": round(float(np.mean(window)), 5), "lr": round(float(sched.get_last_lr()[0]), 8),
                       "grad_norm": round(float(gnorm), 4), "wall_s": round(time.time() - t0, 1),
                       "types": dict(Counter(r["row_type"] for r in batch_rows))}
                if align is not None:
                    row["loss_ce"] = round(float(np.mean(ce_window)), 5)
                    row["loss_align"] = round(float(np.mean(align_window)), 5) if align_window else None
                    row["n_align_batches"] = len(align_window)
                log.write(json.dumps(row) + "\n")
                log.flush()
                if align is None:
                    print(f"[train] step {row['step']:4d}/{total} epoch {row['epoch']:.2f} loss {row['loss']:.5f} "
                          f"lr {row['lr']:.2e} |g| {row['grad_norm']:.3f} wall {row['wall_s']:.0f}s", flush=True)
                else:
                    al_txt = f"{row['loss_align']:.5f}" if row["loss_align"] is not None else "n/a"
                    print(f"[train] step {row['step']:4d}/{total} epoch {row['epoch']:.2f} loss {row['loss']:.5f} "
                          f"= ce {row['loss_ce']:.5f} + {align.weight:g} x align {al_txt} "
                          f"({row['n_align_batches']} batches with corpus rows) lr {row['lr']:.2e} "
                          f"|g| {row['grad_norm']:.3f} wall {row['wall_s']:.0f}s", flush=True)
                window = []
                ce_window, align_window = [], []
    return losses


def generate_ids(model, build, data_dir: Path, rows: list[dict], device: str, max_new_tokens: int = 64):
    """Greedy generation for the preflight check; prints the raw ids, never a decoded string."""
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for r in rows:
            inputs = build(load_image(data_dir, r["image"]), r["instruction"])
            inputs = {k: v.to(device) for k, v in inputs.items()}
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            ids = gen[0, inputs["input_ids"].shape[1]:].tolist()
            out.append({"row_type": r["row_type"], "instruction": r["instruction"], "image": r["image"],
                        "label_ids": r["label_ids"], "gen_ids": [int(t) for t in ids]})
            import hashlib
            print(f"[preflight] {r['row_type']:<18s} sha1 {hashlib.sha1(r['instruction'].encode('utf-8')).hexdigest()[:16]} "
                  f"-> {out[-1]['gen_ids']}", flush=True)
    model.train()
    return out


# --------------------------------------------------------------------------------- self-test (no model)

def self_test() -> None:
    """The collator's label masking, on a fabricated tokenizer output. No transformers, no peft, no GPU."""
    import torch
    fake_a = {"input_ids": torch.tensor([[10, 11, 12, 13]]), "attention_mask": torch.ones(1, 4, dtype=torch.long),
              "pixel_values": torch.arange(12, dtype=torch.float32).reshape(4, 3),
              "image_grid_thw": torch.tensor([[1, 2, 2]])}
    fake_b = {"input_ids": torch.tensor([[20, 21, 22, 23, 24, 25]]),
              "attention_mask": torch.ones(1, 6, dtype=torch.long),
              "pixel_values": torch.arange(12, 24, dtype=torch.float32).reshape(4, 3),
              "image_grid_thw": torch.tensor([[1, 2, 2]])}
    la, lb = [901, 902, 151645], [903, 151645]
    sa, sb = encode_sample(fake_a, la), encode_sample(fake_b, lb)
    assert sa["input_ids"].tolist() == [10, 11, 12, 13] + la, sa["input_ids"].tolist()
    assert sa["labels"].tolist() == [IGNORE] * 4 + la, sa["labels"].tolist()
    assert sa["n_prompt"] == 4
    pad = 151643
    b = collate([sa, sb], pad)
    assert b["input_ids"].shape == (2, 8), b["input_ids"].shape
    assert b["input_ids"][0].tolist() == [10, 11, 12, 13] + la + [pad], b["input_ids"][0].tolist()
    assert b["labels"][0].tolist() == [IGNORE] * 4 + la + [IGNORE], b["labels"][0].tolist()
    assert b["labels"][1].tolist() == [IGNORE] * 6 + lb, b["labels"][1].tolist()
    assert b["attention_mask"][0].tolist() == [1] * 7 + [0]
    assert b["attention_mask"][1].tolist() == [1] * 8
    assert (b["labels"][b["attention_mask"] == 0] == IGNORE).all(), "padding is not masked out of the labels"
    kept = b["labels"][b["labels"] != IGNORE].tolist()
    assert kept == la + lb, kept
    assert b["pixel_values"].shape == (8, 3), b["pixel_values"].shape
    assert b["image_grid_thw"].tolist() == [[1, 2, 2], [1, 2, 2]]
    print(f"[self-test] label masking OK: batch {tuple(b['input_ids'].shape)}, "
          f"{len(kept)} supervised tokens ({la} | {lb}), everything else {IGNORE}")
    print("[self-test] OK", flush=True)


# --------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(RESULTS_DIR / "f1/data/train_install.jsonl"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "f1/adapters/install"))
    ap.add_argument("--data-dir", default=None, help="root the row image paths are relative to "
                    "(default: the directory of --rows)")
    ap.add_argument("--checkpoint", default="declare-lab/nora-long-finetuned-libero-goal")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None, help="optimizer steps (the same for every arm of a round); "
                    "default epochs x (rows // batch)")
    ap.add_argument("--align-targets", default=None, help="I3: the parent targets npz of build_training_rows.py "
                    "--stage teacher_align; adds the alignment term on rows with an align_key")
    ap.add_argument("--align-weight", type=float, default=1.0, help="I3: weight of the alignment term")
    ap.add_argument("--align-rows-ref", default=None, help="I4: the rows file the targets are bound to, when "
                    "--rows is another file with the same align_key rows (same frames and sentences)")
    ap.add_argument("--restore-base", default=None, help="I6: restore_base_g15.json of train_verdict_alignment.py --stage "
                    "restore; its rank-32 parent delta is merged into the base weights before the LoRA")
    ap.add_argument("--rows-derived-from", default=None, help="I9: the approved row file --rows was built from; "
                    "its path and sha1 go into train_meta.json beside the sha1 of --rows")
    ap.add_argument("--preflight", action="store_true", help="24 rows, 30 steps, then generate and print ids")
    ap.add_argument("--preflight-rows", type=int, default=24)
    ap.add_argument("--preflight-steps", type=int, default=30)
    ap.add_argument("--self-test", action="store_true", help="collator assertions only, then exit")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return

    import torch
    rows_path = Path(a.rows)
    data_dir = Path(a.data_dir) if a.data_dir else rows_path.parent
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(rows_path)
    rows_sha1 = sha1_file(rows_path)
    print(f"[data] {len(rows)} rows from {rows_path} (sha1 {rows_sha1[:16]}): "
          f"{dict(Counter(r['row_type'] for r in rows))}", flush=True)
    derived = None
    if a.rows_derived_from:
        src_rows = Path(a.rows_derived_from)
        if not src_rows.exists():
            raise SystemExit(f"--rows-derived-from {src_rows} does not exist")
        derived = {"path": str(src_rows), "sha1": sha1_file(src_rows)}
        print(f"[data] derived from {src_rows} (sha1 {derived['sha1'][:16]})", flush=True)
    torch.manual_seed(a.seed)
    model, processor, device, info = build_model(a)
    build = inputs_builder(processor, device="cpu")     # tensors are moved to the GPU per batch in `train`
    tok = getattr(processor, "tokenizer", processor)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad_id is None:
        raise SystemExit("the tokenizer has neither a pad token nor an eos token")
    check_prompt_path(build, data_dir, rows[0], pad_id)
    align = None
    if a.align_rows_ref and not a.align_targets:
        raise SystemExit("--align-rows-ref needs --align-targets")
    if a.align_targets:
        align = AlignLoss(a.align_targets, rows_path, rows, a.align_weight, device, ref_path=a.align_rows_ref)
        align.attach(model)
        print(f"[align] {align.n_rows} rows carry a parent target ({a.align_targets}, layers {align.layers}, "
              f"weight {align.weight:g}{', STUB targets' if align.stub else ''}); the other "
              f"{len(rows) - align.n_rows} rows get the action cross-entropy only", flush=True)

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.time()
    if a.preflight:
        sample = preflight_rows(rows, a.preflight_rows, a.seed)
        print(f"[preflight] {len(sample)} rows ({dict(Counter(r['row_type'] for r in sample))}), "
              f"{a.preflight_steps} steps", flush=True)
        losses = train(a, model, build, data_dir, sample, pad_id, device, steps=a.preflight_steps)
        k = max(2, len(losses) // 5)
        first, last = float(np.mean(losses[:k])), float(np.mean(losses[-k:]))
        halt = [r for r in sample if r["row_type"] == "halt"][:3]
        repl = [r for r in sample if r["row_type"] != "halt"][:3]
        if not halt or not repl:
            raise SystemExit(f"preflight sample has no halt or no execute rows: "
                             f"{dict(Counter(r['row_type'] for r in sample))}")
        gens = generate_ids(model, build, data_dir, halt + repl, device)
        print(f"[preflight] loss {first:.4f} (first {k} steps) -> {last:.4f} (last {k} steps)", flush=True)
        if not last < first:
            raise SystemExit(f"preflight loss did not fall: {first:.4f} -> {last:.4f}")
        (out / "preflight.json").write_text(json.dumps(
            {"exp": "f1_train_lora_preflight", "ts": stamp, "rows": str(rows_path), "n_rows": len(sample),
             "steps": a.preflight_steps, "loss_first": first, "loss_last": last, "generations": gens,
             "wall_s": round(time.time() - t0, 1), **info}, indent=2))
        print(f"[preflight] OK, wrote {out / 'preflight.json'} (no adapter saved)")
        print("DONE", out)
        return

    losses = train(a, model, build, data_dir, rows, pad_id, device, steps=a.steps, align=align, tokenizer=tok)
    model.save_pretrained(str(out))
    k = max(5, len(losses) // 20)
    meta = {"exp": "f1_train_lora", "ts": stamp, "rows": str(rows_path), "rows_sha1": rows_sha1,
            "rows_derived_from": derived, "out": str(out),
            "checkpoint": a.checkpoint, "n_rows": len(rows),
            "row_types": dict(Counter(r["row_type"] for r in rows)), "steps": len(losses),
            "loss_first": float(np.mean(losses[:k])), "loss_last": float(np.mean(losses[-k:])),
            "wall_s": round(time.time() - t0, 1), **info,
            "args": {k2: v for k2, v in vars(a).items()}}
    if align is not None:
        ka = max(5, len(align.hist_align) // 20)
        meta["align"] = {"targets": align.path, "weight": align.weight, "layers": align.layers,
                         "n_rows_with_target": align.n_rows, "stub_targets": align.stub,
                         "n_batches_with_target": len(align.hist_align),
                         "loss_ce_first": float(np.mean(align.hist_ce[:k])),
                         "loss_ce_last": float(np.mean(align.hist_ce[-k:])),
                         "loss_align_first": float(np.mean(align.hist_align[:ka])) if align.hist_align else None,
                         "loss_align_last": float(np.mean(align.hist_align[-ka:])) if align.hist_align else None}
        print(f"[align] ce {meta['align']['loss_ce_first']:.4f} -> {meta['align']['loss_ce_last']:.4f}, align "
              f"{meta['align']['loss_align_first']} -> {meta['align']['loss_align_last']} over "
              f"{len(align.hist_align)} batches with corpus rows", flush=True)
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[train] loss {meta['loss_first']:.4f} -> {meta['loss_last']:.4f} over {len(losses)} steps, "
          f"{meta['wall_s']:.0f}s; adapter saved to {out}", flush=True)
    print("DONE", out)


if __name__ == "__main__":
    main()
