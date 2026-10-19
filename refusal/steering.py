"""Residual-stream interventions (Arditi et al. 2024 style), applied via forward hooks on decoder blocks.

add_direction(L, layer, vec, alpha): h <- h + alpha * unit(vec) at every position of that block's output.
add_directions(L, specs): the same edit at several layers at once, one (vec, alpha) per layer.
ablate_direction(L, vec): h <- h - (h·û) û at every layer and position (removes the direction everywhere).
Both return a context manager; hooks are removed on exit. `vec` is a numpy array in the model's hidden size.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch


def _unit_tensor(vec: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    v = torch.as_tensor(np.asarray(vec, dtype=np.float32))
    v = v / (v.norm() + 1e-8)
    return v.to(device=like.device, dtype=like.dtype)


def _wrap(out, h):
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


@contextlib.contextmanager
def add_direction(L, layer: int, vec: np.ndarray, alpha: float):
    u = None
    def hook(_m, _i, out):
        nonlocal u
        h = out[0] if isinstance(out, tuple) else out
        if u is None:
            u = _unit_tensor(vec, h)
        return _wrap(out, h + alpha * u)
    handle = L.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextlib.contextmanager
def add_directions(L, specs: list[dict]):
    """Multi-layer add_direction: specs = [{"layer": int, "vec": np.ndarray, "alpha": float}, ...], one hook each."""
    def make_hook(vec: np.ndarray, alpha: float):
        u = None
        def hook(_m, _i, out):
            nonlocal u
            h = out[0] if isinstance(out, tuple) else out
            if u is None:
                u = _unit_tensor(vec, h)
            return _wrap(out, h + alpha * u)
        return hook
    handles = [L.layers[s["layer"]].register_forward_hook(make_hook(s["vec"], float(s["alpha"]))) for s in specs]
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()


@contextlib.contextmanager
def ablate_direction(L, vec: np.ndarray, layers: list[int] | None = None):
    u = None
    def hook(_m, _i, out):
        nonlocal u
        h = out[0] if isinstance(out, tuple) else out
        if u is None:
            u = _unit_tensor(vec, h)
        proj = (h * u).sum(-1, keepdim=True)
        return _wrap(out, h - proj * u)
    handles = [L.layers[li].register_forward_hook(hook) for li in (layers if layers is not None else range(L.n_layers))]
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()


@torch.inference_mode()
def mean_resid_norm(L, cache, texts: list[str], layer: int, image=None) -> float:
    """μ^(l): mean residual norm at `layer` over prompts (t_post_inst position), for α = c·μ (Korznikov et al.)."""
    from refusal.models import residuals
    norms = []
    for t in texts:
        acts, _ = residuals(L, cache, t, image)
        norms.append(float(acts[layer][1].norm()))
    return float(np.mean(norms))
