"""Repository paths, resolved in one place.

Every script reads its inputs and writes its outputs through these names, so the repository can be moved or the
data and results kept elsewhere without editing code:

  REPO_ROOT     the repository root (the folder holding `refusal/`, `experiments/`, `data/`, `results/`)
  DATA_DIR      command sets and other inputs; `REPO_ROOT/data` unless REFUSAL_DATA_DIR is set
  RESULTS_DIR   raw result rows, read and written; `REPO_ROOT/results` unless REFUSAL_RESULTS_DIR is set
  ADAPTERS_DIR  a local download of the Hugging Face model repo ADAPTER_REPO (one folder per adapter, each with
                its `verdict_gate.json`); `REPO_ROOT/adapters` unless REFUSAL_ADAPTERS_DIR is set

Download the adapters with, for example:

  huggingface-cli download fazmain/parent-taught-refusal --local-dir adapters
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _dir(env: str, default: Path) -> Path:
    v = os.environ.get(env)
    return Path(v).expanduser().resolve() if v else default


DATA_DIR = _dir("REFUSAL_DATA_DIR", REPO_ROOT / "data")
RESULTS_DIR = _dir("REFUSAL_RESULTS_DIR", REPO_ROOT / "results")
ADAPTERS_DIR = _dir("REFUSAL_ADAPTERS_DIR", REPO_ROOT / "adapters")
ADAPTER_REPO = "fazmain/parent-taught-refusal"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def repo_path(rel) -> Path:
    """A repository-relative path as written in code or in result rows ("data/...", "results/...") resolved
    against DATA_DIR / RESULTS_DIR; any other relative path against REPO_ROOT; absolute paths unchanged."""
    p = Path(str(rel))
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "data":
        return DATA_DIR.joinpath(*p.parts[1:])
    if p.parts and p.parts[0] == "results":
        return RESULTS_DIR.joinpath(*p.parts[1:])
    return REPO_ROOT / p


def display(p) -> str:
    """`p` relative to REPO_ROOT when it lies inside the repository, else the absolute path (for messages)."""
    p = Path(p)
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def add_experiment_dirs() -> None:
    """Put every `experiments/<n>_<section>/` folder on sys.path.

    Experiment scripts share helpers across sections (the simulator evaluation, the first-call evaluation, the
    training-row builder). The section folders start with a digit, so they are not importable as packages; each
    script is imported by its own module name instead (`import eval_in_simulator`). Module names are unique
    across sections."""
    for d in sorted(EXPERIMENTS_DIR.glob("[0-9]_*")):
        if d.is_dir() and str(d) not in sys.path:
            sys.path.append(str(d))
