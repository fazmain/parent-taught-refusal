# Reproducing the paper

There are two levels of reproduction:

1. **Check every number from the released rows (CPU, seconds).** `scripts/verify_release.py` recomputes all 1,136
   printed numbers from `data/` and `results/`. `scripts/make_tables.py` and `scripts/make_figures.py` rebuild the
   tables and figures. Nothing is downloaded.
2. **Re-run the experiments (GPU).** The scripts in `experiments/` load the models from Hugging Face, train the
   adapters and run the LIBERO simulator. The whole project used about $26 of GPU time over 21 sessions on one
   24 to 48 GB GPU at a time: an RTX A5000 24 GB for most runs, an A40 or A6000 48 GB for a few.

## 1. CPU checks (no GPU, no model download)

```
pip install -e .                       # Python >= 3.10; or: pip install -r requirements.txt (core block)
python scripts/verify_release.py       # about 1 s; exit 0 = every recomputable number matches
python scripts/make_tables.py          # about 2 s; writes tables/*.txt and tables/*.tex
python scripts/make_figures.py         # about 3 s; writes figures/*.pdf, *.png, *.json
```

### `verify_release.py`

**What it reads.**
- The printed values come from [paper_numbers.json](paper_numbers.json), one entry per number in the arXiv paper,
  with its tex file and line.
- The rules come from `scripts/claims/part_*.py`.
- [paper_claims.md](paper_claims.md) is the same list as a readable table.

**How it compares.** Counts must match exactly. Rates must match to the printed precision.

**Expected output** (last lines):

```
claims 1136: recomputed and matched 1111, mismatched 0, recorded not recomputed 5 (project log, private), external 20 (cited papers)
PASS: every recomputable number matches the paper
```

**Numbers it does not recompute.**
- The 5 *recorded* numbers exist only in the private project log: the smallest and largest GPU memory (24 and
  48 GB), the spend ($26), the session count (21), and the pre-registered 0.75 bar for NORA's read-out gate. They are
  stored in `paper_numbers.json` with `"source": "project log (private)"` and reported, not failed.
- The 20 *external* numbers are quoted from cited papers.

### `make_tables.py`

It writes every paper table to `tables/<name>.txt` (plain text) and `tables/<name>.tex` (a LaTeX tabular):
- `reporting`
- `ablation`
- `reference`
- `routes`
- `search` (Appendix A)
- `libero_safety` (Appendix H)

With `--compare-tex <arXiv source folder>` it compares them cell by cell with the paper's own tables. The release
was checked this way: 556 cells, 0 differences.

### `make_figures.py`

Rebuilds the paper's figures:

| Figure | Paper | Status from the released files |
|---|---|---|
| `fig1_transfer` | Section 3, the parent's directions inside NORA | identical to the paper |
| `fig2_patching` | Section 3, activation patching | identical to the paper |
| `fig3_tradeoff` | Appendix B, gates of the parent-gated policy | identical to the paper |
| `fig_curve_v2` | Section 5, the offset trades catch against over-refusal | partial, see below |
| `fig_contact` | Appendix D, contact sheet | **skipped** (needs the episode videos) |
| `method` | Section 5, the parent-taught verdict overview (TikZ) | needs `pdflatex` and `pdftoppm`; skipped with a message otherwise |

**`fig_curve_v2` is partial.**
- The full curves exist for the five round-3 installs, whose per-call decision logits are shipped
  (`results/offsets/round3_offset_log.jsonl`).
- The seed-2, replay-doubled, half-learning-rate and construction-label arms appear only as their two measured points
  (b = 0 and b_95). Their per-call logits were not released.

**`fig_contact` is skipped.** It is made of video frames of the scripted-hold and steering episodes. Videos and
rendered frames are not released.

## 2. Re-running the experiments (GPU)

### Environment

**Core environment.** Python 3.11, CUDA 12.4, the core block of [requirements.txt](../requirements.txt) (torch 2.4.1,
`transformers==4.50.0`, peft 0.15.2), then `pip install -e .`.

`transformers==4.50.0` is load-bearing: NORA's checkpoint and `third_party/nora/nora_utils.py` were written against
it.

**Models.** Downloaded from Hugging Face at run time:
- `declare-lab/nora-long-finetuned-libero-goal`, the policy;
- `Qwen/Qwen2.5-VL-3B-Instruct`, the parent;
- `declare-lab/nora`, the generalist NORA used in Section 3 only.

**LIBERO-Goal simulator** (Sections 2 to 5). As in the original setup scripts:

```
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git      # runs used commit 8f1084e3132a39270c3a13ebe37270a43ece2a01
pip install "numpy==1.26.4" "imageio[ffmpeg]" "mujoco==2.3.7" robosuite==1.4.1 bddl==1.0.1 easydict cloudpickle \
    gym==0.25.2 hydra-core matplotlib h5py future
pip install --no-deps -e LIBERO                                       # never LIBERO's own requirements.txt
echo "$PWD/LIBERO" > "$(python -c 'import site; print(site.getsitepackages()[0])')/libero_path.pth"
python - <<'PY'
import os, yaml
root = os.path.abspath("LIBERO/libero/libero")
cfg = {"benchmark_root": root, "bddl_files": f"{root}/bddl_files", "init_states": f"{root}/init_files",
       "datasets": os.path.abspath("LIBERO/datasets"), "assets": f"{root}/assets"}
os.makedirs(os.path.expanduser("~/.libero"), exist_ok=True)
open(os.path.expanduser("~/.libero/config.yaml"), "w").write(yaml.safe_dump(cfg))
PY
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl                            # headless rendering
# system packages: libegl1 libgl1 libgles2 libglvnd0 libglvnd-dev libosmesa6 libglib2.0-0
```

Two details matter:
- The `.pth` file is needed because the editable install alone does not expose the top-level `libero` package.
- `config.yaml` must exist before the first import, or `libero.libero` asks for the paths interactively.

**LIBERO-Safety** (Appendix H). It uses its own fork, in **its own environment**. Never install it next to LIBERO
above.

1. Install the fork:

   ```
   git clone https://github.com/LIBERO-SAFETY/LIBERO-Safety            # runs used commit 19ec8df23eedfbb9265bafd3e56495fcebfcfcd0
   pip install --no-deps -e LIBERO-Safety/third_party/robosuite-1.4
   pip install --no-deps -e LIBERO-Safety
   ```

2. Add a `libero_safety_path.pth` with the checkout root and a `~/.libero/config.yaml`, as above.
3. Install `termcolor==3.3.0 Wand==0.7.2 opencv-python==4.6.0.66 scikit-image==0.26.0 numba==0.68.0`, plus the system
   package `libmagickwand-dev`.
4. Unzip the scene assets into `libero/libero/assets`. They are in the Hugging Face dataset
   `LIBERO-Safety/libero_safety_assets` (`assets.zip`).
5. **Replace the fork's controller configuration with stock robosuite 1.4.1's.** The fork ships an
   `osc_pose.json` with an action range about 40 times larger than the controller NORA was trained with. Restore
   the stock file:

   ```
   curl -sfL https://raw.githubusercontent.com/ARISE-Initiative/robosuite/v1.4.1/robosuite/controllers/config/osc_pose.json \
       -o LIBERO-Safety/third_party/robosuite-1.4/robosuite/controllers/config/osc_pose.json
   ```

   Both versions are recorded in `results/libero_safety/controller/`.

### Adapter weights

The trained adapters are on Hugging Face at
[`fazmain/parent-taught-refusal`](https://huggingface.co/fazmain/parent-taught-refusal). Download them into
`adapters/`, the default `ADAPTERS_DIR` (override with `REFUSAL_ADAPTERS_DIR`):

```
huggingface-cli download fazmain/parent-taught-refusal --local-dir adapters
```

| Folder | Paper row | b |
|---|---|---|
| `ptv_seed0` | PTV, seed 0 | +3.500 |
| `ptv_seed1` | PTV, seed 1 | +4.069 |
| `ptv_seed2` | PTV, seed 2 (also "no offset": the same adapter at b = 0) | +3.388 |
| `ptv_replay_doubled` | PTV, replay doubled | +4.319 |
| `ptv_construction_labels` | PTV, construction labels | +2.247 |

**Folder contents.**
- Each folder holds `adapter_config.json`, `adapter_model.safetensors`, `train_meta.json` and `verdict_gate.json`.
- `verdict_gate.json` holds the offset b, the verdict token ids and the calibration record. Its b equals the gate
  file of the same arm in `results/offsets/`.
- The repository root holds `hold_chunk.json` (open gripper) and `hold_chunk_closed.json` (closed gripper).

The adapters are released under the Qwen Research License (non-commercial).

**How b is applied.**
- The simulator harness adds b to the `[STOP]` logit at the verdict position: the second generated token, right after
  `[`. No other position or logit changes.
- Use the b stored with each adapter. Do not reuse one adapter's b for another.

### One simulated episode (smoke test)

A smoke run of the parent-taught verdict, seed 1. It runs one short episode (40 steps) per test and pool, on one
initial state. It checks the whole path (model load, adapter merge, offset, simulator, refusal classifier) in a few
minutes on a 24 GB GPU. Smoke outcomes are not results.

```
python experiments/5_simulation/eval_in_simulator.py --smoke --out results/smoke_ptv_seed1 \
    --model-spec verdict_lora:adapters/ptv_seed1 --verdict-gate adapters/ptv_seed1/verdict_gate.json \
    --pools data/pools.json
```

### One reporting-set arm (Table reporting)

This runs the 310 episodes of one arm on initial states 15 to 19. It is the command of the original runs, with the
public paths. It takes about 2.3 hours on one RTX A5000, running two arms at a time.

```
python experiments/5_simulation/eval_in_simulator.py --out results/rerun/ptv_seed1 \
    --model-spec verdict_lora:adapters/ptv_seed1 --verdict-gate adapters/ptv_seed1/verdict_gate.json \
    --pools data/pools.json --max-steps 300 --swap-step 30 \
    --gate-inits 15 16 17 18 19 --pair-inits 15 16 --one-init 15 --min-init 15
```

- **Output:** `results/rerun/ptv_seed1/rollouts.jsonl`, one row per episode, in the same schema as
  `results/reporting/ptv_seed1/rollouts.jsonl`.
- **Expected values:** never-trained refusal 0.94, benign twins 0.07, LIBERO success 0.82.
- **Variation:** simulator rendering and GPU kernels are not bit-exact across machines, so episode outcomes can differ
  slightly.
- **Other arms:** swap the adapter folder for `ptv_seed0`, `ptv_seed2`, `ptv_replay_doubled` or
  `ptv_construction_labels`.
- **Base NORA:** use `--models base` in place of `--model-spec ... --verdict-gate ...`. It runs 230 episodes; there
  is no paraphrase pool for the base model.
- **No offset:** a gate file with b = 0 on the `ptv_seed2` adapter (`results/offsets/ptv_seed2_no_offset.json`).

### Per experiment folder

**Times and outputs.**
- Times are the wall-clock times logged on the original runs, on an RTX A5000 24 GB unless noted. "Not logged"
  means that script's time was not recorded on its own.
- "Writes" names the working folder of a re-run under `results/`. The released copy of the same rows is named after
  the arrow.
- Every GPU script also has a `--smoke` mode (minutes) and, where noted, a `--self-test` that runs on a CPU (see
  "Self-tests" below).

#### `experiments/1_presence/` (Section 3.1: the parent's directions inside NORA)

| Command | Hardware, time | Writes -> released as | Expected |
|---|---|---|---|
| `python experiments/1_presence/text_direction_presence.py --models parent nora --n-train 128 --n-test 128` | GPU, not logged | `results.jsonl` rows `exp=poc_presence` -> `results/diagnosis/presence/generalist_nora.jsonl`, `libero_goal_nora.jsonl` | refusal-direction transfer AUROC 0.98 at layer 21, last prompt token |
| `python experiments/1_presence/verify_text_direction_presence.py --parent parent --other nora --lo 8 --hi 28` | CPU | assertion only | passes on the rows above |
| `python experiments/1_presence/parent_harm_judge.py` | GPU, not logged | the parent's yes/no and graded judgements | labels the robot commands |
| `python experiments/1_presence/robot_command_activations.py` | GPU, about 1.5 h with the probe and monitor | per-layer activations (not released) and transfer rows -> `results/diagnosis/presence/robot_commands.jsonl` | NORA harm-direction AUROC 0.81 at the last instruction token |
| `python experiments/1_presence/robot_direction_transfer.py` | CPU, from saved activations | `results/diagnosis/presence/robot_commands_analysis.jsonl` | parent 0.95 vs NORA 0.60 at layer 21, last prompt token |
| `python experiments/1_presence/harm_probe.py` | CPU, from saved activations | probe rows -> `results/diagnosis/probes/` | out-of-fold probe AUROC 0.84, catch 0.56 |

The raw activation files are not released (they are large), so the last two scripts need a GPU run of
`robot_command_activations.py` first.

#### `experiments/2_disconnection/` (Section 3.2: nothing connects the knowledge to the arm)

| Command | Hardware, time | Released as | Expected |
|---|---|---|---|
| `python experiments/2_disconnection/rollout_commands.py --episodes 5` | GPU + LIBERO, about 2 h (100 baseline episodes + 75 commands x 5) | `results/diagnosis/base_policy/rollouts.jsonl` | no episode holds still; every arm travels at least 0.41 m |
| `python experiments/2_disconnection/patch_task_words.py --episodes 5 --layers-sweep 4 10 16 22 28` | GPU + LIBERO, not logged | `results/diagnosis/patching/readout_patch_rollouts.jsonl` | task swaps in 5 of 5 at layers 4, 10, 16; 0 of 5 at 22, 28 |
| `python experiments/2_disconnection/patch_harm_words.py --layer 10 --episodes 5` | GPU + LIBERO, not logged | `results/diagnosis/patching/harm_patch_rollouts.jsonl` | task A success 1.00 -> 0.07 for harmful and unfamiliar benign alike |
| `python experiments/2_disconnection/conditional_steering.py --out results/c1` | GPU + LIBERO, not logged | `results/diagnosis/scripted_hold_steering/` | no steering setting refuses; scripted hold 0.76 of 45 harmful episodes |
| `python experiments/2_disconnection/monitor_halt_demo.py --per-tag 3 --episodes 3` | GPU + LIBERO, not logged | (demo; the latent monitor) | |

#### `experiments/3_refusal_examples/` (Section 4 and Appendix B)

| Command | Hardware, time | Released as | Expected |
|---|---|---|---|
| `python experiments/3_refusal_examples/build_free_install_data.py --out results/f1/data` | GPU + LIBERO, about 3 h with training | `data/pools.json`, `results/routes/free_lora_38/train_rows.jsonl` | 38 training harm commands, 24 never-trained |
| `python experiments/3_refusal_examples/train_install_lora.py --rows results/f1/data/train_install.jsonl --out results/f1/adapters/install` | GPU, about 8 to 12 min per install | adapter (`train_meta.json` in `results/routes/free_lora_38/adapter/`) | |
| `python experiments/3_refusal_examples/first_call_token_check.py --install results/f1/adapters/install --control results/f1/adapters/control --out results/f1` | GPU, minutes | `results/routes/free_lora_38/first_call_token_check.jsonl` | first-call hold 0.88 (free LoRA) and 0.96 (control) on never-trained commands |
| `python experiments/5_simulation/eval_in_simulator.py --out results/f2 --adapter-install results/f1/adapters/install --adapter-control results/f1/adapters/control` | GPU + LIBERO, about 2.3 h per 310-episode arm | `results/routes/free_lora_38/rollouts.jsonl` | Table routes: free LoRA 0.21 never-trained, control 0.08 |
| `python experiments/3_refusal_examples/build_parent_gated_data.py --stage all --out results/g1` | GPU + LIBERO, part of about 7 to 8 h with the simulator | `results/routes/parent_gated/`, `results/routes/free_lora_338/` | the 338-command free LoRA, 0.29 never-trained |
| `python experiments/3_refusal_examples/verify_parent_gated_judge.py --out results/g1` | GPU, minutes | `results/routes/parent_gated/first_call*.json` | judge threshold at grade 4 |
| `python experiments/3_refusal_examples/probe_gate_first_call.py --out results/g4` | GPU, about 10 min | `results/diagnosis/probes/probe_gate_first_call.json` | probe gate fails: catch 0.40, benign twins 0.23 |
| `python experiments/3_refusal_examples/single_model_judges.py --stage all --out results/g6` | GPU, not logged | `results/judges/` | rank-32 restore judge 0.83 / 0.13 / 0.00 |
| `python experiments/3_refusal_examples/text_classifier_gate.py --corpus <pairs.jsonl> --tag pairs --out results/g6c` | CPU | `results/judges/text_classifier_summary.json` | sentence-encoder gate 0.92 / 0.07 / 0.00 |
| `python experiments/3_refusal_examples/meanpool_probe_gate_build.py --g6 results/g6 --g8 results/g8 --f1-data results/f1/data` | GPU, not logged | `results/diagnosis/probes/mean_pooled_probe_gate.json` | 0.58 benign twins on the confirmation set |
| `confirmation_set_gates.py` | GPU, not logged | `results/judges/confirm_*` | every decider on the 50-sentence confirmation set |

The parent-gated simulator arms run through `eval_in_simulator.py` with
`--model-spec g1:results/g1/adapters/gated:parent_judge` (and `parent_read`, `random_dir`, `oracle`). See its
docstring.

#### `experiments/4_parent_taught_verdict/` (Section 5 and Appendix A, C)

Each round of the install search is one script. Every script runs in stages (`--stage`), on a GPU unless noted.

| Command | Hardware, time | Released as | Expected |
|---|---|---|---|
| `python experiments/4_parent_taught_verdict/build_training_rows.py --stage corpus\|teacher\|rows\|teacher_align --out results/g14` | GPU, about 30 min | `results/training/rows/` (4,213 rows, both label sets), `data/train_corpus.jsonl` | 1,066 generated commands, 92 parent labels differ |
| `python experiments/4_parent_taught_verdict/review_training_rows.py --rows <dir> --hold-chunk <hold_chunk.json>` | CPU | `results/training/rows/audit_lexical_shortcuts.txt` | zero blocking findings; bag-of-words baseline 0.67 |
| `python experiments/4_parent_taught_verdict/first_call_eval.py --out results/g14` | GPU, about 1 h training + 20 min scoring | `results/first_call/round1.jsonl` | round 1: no arm passes |
| `python experiments/4_parent_taught_verdict/train_verdict_alignment.py --stage restore --out results/g15`, then `--stage stage0` | GPU, about 1 h | `results/first_call/round2.jsonl` | round 2: the PTV recipe catches 0.44 at argmax |
| `python experiments/4_parent_taught_verdict/calibrate_verdict_offset.py --stage restore --out results/g16`, then `--stage stage0` | GPU, about 1.2 h | `results/first_call/round3.jsonl`, `results/offsets/round3_*` | the 10% offset rule; offline b_95 |
| `python experiments/4_parent_taught_verdict/fresh_confirmation.py --out results/g17` | GPU, about 50 min (+ about 1.7 h simulator) | `results/first_call/round4_fresh.jsonl`, `results/selection/ptv_seed0/` | round 4: the 5% rule, the fresh 87-command set |
| `python experiments/4_parent_taught_verdict/train_verdict_seeds.py --stage rows\|stage0\|fresh --out results/g18` | GPU, up to about 7 h with the simulator arms | `results/first_call/round5.jsonl`, `results/selection/ptv_seed2/`, `ptv_replay_doubled/` | seeds 1 and 2, replay doubled |
| `python experiments/4_parent_taught_verdict/train_verdict.py --stage check\|stage0\|fresh\|gates\|frames --out results/g19` | GPU, about 9 h with all six reporting arms | `results/first_call/round6.jsonl`, `results/offsets/*.json`, `results/reporting/` | Table reporting |

**Frames.** The first-call stages score each command on rendered LIBERO start frames, two per command.
- The frames are not released.
- `train_verdict.py --stage frames` (GPU + LIBERO) regenerates the round-6 frames.
- `build_free_install_data.py` renders the home frames of the earlier rounds.
- Both render fixed initial states with fixed seeds.

**Install times.** A single install trains in about 12 minutes (1,052 steps; replay doubled 1,474 steps, about
17 minutes).

#### `experiments/5_simulation/` (every simulator arm)

`eval_in_simulator.py` runs the 310-episode plan for any arm, about 2.3 hours per arm on one RTX A5000 with two arms
sharing the GPU. `--plan-only` prints the episode plan without a GPU, but it needs LIBERO for the task list. The
commands are above.

#### `experiments/6_libero_safety/` (Appendix H)

| Command | Hardware, time | Released as | Expected |
|---|---|---|---|
| `python experiments/6_libero_safety/libero_safety_eval.py --build-twins --bddl-root <LIBERO-Safety>/libero/libero/bddl_files/reasoning_safety` | CPU | the 15 benign-twin BDDL files | |
| `python experiments/6_libero_safety/libero_safety_eval.py --arm I4s1b95 --out results/ls1/I4s1b95 --episodes 10` (also `--arm I4s2b95`, `--arm base`) | one A40 48 GB, about 5 h for 900 episodes (three arms at once; 57 s per episode per arm) | `results/libero_safety/<arm>/rollouts.jsonl` | Table liberosafety: [STOP] on 1.00 of explicit harm, 0.00 of benign twins |
| `python experiments/6_libero_safety/verify_libero_safety.py --root results/ls1 --episodes 10` | CPU | assertion only | the run is complete and consistent |

`--arm` takes the script's arm ids: `I4s1b95` and `I4s2b95` are the parent-taught verdict at seeds 1 and 2, with
their b; `base` is base NORA. The scene hazard map `data/libero_safety/hazard_map.json` that `libero_safety_eval.py` reads is not in this release.

## Self-tests

**Run as shipped.** These three self-tests run on a CPU in a fresh clone, a few seconds each:

```
python experiments/3_refusal_examples/train_install_lora.py --self-test
python experiments/3_refusal_examples/build_free_install_data.py --self-test
python experiments/4_parent_taught_verdict/build_training_rows.py --self-test
```

**Need unreleased inputs.** The other 12 self-tests check the code against byte-identical copies of the original
raw outputs and rendered frames under `results/raw/`, which are not released. They stop at the first missing file.
[experiments/README.md](../experiments/README.md#self-tests-in-a-fresh-clone) lists what each one needs.

**Need a GPU run's output.** The `--self-check-cpu <dir>` modes run on a CPU but need a GPU run's output folder.

For a fresh clone, `scripts/verify_release.py` is the CPU check of record.
