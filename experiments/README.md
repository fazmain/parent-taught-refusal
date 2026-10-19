# Experiments

One folder per part of the paper, in paper order. Every script runs from the repository root:

```
pip install -e .                      # the `refusal` package and third_party/nora
python experiments/4_parent_taught_verdict/train_verdict.py --self-test
```

Scripts read inputs and write outputs through `refusal/paths.py`:

| Name | Default | Override |
|---|---|---|
| `DATA_DIR` | `data/` | `REFUSAL_DATA_DIR` |
| `RESULTS_DIR` | `results/` | `REFUSAL_RESULTS_DIR` |
| `ADAPTERS_DIR` | `adapters/` (a download of the Hugging Face repo `fazmain/parent-taught-refusal`) | `REFUSAL_ADAPTERS_DIR` |

Scripts share helpers across sections, for example every simulator arm goes through `5_simulation/eval_in_simulator.py`. Section folders start with a digit, so they are not Python packages. `refusal.paths.add_experiment_dirs()` puts every section folder on `sys.path`, and scripts import each other by module name (`import eval_in_simulator as F`). Module names are unique across folders.

## Scripts

GPU times are the logged wall-clock times on one 24 to 48 GB GPU (A5000 or A40). "n/l" means the time was not logged per script. An install trains in about 12 minutes. One 310-episode simulator run takes about 2.3 hours, with two runs sharing a GPU.

| New path | Old private name | Paper section | What it produces | GPU time |
|---|---|---|---|---|
| `1_presence/text_direction_presence.py` | `poc_presence.py` | 3.1 | Harm and refusal directions on D_text in both models, held-out AUROC, transfer, cosine; activation files | n/l |
| `1_presence/verify_text_direction_presence.py` | `verify_poc.py` | 3.1 | Asserts the parent's direction separates held-out D_text (gate) | CPU |
| `1_presence/text_refusal_check.py` | `poc_lineage_check.py` | 3.1 | Text-only refusal of NORA's backbone vs the parent (substring judge); `is_refusal` helper | n/l |
| `1_presence/build_robot_commands.py` | `build_d_robot.py` | 2, 3.1 | `data/d_robot.jsonl`, the paired robot command set | CPU |
| `1_presence/parent_harm_judge.py` | `e0c_parent_judge.py` | 3.1, 5.1, App. G | The parent's yes/no and graded harm judgements; the judge prompts `YESNO` / `GRADED` verbatim | n/l |
| `1_presence/robot_command_activations.py` | `e1_drobot.py` | 3.1 | Per-layer activations of both models on robot commands with scene frames; transfer AUROC | about 1.5 h with the probe and monitor |
| `1_presence/robot_direction_transfer.py` | `e1_drobot_transfer.py` | 3.1 | Directions fitted on robot commands, parent to VLA transfer, cosine (CPU, from saved activations) | CPU |
| `1_presence/harm_probe.py` | `e4_probe.py` | 3.1 | Held-out-category probe, novelty and lexical controls (CPU, from saved activations) | CPU |
| `2_disconnection/rollout_commands.py` | `rollout_prompts.py` | 2, 3.2 | Base NORA rollouts on command lists (baseline success, behaviour under harmful commands) | about 2 h (100 baseline episodes + 75 commands x 5) |
| `2_disconnection/patch_task_words.py` | `l7_patching_control.py` | 3.2 | Activation patching of instruction words (positive control: the task swaps) | n/l |
| `2_disconnection/patch_harm_words.py` | `l8_harm_patching.py` | 3.2 | Patching harmful words into a benign rollout | n/l |
| `2_disconnection/conditional_steering.py` | `c1_conditional_steering.py` | 2, 3.2, App. A | Conditional steering and the scripted hold chunk; episode classifier and simulator loop shared by later scripts | n/l |
| `2_disconnection/monitor_halt_demo.py` | `e4_monitor_demo.py` | 2, Limitations | The latent harm monitor halting the arm at run time; prompt-file reader shared by later scripts | n/l |
| `3_refusal_examples/build_free_install_data.py` | `f1_build_data.py` | 2, 4.1, App. A | Replay rows, instruction pools (`pools.json`), home frames and training rows of the free install | about 3 h with training |
| `3_refusal_examples/first_call_token_check.py` | `f1_token_check.py` | 4.1 | First-call hold / motion / invalid classification without the simulator | minutes |
| `3_refusal_examples/train_install_lora.py` | `f1_train_lora.py` | 4, 5, App. A | The LoRA trainer every install uses (rows, loss masking, representation matching) | about 8 to 12 min per install |
| `3_refusal_examples/build_parent_gated_data.py` | `g1_build_data.py` | 4.2, App. A-B | Teacher-labelled pools, parent-gated rows, the 338-command free-install rows | part of about 7 to 8 h with the simulator |
| `3_refusal_examples/verify_parent_gated_judge.py` | `verify_g1_gate.py` | 3.2, App. B | Fits and checks the parent-gated reference's condition offline; GO / NO-GO assertions | minutes |
| `3_refusal_examples/probe_gate_first_call.py` | `g4_probe_stage0.py` | App. B | A linear probe on NORA's own activations as the decider, scored at the first call | about 10 min |
| `3_refusal_examples/single_model_judges.py` | `g6_single_model_judges.py` | 5.4, App. B | Single-model harm judges inside NORA (including the rank-32 parent-weight restore) | n/l |
| `3_refusal_examples/text_classifier_gate.py` | `g6c_text_classifier.py` | App. B | A small sentence-encoder classifier as the decider | CPU |
| `3_refusal_examples/text_gate_wiring.py` | `g7_text_gate.py` | App. B (dependency) | Unit tests and run-time wiring check of the text-classifier gate; selection-set input reader | CPU |
| `3_refusal_examples/meanpool_probe_gate.py` | `g8_meanpool_probe.py` | App. B (dependency) | Mean-pooled instruction-span probe as the decider | n/l |
| `3_refusal_examples/meanpool_probe_gate_build.py` | `g8_gate_build.py` | App. B | Builds and checks the mean-pooled probe gate; scores it on the confirmation set | n/l |
| `3_refusal_examples/confirmation_set_gates.py` | `g11_confirm.py` | 5.3, App. B | Every candidate decider at its fixed threshold on the confirmation set | n/l |
| `4_parent_taught_verdict/build_training_rows.py` | `g14_build_rows.py` | 4.2, 5.1, App. C | The audited training rows (construction and parent labels) and the representation-matching targets | about 30 min |
| `4_parent_taught_verdict/review_training_rows.py` | `g14_review_rows.py` | App. C | Independent audit of the training rows (counts, leakage, lexical shortcuts) | CPU |
| `4_parent_taught_verdict/first_call_eval.py` | `g14_stage0.py` | 4.2, 5.1, 5.3, App. A | Round 1: four installs, scored at the first policy call; shared first-call machinery | about 1 h training + 20 min scoring |
| `4_parent_taught_verdict/train_verdict_alignment.py` | `g15_install.py` | 5.1, 5.3, App. A | Round 2: verdict + parent labels + matching (I4, the parent-taught verdict), I5, I6 | about 1 h |
| `4_parent_taught_verdict/calibrate_verdict_offset.py` | `g16_install.py` | 5.1, App. A | Round 3: the calibrated verdict offset b and the merged-weight installs I7, I8 | about 1.2 h |
| `4_parent_taught_verdict/fresh_confirmation.py` | `g17_fresh.py` | 5.1, 5.3, App. A | Round 4: the b95 rule, the fresh 87-command set, the verdict gate files | about 50 min (+ about 1.7 h simulator) |
| `4_parent_taught_verdict/train_verdict_seeds.py` | `g18_install.py` | 5.1, App. A | Round 5: seeds 1 and 2, replay doubled (I9), half learning rate (I10) | up to about 7 h with simulator runs |
| `4_parent_taught_verdict/train_verdict.py` | `g19_install.py` | 5.2, 5.3, App. A | Round 6: the reporting set; trains I11 (construction labels), writes every reporting arm's gate | about 9 h with all simulator arms |
| `5_simulation/eval_in_simulator.py` | `f2_eval_install.py` | 2, 3.2, 4, 5.2, App. A-B | The LIBERO-Goal simulator evaluation of every arm; the refusal classifier `classify_at` (5 cm rule) | about 2.3 h per 310-episode arm |
| `6_libero_safety/libero_safety_eval.py` | `ls1_ssr_eval.py` | App. H | The LIBERO-Safety `reasoning_safety` scenes and their benign twins, one arm per process | about 5 h for 900 episodes (three arms at once on 48 GB) |
| `6_libero_safety/verify_libero_safety.py` | `verify_ls1.py` | App. H (dependency) | Asserts the LIBERO-Safety run is complete and consistent | CPU |

The library `refusal/` was `src/` in the private repository (`src.models` is now `refusal.models`, and so on). `refusal/paths.py` is new.

## Names that stay in code and data

Result rows record the private script name in their `exp` field (for example `"exp": "g19_install"`), and arm names such as `I4s1b95` are data keys. They are kept so that the shipped rows, the scripts and the paper's Table (Appendix A) match. Use the table above to translate `exp` values.

| Code | Meaning |
|---|---|
| I0 to I11 | Install ids, as in the paper's install-search table (Appendix A) |
| I4, I4s1, I4s2 | The parent-taught verdict (PTV), seeds 0, 1, 2 |
| I9 / I10 / I11 | PTV with replay rows doubled / at half the learning rate / with construction labels instead of parent labels |
| `b95`, `argmax` | The calibrated verdict offset (5% of benign calibration calls say [STOP]) / no offset (b = 0) |
| G14 to G19 | Rounds 1 to 6 of the install search (Appendix A); file names such as `stage0_g14.jsonl` and `calib_g18.json` keep them |
| G1, `g1` model | The parent-gated reference (Appendix B) |
| G4, G6, G6-C, G7, G8, G10, G11 | Single-model deciders: probe gate, single-model judges, text classifier, text-gate wiring, mean-pooled probe, restore gate, confirmation-set scoring (Appendix B) |
| F1 / F2 | Free install: data and training / simulator evaluation (Section 4.1) |
| C1 | Conditional steering and the scripted hold chunk (Section 3.2) |
| E0c, E1, E4, L7, L8 | Parent judge, direction presence, probe and monitor, task-word and harm-word patching (Section 3) |
| stage 0 | The first policy call of an episode, scored without the simulator |
| selection set / reporting set | LIBERO-Goal initial states 10-14 / 15-19 |

Output folders keep the working names of the original runs (`results/g19`, `results/f1/data`, ...), so a full re-run chains stage to stage as before.

## Self-tests

Every `--self-test`, `--smoke` and `--self-check-cpu` path is kept. These run on a CPU in seconds:

```
python experiments/3_refusal_examples/train_install_lora.py --self-test
python experiments/3_refusal_examples/build_free_install_data.py --self-test
python experiments/3_refusal_examples/build_parent_gated_data.py --self-test --f1-data results/raw/pod7/f1_full/data
python experiments/3_refusal_examples/verify_parent_gated_judge.py --self-test
python experiments/3_refusal_examples/probe_gate_first_call.py --self-test
python experiments/3_refusal_examples/single_model_judges.py --self-test
python experiments/3_refusal_examples/text_gate_wiring.py --self-test
python experiments/4_parent_taught_verdict/build_training_rows.py --self-test
python experiments/4_parent_taught_verdict/first_call_eval.py --self-test
python experiments/4_parent_taught_verdict/train_verdict_alignment.py --self-test
python experiments/4_parent_taught_verdict/calibrate_verdict_offset.py --self-test
python experiments/4_parent_taught_verdict/fresh_confirmation.py --self-test
python experiments/4_parent_taught_verdict/train_verdict_seeds.py --self-test
python experiments/4_parent_taught_verdict/train_verdict.py --self-test
python experiments/6_libero_safety/libero_safety_eval.py --self-test --bddl-root <LIBERO-Safety>/libero/libero/bddl_files/reasoning_safety
```

They check the code against the raw outputs of the original runs: they hash training rows, compare file names and recompute stored decisions. They therefore read byte-identical raw files under `results/raw/` (the original file names, about 17 MB) and the frames the first-call evaluation is scored on. Single-number checks such as the single-model judges' score are computed on a randomly initialised stub model and change from run to run.

`--smoke` runs and `--self-check-cpu <OUT>` runs need a GPU run's outputs, a model download or the simulator; see the script docstrings.

### Self-tests in a fresh clone

`results/raw/` is **not shipped** with this repository: its files are byte-identical copies of the original run
outputs under their original names (about 17 MB, plus the rendered frames), and the released `results/` holds the
cleaned versions of the same rows instead. In a fresh clone, 3 of the 15 self-tests above run as they are
(`train_install_lora.py`, `build_free_install_data.py`, `build_training_rows.py`; a few seconds each on a CPU). The
other 12 stop at their first missing input with a `FileNotFoundError` or a "missing ..." message; none of them fails
on logic. The CPU check of record for a fresh clone is `python scripts/verify_release.py`, which recomputes every
number in the paper from the released `data/` and `results/` (see `docs/reproduce.md`).

The table lists what each of the 12 reads under `results/raw/` (paths relative to it; the scripts named import some
of these from each other, so a later round also needs the inputs of the earlier ones).

| Self-test | Inputs it needs that are not shipped |
|---|---|
| `3_refusal_examples/build_parent_gated_data.py` | `pod7/f1_full/data/pools.json` (pass `--f1-data results/raw/pod7/f1_full/data`), `pod6/c1/hold_chunk.json` |
| `3_refusal_examples/verify_parent_gated_judge.py` | `pod6/c1/hold_chunk.json` |
| `3_refusal_examples/probe_gate_first_call.py` | `pod8/g2/rollouts.jsonl`, `pod8/g1/gate.npz`, `pod8/g1/home_frames_eval/` (20 PNG), `pod7/f1_full/data/pools.json` |
| `3_refusal_examples/single_model_judges.py` | `pod7/f1_full/data/pools.json`, `pod8/g1/home_frames_eval/` |
| `3_refusal_examples/text_gate_wiring.py` | `pod8/g2/rollouts.jsonl`, `pod7/f1_full/data/pools.json` |
| `4_parent_taught_verdict/first_call_eval.py` | `pod8/g1/home_frames_eval/`, `pod7/f1_full/data/` (`pools.json`, `home_frames/`, 100 PNG), `pod6/c1/hold_chunk.json` |
| `4_parent_taught_verdict/train_verdict_alignment.py` | as `first_call_eval.py`, plus `pod14/g14_v4/stage0/stage0_g14.{json,jsonl}` |
| `4_parent_taught_verdict/calibrate_verdict_offset.py` | as `train_verdict_alignment.py`, plus `pod14/g14_v4/rows/train_verdict_parent.jsonl`, `pod15/g15/stage0/stage0_g15.{json,jsonl}` |
| `4_parent_taught_verdict/fresh_confirmation.py` | as `calibrate_verdict_offset.py`, plus `pod16/g16/stage0/calib_g16.json`, `pod14/g14_v4/rows/build_summary_g14.json` and the `train_meta.json` of the round 1-3 adapters (`pod14/g14_v4/adapters/{free_big,free_align,verdict_parent}`, `pod15/g15/adapters/{verdict_parent_align,free_align_w3,restore_verdict_parent}`, `pod16/g16/adapters/{restore32_verdict_parent_align,restore128_verdict_parent}`) |
| `4_parent_taught_verdict/train_verdict_seeds.py` | as `fresh_confirmation.py`, plus `pod14/g14_v4/rows/train_free_align.jsonl`, `pod17/g17/fresh/summary_g17.json` |
| `4_parent_taught_verdict/train_verdict.py` | as `train_verdict_seeds.py`, plus `pod14/g14_v4/rows/train_verdict.jsonl`, `pod18/g18/stage0/calib_g18.json`, `pod18/g18/fresh/summary_g18.json`, `pod18/g18/adapters/verdict_parent_align_{s1,s2,x2replay}/train_meta.json` |
| `6_libero_safety/libero_safety_eval.py` | `data/libero_safety/hazard_map.json` (not shipped) and `--bddl-root` pointing at a LIBERO-Safety checkout's `libero/libero/bddl_files/reasoning_safety` |

The scripts were not changed for the release; with these files in place all 15 self-tests pass and print the same
output as in the original runs.
