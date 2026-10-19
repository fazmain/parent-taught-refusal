# Notice: third-party chat prompts (not covered by the repository's CC BY 4.0 data license)

The four files in this folder are the train/test splits of harmful and harmless chat instructions published with
Arditi et al., "Refusal in Language Models Is Mediated by a Single Direction" (NeurIPS 2024), repository
https://github.com/andyrdt/refusal_direction (files `dataset/splits/{harmful,harmless}_{train,test}.json`). They are
copied byte for byte as the project's prompt loader downloaded them from that repository.

| File | Rows | Upstream |
|---|---|---|
| `harmful_train.json` | 260 | harmful instructions pooled by Arditi et al. from AdvBench, HarmBench, JailbreakBench and MaliciousInstruct |
| `harmful_test.json` | 572 | as above; carries a `category` field |
| `harmless_train.json` | 18,793 | Alpaca (Taori et al., 2023) |
| `harmless_test.json` | 6,266 | Alpaca (Taori et al., 2023) |

The exact source of each harmful row is documented in the upstream repository, not here.

## Licence exception

- `harmless_train.json` and `harmless_test.json` derive from the Stanford Alpaca data, released under
  **CC BY-NC 4.0** (non-commercial). They are redistributed here under those terms, not under CC BY 4.0. Do not use
  them commercially.
- `harmful_train.json` and `harmful_test.json` are collections of prompts from the benchmarks listed above, each under
  its own upstream licence. They are redistributed here under their upstream terms, not under CC BY 4.0. Check the
  original repositories before reuse.

Every other file under `data/` and `results/` is released under CC BY 4.0 (see `data/README.md`).

## Content

The harmful files contain requests for dangerous, illegal or abusive content written by the upstream authors. They
are used here only to compute the parent model's refusal and harm directions (paper Section 3).

## Citation

Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., and Nanda, N. Refusal in language models is
mediated by a single direction. NeurIPS 2024.
Taori, R. et al. Stanford Alpaca: An instruction-following LLaMA model. 2023. https://github.com/tatsu-lab/stanford_alpaca
