`nora_utils.py` is copied verbatim from https://github.com/declare-lab/nora `experiments/libero/nora_utils.py` (MIT licence), 2026-09-16.
It holds the `Nora` inference class with the LIBERO action un-normalisation stats and the exact prompt/decode path used in NORA's LIBERO evaluation.
We import it unchanged; steering hooks attach to `Nora.model` from outside.
