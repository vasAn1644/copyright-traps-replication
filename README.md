# Copyright Traps Replication

This repository contains the scripts and lightweight results for a replication/extension of the *Copyright Traps for Large Language Models* experiment.

The experiment uses TinyLlama-1.1B-Chat-v1.0, a synthetic continued-pretraining corpus, generated member/non-member trap sequences, and perplexity-based Membership Inference Attacks.

## Pipeline

1. Generate member traps using `src/scripts/gen_traps.py`.
2. Generate non-member traps using `src/scripts/gen_traps.py`.
3. Create the synthetic corpus with `scripts/make_large_paper_style_dataset.py`.
4. Inject member traps with `scripts/make_paper_style_dataset.py`.
5. Run continued pretraining with `scripts/train_continued_pretraining.py`.
6. Evaluate Membership Inference Attack with `scripts/run_mia_paper_style.py`.
7. Run memorization diagnostics with `scripts/diagnose_trap_learning.py`.
8. Evaluate utility with `scripts/eval_utility_perplexity.py`.
9. Generate plots with `scripts/plot_memorization_by_epoch.py` and `scripts/plot_utility_by_epoch.py`.

## Outputs

Lightweight CSV and PNG outputs are stored in `results/`.

Large generated datasets, trap `.pkl` files, and model checkpoints are intentionally excluded from this repository.
