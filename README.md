# Copyright Traps Replication

This repository contains replication and extension scripts for experiments based on *Copyright Traps for Large Language Models*.

The experiments use TinyLlama-1.1B-Chat-v1.0, a synthetic continued-pretraining corpus, generated member/non-member trap sequences, and perplexity-based Membership Inference Attacks.

## Main components

- `scripts/make_large_paper_style_dataset.py` — creates the synthetic training corpus.
- `scripts/train_continued_pretraining.py` — performs continued pretraining and saves checkpoints.
- `scripts/run_mia_paper_style.py` — evaluates Membership Inference Attack performance.
- `scripts/diagnose_trap_learning.py` — analyzes member/non-member trap perplexity across epochs.
- `scripts/analyze_mia_by_nrep.py` — analyzes MIA performance by trap repetition count.
- `scripts/evaluate_utility_perplexity.py` — evaluates model utility using perplexity.
- `scripts/plot_memorization_by_epoch.py` — generates memorization-over-time plots.
- `scripts/plot_utility_by_epoch.py` — generates utility-over-time plots.

## Results

Key result files are stored in `results/`.

