# Copyright Traps Replication

This repository contains a lightweight replication and extension pipeline for the Copyright Traps for Large Language Models experiment.

The project uses `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, synthetic documents, generated member/non-member trap sequences, continued pretraining, and perplexity-based Membership Inference Attack evaluation.

## Main idea

The goal is to test whether a language model assigns lower perplexity to trap sequences that were present in its continued-pretraining data compared to trap sequences that were never seen during training.

## Latest pipeline: xlarge paper-style setup

The current experiment uses a larger synthetic corpus to reduce trap-density artifacts.

Earlier synthetic documents were relatively short. With longer traps and high repetition levels, inserted traps could become too large a part of the document. To make the setup more realistic, this version uses an xlarge synthetic paper-style corpus with much longer documents.

The latest setup uses:

* `seq_len = 50`
* `n_rep = 1, 10, 100`
* no `n_rep = 1000`
* xlarge synthetic documents
* continued pretraining on a balanced subset of trap and clean documents

## Minimal reproduction pipeline

### 1. Generate the xlarge synthetic corpus

```bash
python make_xlarge_paper_style_dataset.py
```

This creates the larger clean synthetic corpus.

### 2. Inject len50 traps without n_rep=1000

```bash
python make_len50_no1000_dataset.py
```

This creates the trap-injected dataset using repetition levels `1`, `10`, and `100`.

### 3. Create the balanced training subset

```bash
python make_len50_xlarge_balanced_train_dataset.py
```

This creates a smaller balanced training dataset used for continued pretraining.

### 4. Run continued pretraining

```bash
python train_continued_pretraining_len50_xlarge_no1000.py 1
python train_continued_pretraining_len50_xlarge_no1000.py 3
python train_continued_pretraining_len50_xlarge_no1000.py 5
```

These commands train models for 1, 3, and 5 epochs.

### 5. Diagnose trap memorization

```bash
python diagnose_trap_learning.py
```

This compares perplexity for member and non-member traps across the base and continued-pretrained models.

### 6. Run Membership Inference Attack evaluation

```bash
python run_mia_paper_style.py
```

This computes perplexity-based MIA scores for member and non-member trap sequences.

### 7. Evaluate model utility

```bash
python eval_utility_perplexity.py
```

This evaluates the models on a clean test dataset to check whether continued pretraining harms general utility.

## Core scripts

* `make_xlarge_paper_style_dataset.py` — creates the larger synthetic corpus.
* `make_len50_no1000_dataset.py` — creates the len50 trap-injected dataset without `n_rep=1000`.
* `make_len50_xlarge_balanced_train_dataset.py` — creates the balanced training subset.
* `train_continued_pretraining_len50_xlarge_no1000.py` — runs continued pretraining.
* `diagnose_trap_learning.py` — evaluates trap memorization using perplexity.
* `run_mia_paper_style.py` — runs perplexity-based MIA evaluation.
* `eval_utility_perplexity.py` — evaluates utility on a clean dataset.

## Notes

Large generated datasets, `.pkl` trap files, model checkpoints, logs, and heavy result files are intentionally excluded from this repository.

This repository is intended to contain only the minimal code needed to reproduce the latest xlarge paper-style experiment.
