# ARC-AGI Baseline

## Overview

This repository contains a modified version of the winning baseline from the 2024 ARC Prize. It implements a fine-tuning pipeline for the Mistral-NeMo-Minitron 8B model (utilizing `unsloth` for 4-bit quantization) to solve Abstract Reasoning Corpus (ARC) tasks.

## Modifications in this Fork

This version introduces specific optimizations to the original baseline:

* **Active Layer Control**: Added a `num_active_layers` variable to `common_stuff.py`. This allows for precise control over the training process by training only the initial $N$ layers of the model while keeping the remaining layers frozen.
* **Hyperparameter Tuning**: Adjusted LoRA (Low-Rank Adaptation) and general training settings to maximize performance within a strict 12-hour runtime limit.


## Codebase Structure

The notebook generates a core utility script, `model_runner.py`, which handles the primary model operations:

### 1. Model Preparation (`prepare_model`)
* Loads base models using `unsloth` (FastLanguageModel) or standard `transformers`.
* **Embedding Optimization**: Includes a `shrink_embeddings` function that resizes the tokenizer and model embeddings to contain only tokens present in the corpus, significantly reducing memory usage.
* **Layer Freezing**: Implements the `num_active_layers` logic to freeze specific transformer layers during fine-tuning.

### 2. Training (`training_run`)
* Wraps the `SFTTrainer` (or `UnslothTrainer`) to manage the training loop.
* Handles dataset formatting, gradient accumulation, and adapter saving.

### 3. Inference (`Decoder` class)
* **Test-Time Augmentation (TTA)**: The `Decoder` class handles predictions including transpositions and rotations of the input grids.
* **Scoring**: Utilities to validate predictions against known outputs.

## Dependencies

* `unsloth` (for optimized 4-bit training)
* `torch`
* `transformers`
* `peft`
* `trl`
* `datasets`
* `numpy`
