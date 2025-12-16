# ARC-AGI Baseline

## Overview

This repository contains a modified version of the winning baseline from the 2024 ARC Prize. It implements a fine-tuning pipeline for the Mistral-NeMo-Minitron 8B model (utilizing `unsloth` for 4-bit quantization) to solve Abstract Reasoning Corpus (ARC) tasks.

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

## Training Pipeline (`train/` folder)

The `train/` directory contains the complete training infrastructure with multiple versions and experimental scripts:

### Structure

* **`main/code/arc-trainer/`**: Core training scripts including multiple iterations (v1-v10) of the training pipeline, with versions exploring different models (Mistral, Qwen), staged training approaches, and various optimization strategies.
* **`misc/`**: Experimental scripts, notebooks, and alternative implementations for model loading, data processing, and fine-tuning experiments.
* **`setup/`**: Configuration and setup files for the training environment.
* **`shared/`**: Shared resources, utilities, and cached data used across different training versions.

### Key Components

* **`arc_loader.py`**: Custom data loader for ARC tasks.
* **`model_tools.py`**: Utility functions for model preparation and manipulation.
* **Training versions**: Multiple training script iterations (v1-v10) representing progressive improvements and experiments with different approaches, including staged training and multi-model comparisons.

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
