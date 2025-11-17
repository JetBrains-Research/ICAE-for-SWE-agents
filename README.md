# In-Context AutoEncoder (ICAE)

This repository contains the official implementation for pre-training, fine-tuning, and running inference with the In-Context AutoEncoder (ICAE) model. It also supports standard Large Language Models (LLMs) for comparison. 

The original ICAE idea is presented in a paper [here](https://arxiv.org/abs/2307.06945) and the improvements using Positional Identifyiers described [here](https://arxiv.org/abs/2409.14364v3)

## Features

*   **Models**: Supports both `ICAE` and standard `SimpleLLM` architectures.
*   **Base Models**: Compatible with `Qwen` and `Mistral` series of models.
*   **Tasks**: Pre-training, fine-tuning, and inference on SQuAD, RepoQA, and trajectory datasets.
*   **Frameworks**: Built on top of Hugging Face's `transformers`, `datasets`, and `accelerate`.
*   **Experiment Tracking**: Integrated with Weights & Biases (`wandb`) for easy monitoring of experiments.

## Repository Structure
```
.
├── configs/              # Configuration files for experiments
├── data/                 # Data for training and evaluation
├── models/               # Model definitions (ICAE and SimpleLLM)
├── scripts/              # Scripts for pre-training, fine-tuning, and inference
├── README.md             # This file
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/.../icae.git
    cd icae
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

All scripts are configured using YAML files located in the `configs/` directory. Each script requires a `--config_path` argument.

### Pre-training

**1. Prepare pre-training data:**
```bash
CUDA_VISIBLE_DEVICES=X python -m icae.data.prepare_data_for_pretraining --config_path icae/configs/pretrain_config.yaml
```

**2. Run pre-training:**
```bash
CUDA_VISIBLE_DEVICES=X python -m icae.scripts.pretrain --config_path icae/configs/pretrain_config.yaml
```

### Fine-tuning

Fine-tune models on the SQuAD dataset:
```bash
CUDA_VISIBLE_DEVICES=X python -m icae.scripts.finetune_SQuAD --config_path icae/configs/finetune_config_icae.yaml
```

### Inference

Evaluate models on SQuAD with two task types:
- `ae` (autoencoding): Reconstruct input text
- `qa` (question answering): Answer questions based on context

```bash
CUDA_VISIBLE_DEVICES=X python -m icae.scripts.inference_SQuAD --config_path icae/configs/inference_config.yaml
```

Set `task: "ae"` or `task: "qa"` in the config file to specify the task type.

## Configuration

Key configuration parameters:
- `model_type`: `"icae"` or `"llm"`
- `do_compress`: Enable compression for ICAE (set to `True` for inference)
- `fixed_mem_size`: Size of memory tokens (must be a power of 2)
- `mean_compression_rate`: Expected compression ratio
- `use_position_identifiers`: Use positional identifiers for compression
- `task`: Inference task type (`"ae"`, `"qa"`, `"repoqa"`, or `"trajectories"`)

## Metrics

The inference scripts automatically compute and save metrics:

- **SQuAD**: BLEU-1, Exact Match (EM), F1 score
- **RepoQA**: BLEU-1, RepoQA similarity score, Pass@0.8 threshold
- **Trajectories**: BLEU-1, token-level accuracy, exact match (per turn and per trajectory), compression rates, timing statistics

Metrics are saved to `icae/data/metrics/` and predictions to `icae/data/predictions/`.

## Notes

*   Currently, only a batch size of 1 (`per_device_train_batch_size=1`) is supported.
*   RepoQA fine-tuning and inference currently only support Qwen models.
*   Trajectory fine-tuning uses a custom trainer that processes trajectories turn-by-turn with accumulated compressed memory.
*   All scripts support both ICAE and SimpleLLM models for comparison.
*   Checkpoints are saved in safetensors format and can be loaded for inference or continued training.