# In-Context AutoEncoder (ICAE)

This repository contains the official implementation for pre-training, fine-tuning, and running inference with the In-Context AutoEncoder (ICAE) model. It also supports standard Large Language Models (LLMs) for comparison. 

The original ICAE idea is presented in a paper [here](https://arxiv.org/abs/2307.06945) and the improvements using Positional Identifyiers described [here](https://arxiv.org/abs/2409.14364v3)

## Features

*   **Models**: Supports both `ICAE` and standard `SimpleLLM` architectures.
*   **Base Models**: Compatible with `Qwen` and `Mistral` series of models.
*   **Tasks**: Scripts for pre-training, fine-tuning on SQuAD, and inference.
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
    The `requirements.txt` file is automatically configured by wandb.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

All scripts are configured using YAML files located in the `configs/` directory. The main configuration files are `pretrain_config.yaml` and `finetune_config.yaml`.

### Pre-training

To prepare the data see, run the following command:
```bash
CUDA_VISIBLE_DEVICES=X python -m icae.data.prepare_data_for_pretraining --config_path configs/pretrain_config.yaml
```
To pre-train a model, run the following command:
```bash
CUDA_VISIBLE_DEVICES=X python -m icae.scripts.pretrain --config_path configs/pretrain_config.yaml
```

### Fine-tuning

To fine-tune a model on the SQuAD dataset, use the following command:
```bash
CUDA_VISIBLE_DEVICES=X python -m icae.scripts.finetune_SQuAD --config_path configs/finetune_config.yaml
```

### Inference

To run inference with a fine-tuned model on the SQuAD dataset, use the command below. This script can perform two tasks: `ae` (autoencoding) and `qa` (question answering), which can be specified in the config file.
```bash
CUDA_VISIBLE_DEVICES=X python -m icae.scripts.inference_SQuAD --config_path configs/finetune_config.yaml
```

## Notes

*   Currently, only a batch size of 1 (`bs=1`) is supported.