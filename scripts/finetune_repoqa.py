import json
from icae.models import ICAE, SimpleLLM
from icae.models.model_utils import train_model
from icae.data.data_utils import (
    tokenize_qwen_llm_ft_repoqa,
    tokenize_qwen_icae_ft_repoqa,
    format_repoqa
)
from icae.configs import get_config
import os
import torch
import wandb
from datasets import Dataset


def main():
    model_args, data_args, training_args, _ = get_config()

    # For RepoQA fine-tuning, we compute loss on the entire answer.
    training_args.leave_tokens_for_lm = 0

    print(model_args)
    print(data_args)
    
    os.environ["WANDB_PROJECT"] = "context-condensation"
    os.environ["WANDB_RUN_GROUP"] = "ICAE-qwen"
    wandb.init(
        project=os.environ["WANDB_PROJECT"], 
        config={ **vars(model_args), **vars(data_args), **vars(training_args) },
        name=f"{model_args.model_name_or_path}-repoqa-{training_args.output_dir}"
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    
    # check model_args.mem_size and min_tokens_for_lm
    assert (training_args.fixed_mem_size & (training_args.fixed_mem_size - 1)) == 0, "training_args.fixed_mem_size must be a power of 2"    
    
    print("Loading RepoQA dataset from cache...")
    
    # Load from cache file
    cache_file_path = "/mnt/shared-fs/gelvan/icae/data/datasets/cache_ntoken_1024_v1.jsonl"
    
    tasks = []
    with open(cache_file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            data.pop('cache_id', None)  # Remove cache_id if present
            tasks.append(data)
    
    print(f"Loaded {len(tasks)} tasks from cache")
    
    # Convert to dataset format
    dataset_dict = {key: [task[key] for task in tasks] for key in tasks[0].keys()}
    dataset = Dataset.from_dict(dataset_dict)
    
    # Shuffle and split: 90% train, 10% test with seed=42
    dataset = dataset.shuffle(seed=42)
    split_idx = int(0.9 * len(dataset))
    train_dataset = dataset.select(range(split_idx))
    eval_dataset = dataset.select(range(split_idx, len(dataset)))
    
    print(f"Split: {len(train_dataset)} train, {len(eval_dataset)} test")
    
    # Trim datasets based on command line arguments
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(data_args.max_train_samples, len(train_dataset))))
    if data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(data_args.max_eval_samples, len(eval_dataset))))
    
    print(f"Using {len(train_dataset)} training examples and {len(eval_dataset)} evaluation examples")
    
    # Pre-process the dataset to a universal format.
    train_dataset = train_dataset.map(format_repoqa, batched=True, remove_columns=train_dataset.column_names, load_from_cache_file=False)
    eval_dataset = eval_dataset.map(format_repoqa, batched=True, remove_columns=eval_dataset.column_names, load_from_cache_file=False)

    # Only support Qwen, not Mistral
    if model_args.model_type == "icae":
        model = ICAE(model_args, training_args)
        if "qwen" in model_args.model_name_or_path.lower():
            train_dataset = train_dataset.map(tokenize_qwen_icae_ft_repoqa, fn_kwargs={"model": model}, load_from_cache_file=False)
            eval_dataset = eval_dataset.map(tokenize_qwen_icae_ft_repoqa, fn_kwargs={"model": model}, load_from_cache_file=False)
        else:
            raise ValueError("Only Qwen model is supported for RepoQA, not Mistral")
    elif model_args.model_type == "llm":
        model = SimpleLLM(model_args, training_args)
        tokenizer = model.tokenizer
        if "qwen" in model_args.model_name_or_path.lower():
            train_dataset = train_dataset.map(tokenize_qwen_llm_ft_repoqa, batched=True, fn_kwargs={"tokenizer": tokenizer, "max_length": training_args.model_max_length}, load_from_cache_file=False)
            eval_dataset = eval_dataset.map(tokenize_qwen_llm_ft_repoqa, batched=True, fn_kwargs={"tokenizer": tokenizer, "max_length": training_args.model_max_length}, load_from_cache_file=False)
        else:
            raise ValueError("Only Qwen model is supported for RepoQA, not Mistral")

    train_model(model, train_dataset, eval_dataset, training_args)


if __name__ == "__main__":
    main()

