from datasets import load_dataset
from icae.models import ICAE, SimpleLLM
from icae.models.model_utils import train_model
from icae.data.data_utils import (
    tokenize_qwen_llm_ft,
    tokenize_mistral_llm_ft,
    tokenize_qwen_icae_ft,
    tokenize_mistral_icae_ft,
    format_squad
)
from icae.configs import get_config
import os
import torch
import wandb


def main():
    model_args, data_args, training_args, _ = get_config()

    # For SQuAD fine-tuning, we compute loss on the entire answer.
    training_args.leave_tokens_for_lm = 0

    print(model_args)
    print(data_args)
    
    os.environ["WANDB_PROJECT"] = "context-condensation"
    os.environ["WANDB_RUN_GROUP"] = "ICAE-qwen"
    wandb.init(
        project=os.environ["WANDB_PROJECT"], 
        config={ **vars(model_args), **vars(data_args), **vars(training_args) },
        name=f"{model_args.model_name_or_path}-{data_args.dataset_repo}-{training_args.output_dir}"
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}  # manually add this argument in the code

    
    # check model_args.mem_size and min_tokens_for_lm
    assert (training_args.fixed_mem_size & (training_args.fixed_mem_size - 1)) == 0, "training_args.fixed_mem_size must be a power of 2"    
    
    print("Loading dataset...")

    dataset = load_dataset("rajpurkar/squad")
    
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    # Shuffle datasets
    train_dataset = train_dataset.shuffle(seed=42)
    eval_dataset = eval_dataset.shuffle(seed=42)
    
    # Trim datasets based on command line arguments
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(data_args.max_train_samples, len(train_dataset))))
    if data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(data_args.max_eval_samples, len(eval_dataset))))
    
    print(f"Using {len(train_dataset)} training examples and {len(eval_dataset)} evaluation examples")
    
    # Pre-process the dataset to a universal format.
    train_dataset = train_dataset.map(format_squad, batched=True, remove_columns=train_dataset.column_names, load_from_cache_file=False)
    eval_dataset = eval_dataset.map(format_squad, batched=True, remove_columns=eval_dataset.column_names, load_from_cache_file=False)

    print('train_dataset[0]: ', train_dataset[0])
    print('eval_dataset[0]: ', eval_dataset[0])

    if model_args.model_type == "icae":
        model = ICAE(model_args, training_args)
        if "qwen" in model_args.model_name_or_path.lower():
            train_dataset = train_dataset.map(tokenize_qwen_icae_ft, fn_kwargs={"model": model}, load_from_cache_file=False)
            eval_dataset = eval_dataset.map(tokenize_qwen_icae_ft, fn_kwargs={"model": model}, load_from_cache_file=False)
        elif "mistral" in model_args.model_name_or_path.lower():
            train_dataset = train_dataset.map(tokenize_mistral_icae_ft, fn_kwargs={"model": model}, batched=True, load_from_cache_file=False)
            eval_dataset = eval_dataset.map(tokenize_mistral_icae_ft, fn_kwargs={"model": model}, batched=True, load_from_cache_file=False)
    elif model_args.model_type == "llm":
        model = SimpleLLM(model_args, training_args)
        tokenizer = model.tokenizer
        if "qwen" in model_args.model_name_or_path.lower():
            train_dataset = train_dataset.map(tokenize_qwen_llm_ft, batched=True, fn_kwargs={"tokenizer": tokenizer, "max_length": training_args.model_max_length}, load_from_cache_file=False)
            eval_dataset = eval_dataset.map(tokenize_qwen_llm_ft, batched=True, fn_kwargs={"tokenizer": tokenizer, "max_length": training_args.model_max_length}, load_from_cache_file=False)
        elif "mistral" in model_args.model_name_or_path.lower():
            train_dataset = train_dataset.map(tokenize_mistral_llm_ft, batched=True, fn_kwargs={"tokenizer": tokenizer, "max_length": training_args.model_max_length}, load_from_cache_file=False)
            eval_dataset = eval_dataset.map(tokenize_mistral_llm_ft, batched=True, fn_kwargs={"tokenizer": tokenizer, "max_length": training_args.model_max_length}, load_from_cache_file=False)

    print('train_dataset ready [0]: ', train_dataset[0])
    print('eval_dataset ready [0]: ', eval_dataset[0])

    train_model(model, train_dataset, eval_dataset, training_args)


if __name__ == "__main__":
    main()