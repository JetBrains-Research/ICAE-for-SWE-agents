from accelerate import Accelerator
from datasets.distributed import split_dataset_by_node
from datasets import Dataset
from icae.models import ICAE
from icae.models.model_utils import train_model
from icae.data.data_utils import DataCollatorForDynamicPadding
from icae.configs import get_config
import os
import torch
import wandb


def main():
    model_args, data_args, training_args, _ = get_config()


    print(model_args)
    print(data_args)
    
    os.environ["WANDB_PROJECT"] = "icae-pretraining"
    wandb.init(
        project=os.environ["WANDB_PROJECT"], 
        config={ **vars(model_args), **vars(data_args), **vars(training_args) },
        name=f"{model_args.model_name_or_path}-{data_args.dataset_repo}-{training_args.output_dir}"
    )
    
    if training_args.per_device_train_batch_size > 1:
        training_args.accelerator_config.dispatch_batches = False
        training_args.accelerator_config.dataloader_drop_last = True
        training_args.accelerator_config.dataloader_pin_memory = True
        training_args.accelerator_config.dataloader_persistent_workers = True
        training_args.accelerator_config.dataloader_prefetch_factor = 4
        
    # check model_args.mem_size and min_tokens_for_lm
    assert (training_args.fixed_mem_size & (training_args.fixed_mem_size - 1)) == 0, "training_args.fixed_mem_size must be a power of 2"    
    assert training_args.leave_tokens_for_lm <= training_args.min_tokens_for_lm, "leave_tokens_for_lm should be fewer than min_tokens_for_lm"

    print("Loading pre-tokenized dataset...")
    
    # Load pre-tokenized data created by prepare_data.py
    train_examples = torch.load(data_args.train_output_file)
    eval_examples = torch.load(data_args.eval_output_file)

    train_dataset = Dataset.from_list(train_examples)
    eval_dataset = Dataset.from_list(eval_examples)
    
    print(f"Loaded {len(train_dataset)} training examples and {len(eval_dataset)} evaluation examples.")
    print("Dataset loaded successfully")

    model = ICAE(model_args, training_args)

    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id

    print("Pre-tokenized dataset check:")
    print(train_dataset[0])

    #data_collator = DataCollatorForDynamicPadding(
    #    pad_token_id=model.tokenizer.pad_token_id
    #)
    data_collator = None

    print("Starting training...")
    train_model(model, train_dataset, eval_dataset, training_args, data_collator)


if __name__ == "__main__":
    main()
