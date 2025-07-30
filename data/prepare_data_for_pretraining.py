"""This is a separate script for preparing data for pretraining."""


import os
import json
import random
from tqdm import tqdm
import torch
from datasets import load_dataset

from icae.models import ICAE
from icae.configs.config import get_config
from icae.data.data_utils import create_icae_example


def get_long_text_list(dataset_repo, long_text_cache):
    if os.path.exists(long_text_cache):
        print(f"Loading {long_text_cache} from cache.")
        with open(long_text_cache, 'r', encoding='utf-8') as f:
            long_text_list = json.load(f)
        print(f"Loaded {len(long_text_list)} long texts from cache.")
        return long_text_list
    else:
        print(f"Cache file {long_text_cache} does not exist. Creating new cache.")
        

    print("Streaming dataset to find long texts...")
    dataset = load_dataset(dataset_repo, split="train", streaming=True)

    long_text_list = []
    for example in tqdm(dataset, desc="Processing dataset to find long texts"):
        if len(example["text"]) >= 4096:
            long_text_list.append(example["text"])
    
    print(f"Found {len(long_text_list)} long texts.")
    with open(long_text_cache, 'w', encoding='utf-8') as f:
        json.dump(long_text_list, f, ensure_ascii=False)

    return long_text_list


def prepare_and_save_data(model_args, data_args, training_args):
    if os.path.exists(data_args.train_output_file) and os.path.exists(data_args.eval_output_file):
        print("Prepared data files already exist. Skipping preparation.")
        return

    os.makedirs(os.path.dirname(data_args.train_output_file), exist_ok=True)

    # --- Use loaded arguments for ICAE model instantiation ---
    model_args.train = False
    # The model's max length should be adjusted for the pre-training task
    training_args.model_max_length = data_args.min_len * 2


    print("Loading model for tokenization...")
    model = ICAE(model_args, training_args)
    
    long_text_list = get_long_text_list(data_args.dataset_repo, data_args.long_text_cache)

    print("Processing long texts to create examples...")
    examples = []
    total_tokens_processed = 0

    for text in tqdm(long_text_list, "Processing long texts"):
        ids = model.tokenizer(text, truncation=False, padding=False)["input_ids"]

        # Add random variation to sequence length
        length_variation = random.randint(-data_args.min_len//10, data_args.min_len//10)
        seq_len = data_args.min_len + length_variation
        
        if len(ids) < seq_len * 2:
            continue
        
        last_start = len(ids) - seq_len * 2
        random_start = random.randint(0, last_start)
        
        input_tokens = ids[random_start : random_start + seq_len]
        lm_target_tokens = ids[random_start + seq_len : random_start + 2 * seq_len]

        # Decide whether this sample is for AE or LM based on lm_ratio
        is_ae = random.random() >= training_args.lm_ratio
        task_type = "ae" if is_ae else "lm"

        processed_example = create_icae_example(
            input_tokens, lm_target_tokens, task_type, model
        )
        examples.append(processed_example)

        total_tokens_processed += len(input_tokens)
        if lm_target_tokens:
            total_tokens_processed += len(lm_target_tokens)
        
        if total_tokens_processed >= data_args.token_num:
            print(f"Reached target token count of {data_args.token_num}.")
            break

    if len(examples) <= data_args.eval_size:
        raise ValueError("Not enough examples generated for train/eval split.")

    train_data = examples[data_args.eval_size:]
    eval_data = examples[:data_args.eval_size]

    print(f"Saving {len(train_data)} training examples to {data_args.train_output_file}")
    torch.save(train_data, data_args.train_output_file)
    print(f"Saving {len(eval_data)} evaluation examples to {data_args.eval_output_file}")
    torch.save(eval_data, data_args.eval_output_file)

    print("Data preparation finished.")

if __name__ == "__main__":
    model_args, data_args, training_args, _ = get_config()
    prepare_and_save_data(model_args, data_args, training_args)