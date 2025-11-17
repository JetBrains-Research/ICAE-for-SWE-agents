import json
import os
import re
import time

import numpy as np
import torch
from tqdm import tqdm
from datasets import Dataset

from icae.configs import get_config
from icae.configs.templates import TemplateManager
from icae.data.data_utils import (
    create_icae_example,
    load_and_process_repoqa,
    compute_bleu,
    compute_repoqa_similarity,
    format_repoqa,
)
from icae.models import ICAE, SimpleLLM


def sanitize_output(model_output: str, lang: str = None) -> str:
    """Sanitize model output to extract code blocks if present."""
    model_output = model_output.strip()
    search_pattern = r"^```(?:\w+)?\s*\n(.*?)(?=^```)```"
    code_blocks = re.findall(search_pattern, model_output, re.DOTALL | re.MULTILINE)

    # If code blocks found, return first one
    if code_blocks:
        return code_blocks[0]
    
    # Otherwise return the raw output
    return model_output


@torch.no_grad()
def run_qa(model, df, device, data_args, model_type="icae"):
    """Run QA inference on RepoQA dataset.
    
    Computes both BLEU and RepoQA similarity metrics.
    """
    bleu_scores = []
    similarity_scores = []
    predictions = []
    inference_times = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="RepoQA Inference"):
        start_time = time.time()
        code_context = row["input"]  # code_context field
        prefix_text = row["prefix"]  # First instruction
        suffix_text = row["suffix"]  # description + second instruction
        answer = row["answer"]  # Target function name
        language = row.get("language", "python")

        if model_type == "icae":
            # 1. Tokenise code_context, prefix, and suffix
            context_tokens = model.tokenizer(code_context, truncation=True, max_length=16384 - 100)["input_ids"]
            prefix_tokens = model.tokenizer(prefix_text, truncation=False)["input_ids"]
            suffix_tokens = model.tokenizer(suffix_text, truncation=False)["input_ids"]

            # 2. Build sample (order: prefix + code_context + suffix)
            example = create_icae_example(
                input_tokens=context_tokens,
                lm_target_tokens=[],  # no answer tokens during inference
                task_type="repoqa",
                model=model,
                text_tokens=(prefix_tokens, suffix_tokens),
            )

            input_ids_tensor = example["input_ids"].unsqueeze(0).to(device)
            prompt_ids = example["prompt_answer_ids"].unsqueeze(0).to(device)

            # 3. Autoregressive generation from the prompt (compression inside the model)
            generate_tokens = model.generate_autoregressive(
                input_ids_tensor,
                prompt_ids,
                max_new_tokens=data_args.max_out_length,
            )
            generated_text = model.tokenizer.decode(
                generate_tokens, skip_special_tokens=True
            )
        else:  # SimpleLLM (only Qwen supported)
            template_manager = TemplateManager(model.tokenizer)
            if "qwen" in model.model_args.model_name_or_path.lower():
                # 1. Build prompt with template and tokenize (order: prefix + code_context + suffix)
                full_prompt = prefix_text + code_context + suffix_text
                prompt_ids = model.tokenizer(full_prompt, truncation=False)["input_ids"]
                prompt_ids = template_manager._apply_chat_template(prompt_ids)
                prompt_ids = torch.LongTensor(prompt_ids).unsqueeze(0).to(device)

                # 2. Generate
                outputs = model.llm.generate(
                    input_ids=prompt_ids,
                    attention_mask=torch.ones_like(prompt_ids),
                    max_new_tokens=data_args.max_out_length,
                    pad_token_id=model.tokenizer.eos_token_id,
                )
                generated_text = model.tokenizer.decode(
                    outputs[0][prompt_ids.shape[1] :],
                    skip_special_tokens=True,
                )
            else:
                raise ValueError("Only Qwen model is supported for RepoQA, not Mistral")

        # this is from RepoQA repo
        sanitized_output = sanitize_output(generated_text, language)

        # 4. Metrics
        # BLEU score
        answer_tokens = model.tokenizer(answer, add_special_tokens=False)["input_ids"]
        generated_tokens = model.tokenizer(sanitized_output, add_special_tokens=False)["input_ids"]
        bleu = compute_bleu(
            tokenizer=model.tokenizer,
            reference_ids=torch.LongTensor(answer_tokens),
            hypothesis_ids=torch.LongTensor(generated_tokens)
        )
        bleu_scores.append(bleu)
        
        # RepoQA similarity score
        similarity = compute_repoqa_similarity(sanitized_output, answer)
        similarity_scores.append(similarity)
        
        # Calculate inference time for this sample
        end_time = time.time()
        inference_time = end_time - start_time
        inference_times.append(inference_time)
        
        # Store prediction details
        predictions.append({
            "language": language,
            "repo": row.get("repo", "unknown"),
            "position_ratio": row.get("position_ratio", 0.0),
            "predicted_answer": generated_text,
            "sanitized_answer": sanitized_output,
            "ground_truth": answer,
            "bleu_score": bleu,
            "similarity_score": similarity,
            "passes_threshold": similarity >= 0.8,  # Threshold of 0.8
            "inference_time_seconds": inference_time
        })

    return bleu_scores, similarity_scores, predictions, inference_times


# -----------------------------------------------------------------------------
# Main entry-point
# -----------------------------------------------------------------------------

def main():
    model_args, data_args, training_args, inference_args = get_config()
    # ------------------------------------------------------------------
    # Setup model + tokenizer
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() and not inference_args.use_cpu else "cpu")

    # Manually attach custom flags expected by ICAE
    model_args.use_position_identifiers = inference_args.use_position_identifiers
    training_args.use_cpu = inference_args.use_cpu
    training_args.train = True  # ensure decoder initialisation
    training_args.restore_from = inference_args.restore_from

    # Only support for Qwen
    if "qwen" not in model_args.model_name_or_path.lower():
        raise ValueError("Only Qwen model is supported for RepoQA")

    if model_args.model_type == "icae":
        model = ICAE(model_args, training_args)
    else:
        model = SimpleLLM(model_args, training_args)

    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    model = model.to(device)
    model.eval()

    ### Inference
    cache_file_path = "/mnt/shared-fs/gelvan/icae/data/datasets/cache_ntoken_1024_v1.jsonl"
    
    print("Loading RepoQA dataset from cache...")
    
    # Load from cache file (same as finetune_repoqa.py)
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
    
    # Apply same shuffle and split as training: 90% train, 10% test with seed=42
    dataset = dataset.shuffle(seed=42)
    split_idx = int(0.9 * len(dataset))
    test_dataset = dataset.select(range(split_idx, len(dataset)))
    
    print(f"Using test split: {len(test_dataset)} samples (10% of total)")
    
    # Apply num_samples limit if specified
    if inference_args.num_samples is not None:
        test_dataset = test_dataset.select(range(min(inference_args.num_samples, len(test_dataset))))
        print(f"Limited to {len(test_dataset)} samples")
    
    # Format the dataset
    test_dataset = test_dataset.map(format_repoqa, batched=True, remove_columns=test_dataset.column_names, load_from_cache_file=False)
    
    # Convert to pandas DataFrame for compatibility with run_qa function
    df = test_dataset.to_pandas()
    
    print(f"Running inference on {len(df)} RepoQA test samples...")
    
    # Track total time
    total_start_time = time.time()
    
    bleu_scores, similarity_scores, predictions, inference_times = run_qa(
        model, df, device, data_args, model_type=model_args.model_type
    )
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    mean_bleu = float(np.mean(bleu_scores)) if bleu_scores else 0.0
    mean_similarity = float(np.mean(similarity_scores)) if similarity_scores else 0.0
    pass_at_threshold = float(np.mean([p["passes_threshold"] for p in predictions]))
    mean_inference_time = float(np.mean(inference_times)) if inference_times else 0.0
    median_inference_time = float(np.median(inference_times)) if inference_times else 0.0
    total_inference_time = float(np.sum(inference_times)) if inference_times else 0.0

    print("\n--- RepoQA Results ---")
    print(f"Mean BLEU-1: {mean_bleu:.4f}")
    print(f"Mean Similarity: {mean_similarity:.4f}")
    print(f"Pass@0.8 (threshold): {pass_at_threshold:.4f} ({pass_at_threshold*100:.2f}%)")
    print(f"Samples: {len(bleu_scores)}")
    print("\n--- Timing Statistics ---")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Mean time per sample: {mean_inference_time:.2f}s")
    print(f"Median time per sample: {median_inference_time:.2f}s")
    print(f"Throughput: {len(bleu_scores)/total_time:.2f} samples/second")

    metrics_summary = {
        "task": "repoqa",
        "model_type": model_args.model_type,
        "model_name_or_path": model_args.model_name_or_path,
        "restore_from": inference_args.restore_from,
        "mean_bleu": mean_bleu,
        "mean_similarity": mean_similarity,
        "pass_at_threshold_0.8": pass_at_threshold,
        "num_samples": inference_args.num_samples if inference_args.num_samples else len(df),
        "total_time_seconds": total_time,
        "mean_time_per_sample_seconds": mean_inference_time,
        "median_time_per_sample_seconds": median_inference_time,
        "throughput_samples_per_second": len(bleu_scores) / total_time if total_time > 0 else 0.0
    }
    
    # Save detailed predictions
    predictions_filename = f"icae/data/predictions/repoqa_predictions_{'_'.join(inference_args.restore_from.split('/'))}.json"
    os.makedirs(os.path.dirname(predictions_filename), exist_ok=True)
    with open(predictions_filename, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"\nSaved predictions to: {predictions_filename}")
    
    output_filename = (
        f"icae/data/metrics/metrics_{metrics_summary['task']}_{metrics_summary['model_type']}.json"
    )
    with open(output_filename, "a") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"\nSaved metrics to: {output_filename}")


if __name__ == "__main__":
    main()

