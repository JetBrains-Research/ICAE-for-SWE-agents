import json
import os

import numpy as np
import torch
from tqdm import tqdm

from icae.configs import get_config
from icae.configs.templates import TemplateManager
from icae.data.data_utils import (
    create_icae_example,
    load_and_process_squad,
    compute_bleu,
    compute_exact_match,
    compute_f1,
)
from icae.models import ICAE, SimpleLLM


# only for ICAE and Qwen
@torch.no_grad()
def run_autoencoding(model: ICAE, df, device):
    bleu_scores = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="AE Inference"):
        input_text = row["input"]

        # 1. Prepare ICAE example
        input_tokens = model.tokenizer(input_text, truncation=False, padding=False)["input_ids"]
        example = create_icae_example(
            input_tokens=input_tokens,
            lm_target_tokens=[],
            task_type="ae",
            model=model,
        )

        # 2. Tensorise and split prompt / answer
        input_ids_tensor = example["input_ids"].unsqueeze(0).to(device)
        prompt_answer_ids = example["prompt_answer_ids"]
        labels_tensor = example["labels"]
        prompt_len = (labels_tensor == -100).sum().item()

        prompt_ids = torch.LongTensor(prompt_answer_ids[:prompt_len]).unsqueeze(0).to(device)
        answer_ids = torch.LongTensor(prompt_answer_ids[prompt_len:]).unsqueeze(0).to(device)
        
        # 3. Autoregressive generation (compression performed internally)
        max_gen_len = answer_ids.shape[1]
        generated_token_ids = model.generate_autoregressive(
            input_ids_tensor,
            prompt_ids,
            max_new_tokens=max_gen_len,
        )

        # 4. Calculate BLEU score
        bleu_scores.append(
            compute_bleu(
                tokenizer=model.tokenizer,
                reference_ids=answer_ids.squeeze(0),
                hypothesis_ids=generated_token_ids
            )
        )

    return bleu_scores


@torch.no_grad()
def run_qa(model, df, device, data_args, model_type="icae"):
    em_scores, f1_scores = [], []
    predictions = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="QA Inference"):
        input_text = row["input"]
        question = row["prompt"]
        answer = row["answer"]

        if model_type == "icae":
            # 1. Tokenise context and question
            input_tokens = model.tokenizer(input_text, truncation=True, max_length=4096 - 50)["input_ids"]
            question_tokens = model.tokenizer(question, truncation=False)["input_ids"]

            # 2. Build sample
            example = create_icae_example(
                input_tokens=input_tokens,
                lm_target_tokens=[],  # no answer tokens during inference
                task_type="squad",
                model=model,
                text_tokens=question_tokens,
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
        else:  # SimpleLLM
            template_manager = TemplateManager(model.tokenizer)
            if "qwen" in model.model_args.model_name_or_path.lower():
                # 1. Build prompt with template and tokenize
                prompt_text = input_text + question
                prompt_ids = model.tokenizer(prompt_text, truncation=False)["input_ids"]
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
            elif "mistral" in model.model_args.model_name_or_path.lower():
                prompt_text = input_text + question
                full_prompt = (
                    model.tokenizer.bos_token + "[INST] " + prompt_text + " [/INST] "
                )
                tokenized_prompt = model.tokenizer(full_prompt, return_tensors="pt").to(device)
                outputs = model.llm.generate(
                    **tokenized_prompt,
                    max_new_tokens=data_args.max_out_length,
                    pad_token_id=model.tokenizer.eos_token_id,
                )
                generated_text = model.tokenizer.decode(
                    outputs[0][tokenized_prompt.input_ids.shape[1] :],
                    skip_special_tokens=True,
                )

        # 4. Metrics
        em = compute_exact_match(generated_text, answer)
        f1 = compute_f1(generated_text, answer)
        em_scores.append(em)
        f1_scores.append(f1)
        
        # Store prediction details
        predictions.append({
            "question": question,
            "predicted_answer": generated_text,
            "ground_truth": answer,
            "exact_match": em,
            "f1_score": f1
        })

    return em_scores, f1_scores, predictions


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

    if model_args.model_type == "icae":
        model = ICAE(model_args, training_args)
    else:
        model = SimpleLLM(model_args, training_args)

    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    model = model.to(device)
    model.eval()


    ### Inference
    if inference_args.task == "ae":
        df = load_and_process_squad(num_samples=inference_args.num_samples, include_answers=False)
        bleu_scores = run_autoencoding(model, df, device)
        mean_bleu = float(np.mean(bleu_scores)) if bleu_scores else 0.0

        print("\n--- AE Results ---")
        print(f"Mean BLEU-1: {mean_bleu:.4f} | Samples: {len(bleu_scores)}")

        metrics_summary = {
            "task": "ae",
            "model_type": model_args.model_type,
            "model_name_or_path": model_args.model_name_or_path,
            "restore_from": inference_args.restore_from,
            "mean_bleu": mean_bleu,
            "num_samples": inference_args.num_samples
        }
    else:
        df = load_and_process_squad(num_samples=inference_args.num_samples, include_answers=True)
        em_scores, f1_scores, predictions = run_qa(model, df, device, data_args, model_type=model_args.model_type)

        mean_em = float(np.mean(em_scores)) if em_scores else 0.0
        mean_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

        print("\n--- QA Results ---")
        print(
            f"Mean EM: {mean_em:.4f} ({mean_em*100:.2f}%) | "
            f"Mean F1: {mean_f1:.4f} ({mean_f1*100:.2f}%) | "
            f"Samples: {len(em_scores)}"
        )

        metrics_summary = {
            "task": "lm",
            "model_type": model_args.model_type,
            "model_name_or_path": model_args.model_name_or_path,
            "restore_from": inference_args.restore_from,
            "mean_em": mean_em,
            "mean_f1": mean_f1,
            "num_samples": inference_args.num_samples
        }
        
        # Save detailed predictions
        predictions_filename = f"icae/data/predictions/qa_predictions_{'_'.join(inference_args.restore_from.split('/'))}.json"
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