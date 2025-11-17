import json
import os
import numpy as np
import torch
import time
import gc
from tqdm import tqdm
from transformers import DynamicCache
from icae.configs import get_config
from icae.data.data_utils import (
    create_icae_example,
    compute_accuracy,
    compute_bleu,
    compute_exact_match,
    truncate_cache,
)
from icae.models import ICAE
from icae.configs.templates import TemplateManager


def load_conversation_trajectories(path):
    """Load conversation trajectories from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def _run(model, trajs, device):
    bleu_scores, accuracy_scores = [], []
    bleu_scores_autoreg, accuracy_scores_autoreg = [], []
    exact_match_scores, exact_match_scores_autoreg = [], []
    compress_times_all = []
    gen_times_all = []
    
    # Track per-trajectory metrics
    trajectory_metrics = []
    
    tm = TemplateManager(model.tokenizer)
    for traj_num, traj in enumerate(tqdm(trajs, desc="Traj Inference")):
        print("="*10 + f"Start trajectory {traj_num}" + "="*10)
        compress_times_traj = []
        gen_times_traj = []
        compression_rates_traj = []
        bleu_scores_traj = []
        accuracy_scores_traj = []
        bleu_scores_autoreg_traj = []
        accuracy_scores_autoreg_traj = []
        exact_match_scores_traj = []
        exact_match_scores_autoreg_traj = []
        total_original_tokens = 0
        total_compressed_tokens = 0
        msgs = traj.get("messages", traj.get("message_sequence", []))
        messages_history = []
        accumulated_compressed_memory = None

        # Step 1: Handle system and first user-assistant pair
        for i in range(min(3, len(msgs))):
            msg = msgs[i]
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                messages_history.append({"role": "system", "content": content})
            elif role == "user":
                messages_history.append({"role": "user", "content": content})
            elif role == "assistant":
                messages_history.append({"role": "assistant", "content": content})

        # Convert initial history to token IDs
        conversation_tokens = model.tokenizer.apply_chat_template(
            messages_history,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        # cut think-think tokens from history
        content_len = len(model.tokenizer(content, truncation=False)['input_ids']) + 2
        k = len(tm.template_tokens['assistant_prefix']) - len(tm.template_tokens['user_prefix'])
        conversation_tokens = conversation_tokens[:-content_len-k] + conversation_tokens[-content_len:]

        # Step 2: Process remaining user-assistant pairs
        for i in range(3, len(msgs) - 1, 2):
            print(f"-"*5 + f"Starting turn {i}" + "-"*5)
            if i + 1 >= len(msgs):
                continue
            if len(conversation_tokens) > 32_768:
                print(f"Skipping trajectory {i} - conversation is {len(conversation_tokens)} tokens long")
                del accumulated_compressed_memory, conversation_tokens
                gc.collect()
                torch.cuda.empty_cache()
                break
                
            curr_msg = msgs[i]
            next_msg = msgs[i + 1]
            
            curr_role = curr_msg.get("role", "")
            if curr_role != "user":
                raise ValueError(f"Current role is {curr_role}, expected user")
            user_content = curr_msg.get("content", "")
            user_tokens = model.tokenizer(user_content, truncation=False)["input_ids"]
            total_original_tokens += len(user_tokens)
            
            next_role = next_msg.get("role", "")
            if next_role != "assistant":
                raise ValueError(f"Next role is {next_role}, expected assistant") 
            assistant_content = next_msg.get("content", "")
            assistant_tokens = model.tokenizer(assistant_content, truncation=False)["input_ids"]
            total_original_tokens += len(assistant_tokens)
            total_compressed_tokens += len(assistant_tokens)


            if model.model_args.do_compress and len(user_tokens) >= model.mem_size:
                # Compress user message and append it to the accumulated compressed memory
                start_time = time.time()
                user_tokens_compressed = model._compress(torch.LongTensor(user_tokens).unsqueeze(0).to(device))
                compress_duration = time.time() - start_time
                compress_times_traj.append(compress_duration)
                compress_times_all.append(compress_duration)
                accumulated_compressed_memory = torch.cat([accumulated_compressed_memory, user_tokens_compressed], dim=0) if accumulated_compressed_memory is not None else user_tokens_compressed
                compression_rate = len(user_tokens) / user_tokens_compressed.size(0)
            else:
                user_tokens_compressed = torch.LongTensor(user_tokens)
                compress_duration = 0
                compression_rate = 1.0
            total_compressed_tokens += user_tokens_compressed.size(0)
            compression_rates_traj.append(compression_rate)

            # Process assistant message
            assistant_tokens = tm.create_answer_with_suffix(assistant_tokens)

            # Create ICAE example
            example = create_icae_example(
                input_tokens=user_tokens,
                lm_target_tokens=assistant_tokens,
                task_type="swebench",
                model=model,
                text_tokens=conversation_tokens      # this is the entire sequence of messages with special tokens
            )

            prompt_answer_ids = example["prompt_answer_ids"]
            labels_tensor = example["labels"]
            prompt_len = (labels_tensor == -100).sum().item()
            
            pa_ids = prompt_answer_ids.squeeze(0).tolist()
            lbls = labels_tensor.tolist()
            prompt_len = sum(x == -100 for x in lbls)
            prompt_ids = pa_ids[:prompt_len]
            answer_ids = pa_ids[prompt_len:]
            
            # important cut of think tokens in historty
            k = len(tm.template_tokens['assistant_prefix']) - len(tm.template_tokens['user_prefix'])
            conversation_tokens = prompt_ids[:-k] + answer_ids  ### prompt_ids[:-k]

            prompt_answer_ids_tensor = torch.LongTensor(prompt_answer_ids).unsqueeze(0).to(device)
            labels_tensor_unsqueezed = torch.LongTensor(labels_tensor).unsqueeze(0).to(device)

            start_time = time.time()
            out = model(
                input_ids=None,
                prompt_answer_ids=prompt_answer_ids_tensor,
                labels=labels_tensor_unsqueezed,
                is_ae=example.get("is_ae", False),
                compressed_memory=accumulated_compressed_memory,
            )
            forward_pass_duration = time.time() - start_time
            gen_times_traj.append(forward_pass_duration)
            gen_times_all.append(forward_pass_duration)

            # Compute BLEU and Accuracy metrics from the forward pass logits
            logits = out["logits"].detach()
            pred_ids = torch.argmax(logits, dim=-1).squeeze(0)
            lbl_ids = labels_tensor_unsqueezed.squeeze(0).detach()

            answer_mask = lbl_ids != -100
            answer_positions = torch.nonzero(answer_mask, as_tuple=False).squeeze(-1)
            
            valid_positions = answer_positions[answer_positions > 0]
            # In teacher forcing, the hypothesis for a given token is based on the previous ground-truth token
            hyp_ids = pred_ids[valid_positions - 1]
            ref_ids = lbl_ids[valid_positions]

            bleu_score = compute_bleu(model.tokenizer, ref_ids, hyp_ids)
            accuracy_score = compute_accuracy(ref_ids, hyp_ids)
            reference_text_tf = model.tokenizer.decode(ref_ids.tolist(), skip_special_tokens=True).strip()
            prediction_text_tf = model.tokenizer.decode(hyp_ids.tolist(), skip_special_tokens=True).strip()
            exact_match_score = float(compute_exact_match(prediction_text_tf, reference_text_tf))
            bleu_scores.append(bleu_score)
            accuracy_scores.append(accuracy_score)
            exact_match_scores.append(exact_match_score)
            bleu_scores_traj.append(bleu_score)
            accuracy_scores_traj.append(accuracy_score)
            exact_match_scores_traj.append(exact_match_score)

            # Autoregressive generation for evaluation (non-teacher forcing)
            prompt_ids_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
            max_autoreg_tokens = max(len(answer_ids), 1)
            generated_ids = model.generate_autoregressive(
                input_ids=None,
                prompt_ids=prompt_ids_tensor,
                max_new_tokens=max_autoreg_tokens,
                stop_at_eos=True,
                compressed_memory=accumulated_compressed_memory,
            )

            reference_ids_for_metrics = answer_ids.copy()
            generated_ids_for_metrics = generated_ids.copy()

            if len(generated_ids_for_metrics) < len(reference_ids_for_metrics):
                generated_ids_for_metrics.extend([-1] * (len(reference_ids_for_metrics) - len(generated_ids_for_metrics)))
            elif len(generated_ids_for_metrics) > len(reference_ids_for_metrics):
                reference_ids_for_metrics = reference_ids_for_metrics + [-1] * (len(generated_ids_for_metrics) - len(reference_ids_for_metrics))

            ref_tensor_full = torch.tensor(reference_ids_for_metrics, dtype=torch.long)
            gen_tensor_full = torch.tensor(generated_ids_for_metrics, dtype=torch.long)

            if generated_ids:
                ref_tensor_bleu = torch.tensor(answer_ids, dtype=torch.long)
                gen_tensor_bleu = torch.tensor(generated_ids, dtype=torch.long)
                bleu_score_autoreg = compute_bleu(model.tokenizer, ref_tensor_bleu, gen_tensor_bleu)
            else:
                bleu_score_autoreg = 0.0

            accuracy_score_autoreg = compute_accuracy(ref_tensor_full, gen_tensor_full)
            reference_text_autoreg = model.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            generated_text_autoreg = model.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            exact_match_score_autoreg = float(compute_exact_match(generated_text_autoreg, reference_text_autoreg))

            bleu_scores_autoreg.append(bleu_score_autoreg)
            accuracy_scores_autoreg.append(accuracy_score_autoreg)
            exact_match_scores_autoreg.append(exact_match_score_autoreg)
            bleu_scores_autoreg_traj.append(bleu_score_autoreg)
            accuracy_scores_autoreg_traj.append(accuracy_score_autoreg)
            exact_match_scores_autoreg_traj.append(exact_match_score_autoreg)

            print(f"[AUTOREG] BLEU-1:\t {bleu_score_autoreg:.4f}, Accuracy:\t {accuracy_score_autoreg:.4f}, Exact Match:\t {exact_match_score_autoreg:.4f}, Lengths (ref/gen): {len(answer_ids)}/{len(generated_ids)}")

            print(f"[TIME] Compression:\t {compress_duration:.2f}s for {len(user_tokens)} tokens (x{compression_rate:.1f} compression)")
            print(f"[TIME] Forward pass:\t {forward_pass_duration:.2f}s for {len(ref_ids)} tokens ({len(ref_ids) / forward_pass_duration if forward_pass_duration > 0 else 0:.1f} tokens/s) with {prompt_len} tokens in prompt")
            print(f"[METRICS] BLEU-1:\t {bleu_score:.4f}, Accuracy:\t {accuracy_score:.4f}, Exact Match:\t {exact_match_score:.4f}")

            # If this is the last assistant response in the trajectory, calculate and log trajectory metrics
            if i + 3 >= len(msgs):
                mean_comp_traj = float(np.mean(compress_times_traj)) if compress_times_traj else 0.0
                mean_gen_traj = float(np.mean(gen_times_traj)) if gen_times_traj else 0.0
                mean_comp_rate = float(np.mean(compression_rates_traj)) if compression_rates_traj else 1.0
                mean_bleu_traj = float(np.mean(bleu_scores_traj)) if bleu_scores_traj else 0.0
                mean_accuracy_traj = float(np.mean(accuracy_scores_traj)) if accuracy_scores_traj else 0.0
                mean_bleu_autoreg_traj = float(np.mean(bleu_scores_autoreg_traj)) if bleu_scores_autoreg_traj else 0.0
                mean_accuracy_autoreg_traj = float(np.mean(accuracy_scores_autoreg_traj)) if accuracy_scores_autoreg_traj else 0.0
                mean_exact_match_traj = float(np.mean(exact_match_scores_traj)) if exact_match_scores_traj else 0.0
                mean_exact_match_autoreg_traj = float(np.mean(exact_match_scores_autoreg_traj)) if exact_match_scores_autoreg_traj else 0.0
                
                # Store trajectory-level metrics
                traj_metrics = {
                    "trajectory_id": traj_num,
                    "mean_bleu": mean_bleu_traj,
                    "mean_accuracy": mean_accuracy_traj,
                    "mean_bleu_autoregressive": mean_bleu_autoreg_traj,
                    "mean_accuracy_autoregressive": mean_accuracy_autoreg_traj,
                    "mean_exact_match": mean_exact_match_traj,
                    "mean_exact_match_autoregressive": mean_exact_match_autoreg_traj,
                    "mean_compression_time": mean_comp_traj,
                    "mean_forward_pass_time": mean_gen_traj,
                    "mean_compression_rate": mean_comp_rate,
                    "total_original_tokens": total_original_tokens,
                    "total_compressed_tokens": total_compressed_tokens,
                    "num_turns": len(bleu_scores_traj)
                }
                trajectory_metrics.append(traj_metrics)
                
                print(f"Trajectory summary — mean compression: {mean_comp_traj:.4f}s, mean forward pass: {mean_gen_traj:.4f}s")
                print(f"Mean compression rate of tool_outputs: x{mean_comp_rate:.1f}")
                print(f"Mean bleu (trajectory): {mean_bleu_traj:.4f}")
                print(f"Mean accuracy (trajectory): {mean_accuracy_traj:.4f}")
                print(f"Mean exact match (trajectory): {mean_exact_match_traj:.4f}")
                print(f"Mean exact match (trajectory, autoregressive): {mean_exact_match_autoreg_traj:.4f}")
                print(f"Total tokens — original: {total_original_tokens}, compressed: {total_compressed_tokens}")
                print(f"Tokens saved: {total_original_tokens - total_compressed_tokens} (x{total_original_tokens / total_compressed_tokens:.1f} total compression)")
                print(f"="*10 + f"End trajectory {traj_num}" + "="*10)
    return (
        bleu_scores,
        bleu_scores_autoreg,
        accuracy_scores,
        accuracy_scores_autoreg,
        exact_match_scores,
        exact_match_scores_autoreg,
        trajectory_metrics,
    )


def main():
    model_args, data_args, training_args, inference_args = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() and not inference_args.use_cpu else "cpu")
    model_args.use_position_identifiers = inference_args.use_position_identifiers
    training_args.use_cpu = inference_args.use_cpu
    training_args.train = True
    training_args.restore_from = inference_args.restore_from
    model = ICAE(model_args, training_args)
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model = model.to(device)
    model.eval()
    dataset_path = getattr(inference_args, "dataset_path", "icae/trajectories/openai/swe-smith/smith_val_openai.jsonl")
    trajectories = load_conversation_trajectories(dataset_path)
    print(f"Loaded {len(trajectories)} trajectories")
    trajectories = trajectories[259:] ### Last half is used for testing
    print(f"Loaded {len(trajectories)} trajectories")
    with torch.inference_mode():
        (
            bleu_scores,
            bleu_scores_autoreg,
            accuracy_scores,
            accuracy_scores_autoreg,
            exact_match_scores,
            exact_match_scores_autoreg,
            trajectory_metrics,
        ) = _run(model, trajectories, device)
    
    # Calculate overall metrics across all trajectories
    mean_bleu = float(np.mean(bleu_scores)) if bleu_scores else 0.0
    mean_accuracy = float(np.mean(accuracy_scores)) if accuracy_scores else 0.0
    mean_bleu_autoreg = float(np.mean(bleu_scores_autoreg)) if bleu_scores_autoreg else 0.0
    mean_accuracy_autoreg = float(np.mean(accuracy_scores_autoreg)) if accuracy_scores_autoreg else 0.0
    mean_exact_match = float(np.mean(exact_match_scores)) if exact_match_scores else 0.0
    mean_exact_match_autoreg = float(np.mean(exact_match_scores_autoreg)) if exact_match_scores_autoreg else 0.0
    
    # Calculate averages from trajectory-level metrics
    traj_bleu_scores = [traj["mean_bleu"] for traj in trajectory_metrics]
    traj_accuracy_scores = [traj["mean_accuracy"] for traj in trajectory_metrics]
    traj_bleu_scores_autoreg = [traj["mean_bleu_autoregressive"] for traj in trajectory_metrics]
    traj_accuracy_scores_autoreg = [traj["mean_accuracy_autoregressive"] for traj in trajectory_metrics]
    traj_exact_match_scores = [traj["mean_exact_match"] for traj in trajectory_metrics]
    traj_exact_match_scores_autoreg = [traj["mean_exact_match_autoregressive"] for traj in trajectory_metrics]
    traj_compression_times = [traj["mean_compression_time"] for traj in trajectory_metrics]
    traj_forward_pass_times = [traj["mean_forward_pass_time"] for traj in trajectory_metrics]
    traj_compression_rates = [traj["mean_compression_rate"] for traj in trajectory_metrics]
    
    avg_bleu_by_trajectory = float(np.mean(traj_bleu_scores))
    avg_accuracy_by_trajectory = float(np.mean(traj_accuracy_scores))
    avg_bleu_autoreg_by_trajectory = float(np.mean(traj_bleu_scores_autoreg)) if traj_bleu_scores_autoreg else 0.0
    avg_accuracy_autoreg_by_trajectory = float(np.mean(traj_accuracy_scores_autoreg)) if traj_accuracy_scores_autoreg else 0.0
    avg_exact_match_by_trajectory = float(np.mean(traj_exact_match_scores)) if traj_exact_match_scores else 0.0
    avg_exact_match_autoreg_by_trajectory = float(np.mean(traj_exact_match_scores_autoreg)) if traj_exact_match_scores_autoreg else 0.0
    avg_compression_time_by_trajectory = float(np.mean(traj_compression_times))
    avg_forward_pass_time_by_trajectory = float(np.mean(traj_forward_pass_times))
    avg_compression_rate_by_trajectory = float(np.mean(traj_compression_rates))
    
    total_original_tokens = sum(traj["total_original_tokens"] for traj in trajectory_metrics)
    total_compressed_tokens = sum(traj["total_compressed_tokens"] for traj in trajectory_metrics)
    total_turns = sum(traj["num_turns"] for traj in trajectory_metrics)
    
    # Print comprehensive results
    print("\n" + "="*80)
    print("FINAL METRICS SUMMARY")
    print("="*80)
    
    print(f"\n--- PER-TURN AVERAGES (averaged across all {len(bleu_scores)} turns) ---")
    print(f"Mean BLEU-1: {mean_bleu:.4f}")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Mean Exact Match: {mean_exact_match:.4f}")
    print(f"Mean BLEU-1 (autoregressive): {mean_bleu_autoreg:.4f}")
    print(f"Mean Accuracy (autoregressive): {mean_accuracy_autoreg:.4f}")
    print(f"Mean Exact Match (autoregressive): {mean_exact_match_autoreg:.4f}")
    print(f"Total turns processed: {len(bleu_scores)}")
    
    print(f"\n--- PER-TRAJECTORY AVERAGES (averaged across {len(trajectory_metrics)} trajectories) ---")
    print(f"Mean BLEU-1 by trajectory: {avg_bleu_by_trajectory:.4f}")
    print(f"Mean Accuracy by trajectory: {avg_accuracy_by_trajectory:.4f}")
    print(f"Mean Exact Match by trajectory: {avg_exact_match_by_trajectory:.4f}")
    print(f"Mean BLEU-1 by trajectory (autoregressive): {avg_bleu_autoreg_by_trajectory:.4f}")
    print(f"Mean Accuracy by trajectory (autoregressive): {avg_accuracy_autoreg_by_trajectory:.4f}")
    print(f"Mean Exact Match by trajectory (autoregressive): {avg_exact_match_autoreg_by_trajectory:.4f}")
    print(f"Mean Compression Time by trajectory: {avg_compression_time_by_trajectory:.4f}s")
    print(f"Mean Forward Pass Time by trajectory: {avg_forward_pass_time_by_trajectory:.4f}s")
    print(f"Mean Compression Rate by trajectory: x{avg_compression_rate_by_trajectory:.1f}")
    
    print(f"\n--- OVERALL STATISTICS ---")
    print(f"Total trajectories processed: {len(trajectory_metrics)}")
    print(f"Total turns processed: {total_turns}")
    print(f"Total original tokens: {total_original_tokens:,}")
    print(f"Total compressed tokens: {total_compressed_tokens:,}")
    if total_compressed_tokens > 0:
        print(f"Overall compression ratio: x{total_original_tokens / total_compressed_tokens:.1f}")
        print(f"Tokens saved: {total_original_tokens - total_compressed_tokens:,}")
    
    print("="*80)

    # Save metrics summary to file (similar to other inference scripts)
    metrics_summary = {
        "task": "trajectories",
        "model_type": model_args.model_type,
        "model_name_or_path": model_args.model_name_or_path,
        "restore_from": inference_args.restore_from,
        "dataset_path": dataset_path,
        "per_turn": {
            "mean_bleu": mean_bleu,
            "mean_accuracy": mean_accuracy,
            "mean_exact_match": mean_exact_match,
            "mean_bleu_autoregressive": mean_bleu_autoreg,
            "mean_accuracy_autoregressive": mean_accuracy_autoreg,
            "mean_exact_match_autoregressive": mean_exact_match_autoreg,
            "num_turns": len(bleu_scores),
        },
        "per_trajectory": {
            "mean_bleu": avg_bleu_by_trajectory,
            "mean_accuracy": avg_accuracy_by_trajectory,
            "mean_exact_match": avg_exact_match_by_trajectory,
            "mean_bleu_autoregressive": avg_bleu_autoreg_by_trajectory,
            "mean_accuracy_autoregressive": avg_accuracy_autoreg_by_trajectory,
            "mean_exact_match_autoregressive": avg_exact_match_autoreg_by_trajectory,
            "mean_compression_time_s": avg_compression_time_by_trajectory,
            "mean_forward_pass_time_s": avg_forward_pass_time_by_trajectory,
            "mean_compression_rate": avg_compression_rate_by_trajectory,
        },
        "overall": {
            "num_trajectories": len(trajectory_metrics),
            "num_turns": total_turns,
            "total_original_tokens": total_original_tokens,
            "total_compressed_tokens": total_compressed_tokens,
            "overall_compression_ratio": (total_original_tokens / total_compressed_tokens) if total_compressed_tokens > 0 else 0.0,
            "tokens_saved": total_original_tokens - total_compressed_tokens,
        },
    }

    output_filename = (
        f"icae/data/metrics/metrics_{metrics_summary['task']}_{metrics_summary['model_type']}.json"
    )
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, "a") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"\nSaved metrics to: {output_filename}")


if __name__ == "__main__":
    main()