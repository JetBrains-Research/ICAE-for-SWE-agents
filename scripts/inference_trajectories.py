import json
import numpy as np
import torch
import time
import gc
from tqdm import tqdm
from transformers import DynamicCache
from icae.configs import get_config
from icae.data.data_utils import create_icae_example, compute_bleu, compute_accuracy, truncate_cache
from icae.models import ICAE
from icae.configs.templates import TemplateManager


def load_conversation_trajectories(path):
    """Load conversation trajectories from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def _run(model, trajs, device):
    bleu_scores, accuracy_scores = [], []
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
        total_original_tokens = 0
        total_compressed_tokens = 0
        msgs = traj.get("messages", traj.get("message_sequence", []))
        # cache = DynamicCache()
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
                ### TODO: for the future just do pure generation here without ICAE

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

            # print(f'prompt_answer_ids_tensor: {tm._safe_decode_with_mem_tokens(prompt_answer_ids_tensor[0])}')

            start_time = time.time()
            out = model(
                input_ids=None,
                prompt_answer_ids=prompt_answer_ids_tensor,
                labels=labels_tensor_unsqueezed,
                is_ae=example.get("is_ae", False),
                compressed_memory=accumulated_compressed_memory,
                # cache = cache
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
            bleu_scores.append(bleu_score)
            accuracy_scores.append(accuracy_score)
            bleu_scores_traj.append(bleu_score)
            accuracy_scores_traj.append(accuracy_score)

            print(f"[TIME] Compression:\t {compress_duration:.2f}s for {len(user_tokens)} tokens (x{compression_rate:.1f} compression)")
            print(f"[TIME] Forward pass:\t {forward_pass_duration:.2f}s for {len(ref_ids)} tokens ({len(ref_ids) / forward_pass_duration if forward_pass_duration > 0 else 0:.1f} tokens/s) with {prompt_len} tokens in prompt")
            print(f"[METRICS] BLEU-1:\t {bleu_score:.4f}, Accuracy:\t {accuracy_score:.4f}")

            # If this is the last assistant response in the trajectory, calculate and log trajectory metrics
            if i + 3 >= len(msgs):
                mean_comp_traj = float(np.mean(compress_times_traj)) if compress_times_traj else 0.0
                mean_gen_traj = float(np.mean(gen_times_traj)) if gen_times_traj else 0.0
                mean_comp_rate = float(np.mean(compression_rates_traj)) if compression_rates_traj else 1.0
                mean_bleu_traj = float(np.mean(bleu_scores_traj)) if bleu_scores_traj else 0.0
                mean_accuracy_traj = float(np.mean(accuracy_scores_traj)) if accuracy_scores_traj else 0.0
                
                # Store trajectory-level metrics
                traj_metrics = {
                    "trajectory_id": traj_num,
                    "mean_bleu": mean_bleu_traj,
                    "mean_accuracy": mean_accuracy_traj,
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
                print(f"Total tokens — original: {total_original_tokens}, compressed: {total_compressed_tokens}")
                print(f"Tokens saved: {total_original_tokens - total_compressed_tokens} (x{total_original_tokens / total_compressed_tokens:.1f} total compression)")
                print(f"="*10 + f"End trajectory {traj_num}" + "="*10)
    return bleu_scores, accuracy_scores, trajectory_metrics


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
    trajectories = trajectories[155:]  ### TODO: this is last half that is used for testing
    ### TODO: it is here just to be safe
    with torch.inference_mode():
        bleu_scores, accuracy_scores, trajectory_metrics = _run(model, trajectories, device)
    
    # Calculate overall metrics across all trajectories
    mean_bleu = float(np.mean(bleu_scores)) if bleu_scores else 0.0
    mean_accuracy = float(np.mean(accuracy_scores)) if accuracy_scores else 0.0
    
    # Calculate averages from trajectory-level metrics
    traj_bleu_scores = [traj["mean_bleu"] for traj in trajectory_metrics]
    traj_accuracy_scores = [traj["mean_accuracy"] for traj in trajectory_metrics]
    traj_compression_times = [traj["mean_compression_time"] for traj in trajectory_metrics]
    traj_forward_pass_times = [traj["mean_forward_pass_time"] for traj in trajectory_metrics]
    traj_compression_rates = [traj["mean_compression_rate"] for traj in trajectory_metrics]
    
    avg_bleu_by_trajectory = float(np.mean(traj_bleu_scores))
    avg_accuracy_by_trajectory = float(np.mean(traj_accuracy_scores))
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
    print(f"Total turns processed: {len(bleu_scores)}")
    
    print(f"\n--- PER-TRAJECTORY AVERAGES (averaged across {len(trajectory_metrics)} trajectories) ---")
    print(f"Mean BLEU-1 by trajectory: {avg_bleu_by_trajectory:.4f}")
    print(f"Mean Accuracy by trajectory: {avg_accuracy_by_trajectory:.4f}")
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


if __name__ == "__main__":
    main()