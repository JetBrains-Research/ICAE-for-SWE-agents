import json
import numpy as np
import torch
import time
from tqdm import tqdm
from transformers import DynamicCache
from icae.configs import get_config
from icae.data.data_utils import create_icae_example, compute_bleu, compute_accuracy
from icae.models import ICAE
from icae.configs.templates import TemplateManager


def load_conversation_trajectories(path):
    """Load conversation trajectories from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def _run(model, trajs, device, max_len):
    bleu_scores, accuracy_scores = [], []
    compress_times_all = []
    gen_times_all = []
    tm = TemplateManager(model.tokenizer)
    for traj_num, traj in enumerate(tqdm(trajs, desc="Traj Inference")):
        print("="*10 + f"Start trajectory {traj_num}" + "="*10)
        compress_times_traj = []
        gen_times_traj = []
        compression_rates_traj = []
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

        # Step 2: Process remaining user-assistant pairs
        for i in range(3, len(msgs) - 1, 2):
            print(f"-"*5 + f"Starting turn {i}" + "-"*5)
            if i + 1 >= len(msgs):
                continue
                
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


            ### TODO: maybe decide differently? Now we just do not compress if the user message is shorter than the memory size
            if len(user_tokens) >= model.mem_size:
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

            prompt_ids = torch.LongTensor(prompt_answer_ids[:prompt_len]).unsqueeze(0).to(device)
            ### answer_ids = torch.LongTensor(prompt_answer_ids[prompt_len:]).unsqueeze(0)
            
            conversation_tokens = prompt_answer_ids.squeeze(0).tolist()           # this sequence is updated (appended) internally for the next turn

            true_answer_len = len(assistant_tokens)
            start_time = time.time()
            gen = model.generate_autoregressive(
                input_ids=None,
                prompt_ids=prompt_ids,
                max_new_tokens=true_answer_len,
                compressed_memory=accumulated_compressed_memory,
                # cache=cache
            )
            gen_duration = time.time() - start_time
            gen_times_traj.append(gen_duration)
            gen_times_all.append(gen_duration)
            ref_ids = torch.LongTensor(model.tokenizer(assistant_content, truncation=False)["input_ids"])
            hyp_ids = torch.LongTensor(gen)

            bleu_score = compute_bleu(model.tokenizer, ref_ids, hyp_ids)
            accuracy_score = compute_accuracy(ref_ids, hyp_ids)
            bleu_scores.append(bleu_score)
            accuracy_scores.append(accuracy_score)

            print(f"[TIME] Compression:\t {compress_duration:.2f}s for {len(user_tokens)} tokens (x{compression_rate:.1f} compression)")
            print(f"[TIME] Generation:\t {gen_duration:.2f}s for {len(gen)} tokens ({len(gen) / gen_duration:.1f} tokens/s) with {prompt_len} tokens in prompt")
            print(f"[METRICS] BLEU-1:\t {bleu_score:.4f}, Accuracy:\t {accuracy_score:.4f}")

            # If this is the last assistant response in the trajectory, print mean times for this trajectory
            if i + 3 >= len(msgs):
                mean_comp_traj = float(np.mean(compress_times_traj)) if compress_times_traj else 0.0
                mean_gen_traj = float(np.mean(gen_times_traj)) if gen_times_traj else 0.0
                mean_comp_rate = float(np.mean(compression_rates_traj)) if compression_rates_traj else 1.0
                print(f"Trajectory summary — mean compression: {mean_comp_traj:.4f}s, mean generation: {mean_gen_traj:.4f}s")
                print(f"Mean compression rate of tool_outputs: x{mean_comp_rate:.1f}")
                print(f"Mean bleu: {np.mean(bleu_scores):.4f}")
                print(f"Mean accuracy: {np.mean(accuracy_scores):.4f}")
                print(f"Total tokens — original: {total_original_tokens}, compressed: {total_compressed_tokens}")
                print(f"Tokens saved: {total_original_tokens - total_compressed_tokens} (x{total_original_tokens / total_compressed_tokens:.1f} total compression)")
                print(f"="*10 + f"End trajectory {traj_num}" + "="*10)
    return bleu_scores, accuracy_scores


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
    dataset_path = getattr(inference_args, "dataset_path", "icae/trajectories/openai/swe-smith/smith_train_openai.jsonl")
    trajectories = load_conversation_trajectories(dataset_path)
    ### TODO: it is here just to be safe
    with torch.inference_mode():
        bleu_scores, accuracy_scores = _run(model, trajectories, device, data_args.max_out_length)
    mean_bleu = float(np.mean(bleu_scores)) if bleu_scores else 0.0
    mean_accuracy = float(np.mean(accuracy_scores)) if accuracy_scores else 0.0
    print(f"Mean BLEU-1 for all trajectories: {mean_bleu:.4f} | Samples: {len(bleu_scores)}")
    print(f"Mean Accuracy for all trajectories: {mean_accuracy:.4f} | Samples: {len(accuracy_scores)}")


if __name__ == "__main__":
    main() 