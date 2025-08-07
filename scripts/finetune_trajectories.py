import json
import os
from torch.utils.data import Dataset
import torch
import gc
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import wandb
from transformers import DynamicCache


from icae.configs import get_config
from icae.models import ICAE
from icae.models.model_utils import ICAETrainer, print_loaded_layers
from icae.data.data_utils import create_icae_example, compute_bleu, compute_accuracy
from icae.configs.templates import TemplateManager
from icae.scripts.inference_trajectories import load_conversation_trajectories


class TrajectoryDataset(Dataset):
    """A simple ``torch.utils.data.Dataset`` wrapper around a list of trajectories."""

    def __init__(self, trajectories: List[Dict[str, Any]]):
        self.trajs = trajectories

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        result = {"input_ids": self.trajs[idx]}
        return result


class TrajectoryCollator:
    """Simply aggregates a list of trajectory dicts into a batch."""

    def __call__(self, examples: List[Dict[str, Any]]):
        return {"trajectories": [ex["input_ids"] for ex in examples]}


# -----------------------------------------------------------------------------
# Custom Trainer for trajectory-level training
# -----------------------------------------------------------------------------

class ICAETrajectoryTrainer(ICAETrainer):
    """
    Custom trainer that:
      • iterates over trajectories manually,
      • does one optimiser step per assistant turn,
      • keeps `accumulated_compressed_memory` across turns.
    """

    # ---------- OUR OWN OUTER LOOP ----------
    def train(
        self,
        resume_from_checkpoint = None,
        trial = None,
        ignore_keys_for_eval = None,
        **kwargs,
    ):
        args = self.args
        self.is_in_train = True
        self.model.train()

        # Make sure optimiser/scheduler are initialised
        if self.optimizer is None:
            self.create_optimizer()
        if self.lr_scheduler is None:
            self.create_scheduler(num_training_steps=args.num_training_steps)

        global_step = 0
        for epoch in range(int(args.num_train_epochs)):
            for i, traj in enumerate(self.train_dataset):
                print(f"Now in epoch #{epoch} -- trajectory #{i} of #{len(self.train_dataset)}")
                msgs = traj['input_ids']['messages']

                # --- build initial history ---
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in msgs[:3]                         # system + first U-A
                ]
                conversation_tokens = self.model.tokenizer.apply_chat_template(
                    history, tokenize=True, add_generation_prompt=False, enable_thinking=False
                )
                accumulated_compressed_memory = None
                template_mgr = TemplateManager(self.model.tokenizer)

                # --- iterate over remaining user–assistant pairs ---
                for idx in range(3, len(msgs) - 1, 2):
                    print(f"Processing turn #{idx} of #{len(msgs)}", end="\t")
                    user_msg, asst_msg = msgs[idx], msgs[idx + 1]
                    if user_msg["role"] != "user" or asst_msg["role"] != "assistant":
                        continue

                    # 1. tokenise assistant reply
                    asst_tokens = self.model.tokenizer(asst_msg["content"], truncation=False)["input_ids"]
                    asst_tokens = template_mgr.create_answer_with_suffix(asst_tokens)
                    ### TODO: this is super hacky but needed for now
                    print(f"asst_tokens: {len(asst_tokens)}, conversation_tokens: {len(conversation_tokens)}")
                    if len(asst_tokens) > 700 or len(conversation_tokens) > 14_000: #or i == 31 or i == 33 or i == 35:
                        print(f"Skipping trajectory {i} - response is {len(asst_tokens)} tokens long and conversation is {len(conversation_tokens)} tokens long")
                        del accumulated_compressed_memory, conversation_tokens
                        gc.collect()
                        torch.cuda.empty_cache()
                        break

                    # 2. tokenise & compress user message
                    user_tokens = self.model.tokenizer(user_msg["content"], truncation=False)["input_ids"]
                    if len(user_tokens) >= self.model.mem_size:
                        current_comp = self.model._compress(
                            torch.LongTensor(user_tokens).unsqueeze(0).to(args.device)
                        )
                        if accumulated_compressed_memory is None:
                            accumulated_compressed_memory = current_comp.to("cpu")
                        else:
                            accumulated_compressed_memory = torch.cat(
                                [accumulated_compressed_memory, current_comp.to("cpu")], dim=0
                            )
                        del current_comp
                    
                    # 3. build ICAE example (prompt+labels already contain placeholders)
                    example = create_icae_example(
                        input_tokens=user_tokens,
                        lm_target_tokens=asst_tokens,
                        task_type="swebench",
                        model=self.model,
                        text_tokens=conversation_tokens,
                    )

                    prompt_answer = torch.LongTensor(example["prompt_answer_ids"]).unsqueeze(0).to(args.device)
                    labels        = torch.LongTensor(example["labels"]).unsqueeze(0).to(args.device)
                    
                    # 4. forward / backward / optimiser.step
                    out = self.model(
                        input_ids=None,
                        prompt_answer_ids=prompt_answer,
                        labels=labels,
                        is_ae=example["is_ae"],
                        compressed_memory=accumulated_compressed_memory.to(args.device) if accumulated_compressed_memory is not None else None,
                    )
                    loss = out["loss"]
                    loss.backward()                       # gradients w.r.t. LoRA only
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if accumulated_compressed_memory is not None:
                        accumulated_compressed_memory = accumulated_compressed_memory.detach()

                    # ------------------------
                    # Compute BLEU and Accuracy metrics from forward logits
                    # ------------------------
                    with torch.inference_mode():
                        logits = out["logits"].detach()              # (1, seq_len, vocab)
                        pred_ids = torch.argmax(logits, dim=-1).squeeze(0)  # (seq_len)
                        lbl_ids  = labels.squeeze(0).detach()
                        answer_mask = lbl_ids != -100                 # positions corresponding to answer tokens
                        answer_positions = torch.nonzero(answer_mask, as_tuple=False).squeeze(-1)
                        valid_positions = answer_positions[answer_positions > 0]    # skip the first answer token
                        ref_ids = lbl_ids[valid_positions]
                        hyp_ids = pred_ids[valid_positions - 1]
                        #print("Reference:", self.model.tokenizer.decode(ref_ids, skip_special_tokens=True))
                        #print("Hypothesis:", self.model.tokenizer.decode(hyp_ids, skip_special_tokens=True))
                        bleu_score = compute_bleu(self.model.tokenizer, ref_ids, hyp_ids)
                        accuracy_score = compute_accuracy(ref_ids, hyp_ids)


                    # 5. update context for the next turn
                    conversation_tokens = prompt_answer.squeeze(0).tolist()
                    global_step += 1

                    wandb.log({
                        "train/step_loss": loss.item(),
                        "train/bleu": bleu_score,
                        "train/accuracy": accuracy_score,
                        "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "step": global_step,
                    })

                    # GPU Memory Profiling
                    if torch.cuda.is_available():
                        allocated_mem = torch.cuda.memory_allocated() / 1024**2
                        cached_mem = torch.cuda.memory_reserved() / 1024**2
                        max_allocated_mem = torch.cuda.max_memory_allocated() / 1024**2
                        max_cached_mem = torch.cuda.max_memory_reserved() / 1024**2

                        print(f"\n--- GPU Memory Stats (step {global_step}) ---")
                        print(f"Allocated: {allocated_mem:.2f} MB")
                        print(f"Cached: {cached_mem:.2f} MB")
                        print(f"Peak allocated: {max_allocated_mem:.2f} MB")
                        print(f"Peak reserved: {max_cached_mem:.2f} MB")

                        wandb.log({
                            "gpu/allocated_mb": allocated_mem,
                            "gpu/cached_mb": cached_mem,
                            "gpu/max_allocated_mb": max_allocated_mem,
                            "gpu/max_cached_mb": max_cached_mem,
                            "step": global_step,
                        })

                        print("--------------------------------------\n")

                        torch.cuda.reset_peak_memory_stats()

                    # 6. Memory cleanup for the current turn
                    # Delete tensors and variables that are no longer needed for the next turn
                    del loss, out, labels, prompt_answer, example, asst_tokens, user_tokens, user_msg, asst_msg, ref_ids, hyp_ids, bleu_score, accuracy_score, logits, pred_ids, lbl_ids

                    # Force garbage collection and empty CUDA cache
                    gc.collect()
                    torch.cuda.empty_cache()

                # ----- end trajectory loop -----

            ### TODO: do we need this?
            # optional epoch-level evaluation / checkpoint save
            #if args.evaluation_strategy == IntervalStrategy.EPOCH:
            #    self.evaluate()
            #if args.save_strategy == IntervalStrategy.EPOCH:
            #    self._save_checkpoint(model=self.model, trial=trial)

        self.is_in_train = False
        return global_step

# -----------------------------------------------------------------------------
# Training entry point
# -----------------------------------------------------------------------------

def main():
    model_args, data_args, training_args, inference_args = get_config()

    os.environ.setdefault("WANDB_PROJECT", "icae-swebench-finetune")

    wandb.init(
        project=os.environ["WANDB_PROJECT"],
        config={**vars(model_args), **vars(data_args), **vars(training_args)},
        name=f"{training_args.output_dir}",
    )

    # Ensure gradient checkpointing kwargs present
    # training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # ---------------------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------------------
    train_dataset_path = getattr(inference_args, "train_dataset_path", "icae/trajectories/openai/swe-smith/smith_train_openai.jsonl")
    eval_dataset_path = getattr(inference_args, "eval_dataset_path", "icae/trajectories/openai/swe-smith/smith_val_openai.jsonl")
    print(f"Loading trajectories from {train_dataset_path}...")
    train_trajectories = load_conversation_trajectories(train_dataset_path)
    print(f"Loading trajectories from {eval_dataset_path}...")
    eval_trajectories = load_conversation_trajectories(eval_dataset_path)
    print(f"Loaded {len(train_trajectories)} train trajectories and {len(eval_trajectories)} eval trajectories")
    

    train_dataset = TrajectoryDataset(train_trajectories)
    eval_dataset = TrajectoryDataset(eval_trajectories)

    # ---------------------------------------------------------------------
    # Model initialisation
    # ---------------------------------------------------------------------
    model = ICAE(model_args, training_args)

    # HF Trainer requires a pad_token
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    # ---------------------------------------------------------------------
    # Actual training
    # ---------------------------------------------------------------------

    collator = TrajectoryCollator()
    trainer = ICAETrajectoryTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    
    # Load from checkpoint if available
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        from transformers.trainer_utils import get_last_checkpoint

        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            device = next(model.parameters()).device
            print_loaded_layers(model, last_checkpoint, str(device))
            print(f"Resuming from checkpoint {last_checkpoint}")

    # Kick off training
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()

    # Optional evaluation (can be removed if not desired)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    wandb.finish()


if __name__ == "__main__":
    main()
