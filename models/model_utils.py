from transformers import Trainer
import os
import torch
from typing import Optional
from transformers.trainer_utils import get_last_checkpoint
from tqdm import tqdm
import logging
from safetensors.torch import load_file
from icae.data.data_utils import compute_bleu
import wandb

__all__ = [
    "train_model",
    "ICAETrainer",
    "print_trainable_parameters",
    "freeze_model"
]

def print_trainable_parameters(model):
    """Print the total and trainable parameter counts of a model."""
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    pct = 100 * trainable_parameters / all_param if all_param else 0
    print(
        f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {pct:.2f}"
    )

def print_loaded_layers(model: torch.nn.Module, checkpoint_path: str, device: str):
    """Pretty-print summary of parameter tensors restored from *checkpoint_path*."""
    state_dict = load_file(checkpoint_path, device=device)
    loaded_layers = []
    skipped_layers = []

    for name, _ in model.named_parameters():
        if name in state_dict:
            loaded_layers.append(name)
        else:
            skipped_layers.append(name)

    print(f"\nTotal layers loaded: {len(loaded_layers)}")
    print(f"Total layers skipped: {len(skipped_layers)}")

    print("\nLoaded layers:")
    for layer in loaded_layers[:3]:
        print(f"✓ {layer}")
    print("...")
    for layer in loaded_layers[-3:]:
        print(f"✓ {layer}")

    print("\nSkipped layers:")
    for layer in skipped_layers[:3]:
        print(f"✗ {layer}")
    print("...")
    for layer in skipped_layers[-3:]:
        print(f"✗ {layer}")

def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False 


class ICAETrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cumulative_trained_tokens = 0

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Custom save method to exclude the frozen decoder from checkpoints.
        This prevents safetensors from erroring on tied weights.
        """
        if state_dict is None:
            state_dict = self.model.state_dict()

        # Do not save the decoder since it is not trained
        keys_to_remove = [k for k in state_dict if k.startswith('decoder.')]
        for k in keys_to_remove:
            del state_dict[k]

        # Handle tied weights within the main ICAE model for safetensors
        if '4B' in self.model.model_args.model_name_or_path or '3B' in self.model.model_args.model_name_or_path:
            if 'icae.base_model.model.lm_head.weight' in state_dict:
                del state_dict['icae.base_model.model.lm_head.weight']

        super()._save(output_dir, state_dict=state_dict)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=1):
        """
        Overrides the default compute_loss to log autoencoding and language modeling
        losses separately to wandb during training.
        """
        outputs = model(**inputs)
        loss = outputs.get("loss")

        # During training, log the per-step ae and lm losses if they are present.
        if model.training and self.state.is_local_process_zero:
            # Update cumulative token count regardless of logging
            if "trained_tokens" in outputs:
                self.cumulative_trained_tokens += outputs["trained_tokens"]
            
            # Only log every logging_steps steps
            if self.state.global_step % self.args.logging_steps == 0:
                log_dict = {}
                if "ae_loss" in outputs and outputs["ae_loss"] > 0:
                    log_dict["train/ae_loss"] = outputs["ae_loss"]
                if "lm_loss" in outputs and outputs["lm_loss"] > 0:
                    log_dict["train/lm_loss"] = outputs["lm_loss"]

                if log_dict:
                    log_dict["train/trained_tokens"] = self.cumulative_trained_tokens
                    wandb.log(log_dict)

        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        # Run the default evaluation
        metrics = super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        # If the model is a LLM or the LM ratio is 1.0, skip the AE inference
        if self.model.model_args.model_type == "llm" or self.model.training_args.lm_ratio == 1.0:
            return metrics

        ### Run AE inference for BLEU score ###
        # Filter for AE tasks using the 'is_ae' flag from the dataset.
        ae_eval_dataset = self.eval_dataset.filter(lambda example: example['is_ae'][0] == 1)

        # Limit number of samples for speed
        num_samples = min(self.args.eval_ae_num_samples, len(ae_eval_dataset))
        ae_eval_samples = ae_eval_dataset.select(range(num_samples))

        bleu_scores = []

        # Suppress verbose generation warnings during inference
        logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

        for example in tqdm(ae_eval_samples, desc="Running AE Inference for BLEU score"):
            with torch.no_grad():
                # Prepare all data tensors and move to the correct device
                input_ids_tensor = torch.tensor(example['input_ids']).unsqueeze(0).to(self.args.device)
                prompt_answer_ids_tensor = torch.tensor(example['prompt_answer_ids']).unsqueeze(0).to(self.args.device)
                labels_tensor = torch.tensor(example['labels']).to(self.args.device)

                # 1. Split prompt and answer using the 'labels' tensor
                # The prompt part is where labels are -100
                prompt_len = (labels_tensor == -100).sum()
                prompt_ids = prompt_answer_ids_tensor[:, :prompt_len]
                answer_ids = prompt_answer_ids_tensor[:, prompt_len:]

                # 2. Autoregressively generate the reconstruction using the model's helper
                max_gen_len = answer_ids.shape[1]
                generated_tokens = self.model.generate_autoregressive(
                    input_ids_tensor,
                    prompt_ids,
                    max_new_tokens=max_gen_len,
                )

                # 4. Calculate BLEU score
                bleu_scores.append(
                    compute_bleu(
                        tokenizer=self.model.tokenizer, 
                        reference_ids=answer_ids.squeeze(0),
                        hypothesis_ids=generated_tokens
                    )
                )

        # Restore logging level
        logging.getLogger("transformers.generation.utils").setLevel(logging.WARNING)

        if bleu_scores:
            avg_bleu = sum(bleu_scores) / len(bleu_scores)
            metrics[f"{metric_key_prefix}/avg_bleu"] = avg_bleu
            log_dict = {}
            log_dict[f"{metric_key_prefix}/avg_bleu"] = avg_bleu
            wandb.log(log_dict)

        return metrics


def train_model(model, train_dataset, eval_dataset, training_args, data_collator=None):

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # print training_args at local_rank 0
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    if local_rank == 0:
        print(training_args)

    trainer = ICAETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
    
    print(f"Loaded from the checkpoint: {last_checkpoint}")
    
    if last_checkpoint is not None:
        device = next(model.parameters()).device
        print_loaded_layers(model, last_checkpoint, device)

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    wandb.finish()