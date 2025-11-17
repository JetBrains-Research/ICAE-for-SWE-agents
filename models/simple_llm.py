"""Simple LLM wrapper with LoRA support."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model
from safetensors.torch import load_file

from icae.models.model_utils import print_trainable_parameters, print_loaded_layers


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleLLM(nn.Module):
    def __init__(self, model_args, training_args, lora_config=None):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path

        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if not training_args.bf16 else torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_fast=False, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if lora_config is None:
            from peft import LoraConfig
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=32,
                lora_dropout=getattr(model_args, "lora_dropout", 0.05),
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=model_args.lora_target_modules,
            )

        self.llm = get_peft_model(self.llm, lora_config)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        print_trainable_parameters(self.llm)

        # Optionally restore from a provided checkpoint
        if self.training_args.restore_from:
            print(f"Loading from the pretrained checkpoint: {self.training_args.restore_from}...")
            state_dict = load_file(self.training_args.restore_from)
            self.load_state_dict(state_dict, strict=False)
            print_loaded_layers(self, self.training_args.restore_from, str(device))
            print(f"Finished loading from {self.training_args.restore_from}")
        else:
            print("No checkpoint provided. Initializing SimpleLLM from scratch.")


    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )
        # Enable input gradients for LoRA + gradient checkpointing compatibility
        self.llm.enable_input_require_grads()


    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False
        )
        return {"loss": outputs.loss, "logits": outputs.logits}