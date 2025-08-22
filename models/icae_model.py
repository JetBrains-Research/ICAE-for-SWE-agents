"""ICAE model definition (multi-span variant)."""

import math
from typing import List, Optional

import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from transformers.modeling_utils import load_sharded_checkpoint
from safetensors.torch import load_file
from icae.configs.templates import TemplateManager
from icae.models.model_utils import freeze_model, print_trainable_parameters, print_loaded_layers


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ICAE(nn.Module):
    """
    Implementation of the In-Context AutoEncoder (ICAE) that supports multi-span concatenation.
    Always uses LoRA for finetuning.
    """

    def __init__(self, model_args, training_args, lora_config=None):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path
        self.target_dtype = torch.bfloat16 if training_args.bf16 else torch.float16

        if lora_config is None:
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=32,
                lora_dropout=getattr(model_args, "lora_dropout", 0.05),
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=model_args.lora_target_modules,
            )

        # Base encoder model
        self.icae = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.target_dtype,
            # attn_implementation="flash_attention_2"
        )

        # Shared tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_fast=False, trust_remote_code=True
        )

        self.restore = self.training_args.restore_from

        # Independent decoder for gradient-checkpointing during training
        if self.training_args.train:
            self.decoder = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.target_dtype,
                attn_implementation="flash_attention_2"
            )
            if not self.model_args.freeze_decoder:
                self.decoder = get_peft_model(self.decoder, lora_config)

        # Model/sequence-level constants
        self.vocab_size = self.icae.config.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mean_compression_rate = training_args.mean_compression_rate

        # Memory / special-token bookkeeping
        self.mem_size = self.training_args.fixed_mem_size
        self.vocab_size_with_mem = self.vocab_size + self.mem_size

        # Reserve ids for additional special tokens after memory tokens
        self.ae_token_id = self.vocab_size_with_mem + 0
        self.lm_token_id = self.vocab_size_with_mem + 1
        self.ft_token_id = self.vocab_size_with_mem + 2

        # Resize token embeddings to accommodate new special tokens
        self.icae.resize_token_embeddings(self.vocab_size_with_mem + 3)

        # Required by HF Trainer when loading checkpoints
        # self._keys_to_ignore_on_save = None

        # Tokenizer-specific constants
        self.eos_id = self.tokenizer.eos_token_id

        # Wrap the encoder with LoRA adapters
        self.dim = self.icae.config.hidden_size
        self.icae = get_peft_model(self.icae, lora_config)

        # Embeddings for memory & additional special tokens
        self.memory_token_embed = nn.Embedding(self.mem_size + 3, self.dim, padding_idx=None, device=device, dtype=self.target_dtype)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.append_sequence = torch.arange(
            self.vocab_size, self.vocab_size + self.mem_size, dtype=torch.long
        ).unsqueeze(0)

        if self.training_args.train:
            self._init_training_components()

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _init_training_components(self):
        if self.model_args.freeze_encoder:
            print("Freezing the encoder...")
            freeze_model(self.icae)
            self.icae.eval()
        if self.model_args.freeze_decoder:
            print("Freezing the decoder...")
            freeze_model(self.decoder)
            self.decoder.eval()

        self.icae = self.icae.to(device=device, dtype=self.target_dtype)
        self.decoder = self.decoder.to(device=device, dtype=self.target_dtype)
        self.memory_token_embed = self.memory_token_embed.to(device=device, dtype=self.target_dtype)

        print_trainable_parameters(self)

        
        ### TODO: if restore is a directory, load the DECODER from it only!
        if self.restore and Path(self.restore).is_dir():
            # folder with e.g. model-00001-of-0000x.safetensors + model.safetensors.index.json
            load_sharded_checkpoint(self.decoder, self.restore, strict=True, prefer_safe=True)
            print_loaded_layers(self.decoder, self.restore, str(device))
            print(f"Finished loading from {self.restore}")
        elif self.restore:
            state_dict = load_file(self.restore)
            self.load_state_dict(state_dict, strict=False)
            print_loaded_layers(self, self.restore, str(device))
            print(f"Finished loading from {self.restore}")
        else:
            print("No checkpoint provided. Initializing ICAE from scratch.")

        if self.training_args.gradient_checkpointing:
            self.icae.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.icae.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    @torch.no_grad()
    def get_memory_placeholders(self, input_ids) -> List[int]:
        """Returns a list of memory placeholder tokens."""
        seq_len = input_ids.size(1) if input_ids.ndim > 1 else len(input_ids)
        num_placeholders = self._compute_num_segments(seq_len) * self.mem_size
        return [self.vocab_size] * num_placeholders

    @torch.no_grad()
    def _compute_num_segments(self, total_length: int):
        return math.ceil(total_length / (self.mem_size * self.mean_compression_rate))


    def _tokens_to_embeddings(self, token_ids):
        embeddings = self.icae.get_base_model().model.embed_tokens(token_ids)
        special_flags = token_ids >= self.vocab_size
        embeddings[special_flags] = self.memory_token_embed(
            token_ids[special_flags] - self.vocab_size
        ).to(embeddings)
        return embeddings


    def _compress(self, input_ids: torch.LongTensor = None):
        """Compress a (potentially long) ``input_ids`` sequence into memory slots."""
        total_length = input_ids.size(1)
        num_segments = self._compute_num_segments(total_length)
        segment_length = math.ceil(total_length / num_segments)

        max_compressed_length = num_segments * self.mem_size
        compressed_memory = torch.zeros(
            (max_compressed_length, self.dim),
            dtype=self.target_dtype,
            device="cpu",
        )

        for segment_idx in range(num_segments):
            start_idx = segment_idx * segment_length
            end_idx = min((segment_idx + 1) * segment_length, total_length)
            segment_input_ids = input_ids[:, start_idx:end_idx]
            segment_input_ids = torch.cat(
                [segment_input_ids, self.append_sequence.to(input_ids.device)], dim=1
            )
            mem_flag = segment_input_ids >= self.vocab_size

            segment_input_embedding = self._tokens_to_embeddings(segment_input_ids)

            if self.model_args.use_position_identifiers:
                seq_len = end_idx - start_idx
                position_ids = torch.arange(1, seq_len + 1, device=segment_input_embedding.device).unsqueeze(0)
                mem_position_ids = (
                    torch.linspace(1, seq_len, self.mem_size, device=segment_input_embedding.device)
                    .long()
                    .unsqueeze(0)
                )
                encode_position_ids = torch.cat([position_ids, mem_position_ids], dim=1)

                ### TODO: check if this is correct instead of just using self.icae(..., output_hidden_states=True)
                segment_compressed_memory = self.icae.get_base_model().model(
                    inputs_embeds=segment_input_embedding,
                    position_ids=encode_position_ids,
                    output_hidden_states=False,
                ).last_hidden_state
            else:
                segment_compressed_memory = self.icae.get_base_model().model(
                    inputs_embeds=segment_input_embedding,
                    output_hidden_states=False,
                ).last_hidden_state

            # collect memory tokens
            compressed_memory[
                segment_idx * self.mem_size : self.mem_size * (segment_idx + 1)
            ] = segment_compressed_memory[mem_flag].cpu()

            ### TODO: throughtly check this! i believe that with this commented out we spend more memory, but it's faster
            del segment_input_ids, segment_input_embedding, segment_compressed_memory
            torch.cuda.empty_cache()

        return compressed_memory.to(input_ids.device)


    def _run_decoder(self, embeddings: torch.Tensor, past_key_values=None, use_cache=True):
        """Run the (frozen) decoder or, if not present, the base ICAE model."""
        if hasattr(self, "decoder") and self.decoder is not None:
            return self.decoder(
                inputs_embeds=embeddings,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=False
            )
        else:
            # Fall back to the base model with LoRA adapters disabled
            with self.icae.disable_adapter():
                return self.icae(
                    inputs_embeds=embeddings,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_hidden_states=False
                )


    def _inject_memory(
        self,
        token_ids: torch.LongTensor,
        embeddings: torch.Tensor,
        memory_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """Replace placeholder memory-token embeddings inside *embeddings* with *memory_vectors*."""
        if memory_vectors is None:
            return embeddings
        
        mem_mask = (token_ids >= self.vocab_size) & (token_ids < self.vocab_size + self.mem_size)  # shape (B, L)

        # Allow a leading batch dimension on memory_vectors
        if memory_vectors.dim() == 3:
            memory_vectors = memory_vectors.squeeze(0)

        if mem_mask.sum() != memory_vectors.shape[0]:
            raise ValueError(f"Memory mask length mismatch: {mem_mask.sum()} vs {memory_vectors.shape[0]}")

        embeddings[mem_mask] = memory_vectors.to(embeddings)
        return embeddings


    # ------------------------------------------------------------------
    # Main methods
    # ------------------------------------------------------------------
    def prepare_prompt_embeddings(
            self, 
            input_ids: torch.LongTensor = None, 
            prompt_ids: torch.LongTensor = None,
            compressed_memory: torch.Tensor = None,
        ) -> torch.Tensor:
        """
            Convenience wrapper that compresses input_ids into memory slots, 
            converts prompt_ids (containing the placeholder tokens) to embeddings, 
            and injects the compressed memory into those placeholders.
        """
        # 1. Compress the long context into fixed-size memory slots
        compressed_memory = self._compress(input_ids) if input_ids is not None else compressed_memory

        # 2. Obtain embeddings for the prompt tokens
        prompt_embs = self._tokens_to_embeddings(prompt_ids)

        # 3. Replace placeholder embeddings with the compressed memory
        prompt_embs = self._inject_memory(
            token_ids=prompt_ids,
            embeddings=prompt_embs,
            memory_vectors=compressed_memory,
        )
        return prompt_embs
    
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        prompt_answer_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        is_ae: Optional[torch.LongTensor] = None,
        compressed_memory: torch.Tensor = None,
        cache = None,
    ):
        # Prepare embeddings
        prompt_answer_embs = self.prepare_prompt_embeddings(input_ids, prompt_answer_ids, compressed_memory=compressed_memory)

        past_key_values = cache

        decoder_outputs = self._run_decoder(embeddings=prompt_answer_embs, past_key_values=past_key_values, use_cache=True)

        logits = decoder_outputs.logits
        effective_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        target_ids = labels[:, 1:].reshape(-1)
        loss = self.loss_fct(effective_logits, target_ids)

        trained_tokens = (target_ids != -100).sum().item()
        ae_loss = 0.0
        lm_loss = 0.0

        if is_ae is not None:
            # batch size is 1, so we can use item()
            if is_ae.item() == 1:
                ae_loss = loss.item()
            else:
                lm_loss = loss.item()

        return {
            "loss": loss,
            "logits": logits,
            "ae_loss": ae_loss,
            "lm_loss": lm_loss,
            "trained_tokens": trained_tokens,
        }
    
    @torch.no_grad() # TODO: maybe check? seems fine to me
    def generate_autoregressive(
        self,
        input_ids: torch.LongTensor,
        prompt_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_at_eos: bool = True,
        compressed_memory: torch.Tensor = None,
        cache = None
    ) -> list:
        """Generate tokens autoregressively given *input_ids* (context to compress)
        and *prompt_ids* (decoder prompt containing placeholder memory tokens).
        """
        # 1. Prepare decoder embeddings
        prompt_embs = self.prepare_prompt_embeddings(input_ids, prompt_ids, compressed_memory=compressed_memory)

        # 2. Greedy decoding loop
        output_embs = prompt_embs
        generated_tokens = []
        past_key_values = cache

        for _ in range(max_new_tokens):
            out = self._run_decoder(
                embeddings=output_embs,
                past_key_values=past_key_values,
                use_cache=True
            )

            logits = out.logits[:, -1, :]
            past_key_values = out.past_key_values

            logits = logits[:, :self.vocab_size]

            next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)  # shape (1, 1)

            if stop_at_eos and next_token_id.item() == self.tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token_id.item())
            # Prepare embedding for the next step (only the last token is needed)
            output_embs = self._tokens_to_embeddings(next_token_id)

        return generated_tokens 

