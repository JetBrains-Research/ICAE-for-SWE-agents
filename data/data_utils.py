import time
import numpy as np
import torch
import re
import string
from datasets import load_dataset
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from icae.configs import TemplateManager
from transformers import DynamicCache


__all__ = [
    # Core functions
    "create_icae_example",
    
    # Fine-tuning functions - general
    "format_squad",
    "load_and_process_squad",
    
    # Fine-tuning functions - model specific
    "tokenize_qwen_icae_ft",
    "tokenize_mistral_icae_ft",
    "tokenize_mistral_llm_ft", 
    "tokenize_qwen_llm_ft",
    
    # Utility
    "truncate_cache",
    "compute_bleu",
    "normalize_text",
    "compute_exact_match",
    "compute_f1",
    "DataCollatorForDynamicPadding"
]

#######################
# CORE FUNCTIONS
#######################
### this is actually only for Qwen
def create_icae_example(input_tokens, lm_target_tokens, task_type, model, text_tokens=None):
    """
    Creates a single tokenized example for the ICAE model.
    This function contains the core tokenization logic for both pre-training and fine-tuning.
    """
    template_manager = TemplateManager(model.tokenizer)


    # compress
    if model.model_args.do_compress and len(input_tokens) >= model.mem_size:
        encoder_input_ids = template_manager.create_encoder_input(input_tokens)
        # Compute memory token placeholders *without* the template overhead !
        memory_token_placeholders = model.get_memory_placeholders(torch.LongTensor(input_tokens))
        ### if we want to delete tool outputs we do this:
        # encoder_input_ids = template_manager.create_encoder_input([])
        # memory_token_placeholders = model.tokenizer('<TOOL_OUTPUT>', add_special_tokens=False)['input_ids']
    else: # do not compress anything
        encoder_input_ids = input_tokens
        memory_token_placeholders = input_tokens

    # Create decoder prompt and labels based on the task
    if task_type == "squad":  # Instruction fine-tuning (e.g., SQuAD)
        prompt_content = template_manager.create_squad_prompt(memory_token_placeholders, text_tokens)
        answer_ids = lm_target_tokens  # The full answer is the target
        labels = [-100] * len(prompt_content) + answer_ids
    elif task_type == "ae":  # autoencoding
        # For AE, we only care about reconstructing the input_tokens.
        a = input_tokens
        prompt_content = template_manager.create_decoder_prompt_ae(memory_token_placeholders, model.ae_token_id)
        answer_ids = template_manager.create_answer_with_suffix(a)
        labels = [-100] * len(prompt_content) + answer_ids
    elif task_type == "lm":  # language modeling
        # For LM, input_tokens is context, lm_target_tokens is the generation target.
        prompt_content = template_manager.create_decoder_prompt_lm(memory_token_placeholders)
        answer_ids = lm_target_tokens
        leave_tokens = model.training_args.leave_tokens_for_lm
        labels = [-100] * len(prompt_content) + ([-100] * leave_tokens + answer_ids[leave_tokens:])
    elif task_type == "swebench":
        prompt_content = template_manager.create_swebench_prompt(memory_token_placeholders, text_tokens)
        answer_ids = lm_target_tokens
        labels = [-100] * len(prompt_content) + answer_ids
    
    prompt_answer_ids = prompt_content + answer_ids

    example = {
        "input_ids": torch.LongTensor(encoder_input_ids),
        "prompt_answer_ids": torch.LongTensor(prompt_answer_ids),
        "labels": torch.LongTensor(labels),
        "is_ae": torch.LongTensor([1 if task_type == "ae" else 0])
    }

    # Print example details
    # print("\n=== Example Details ===")
    # print(f"Task type: {task_type}")
    # print(f"Input length: {len(encoder_input_ids)}")
    # print(f"Prompt+Answer length: {len(prompt_answer_ids)}")
    # print(f"Labels length: {len(labels)}")
    
    #print("\nInput text:")
    #print(template_manager._safe_decode_with_mem_tokens(encoder_input_ids))
    #print("-" * 10 + 'start of prompt+answer text' + '-'*10)
    #print(template_manager._safe_decode_with_mem_tokens(prompt_answer_ids))
    #print('-'*10 + 'end of prompt+answer text' + '-'*10)
    #print("\nLabels:")
    #non_ignore_positions = [i for i, x in enumerate(labels) if x != -100]
    #print(f"{non_ignore_positions[0]}...{non_ignore_positions[-1]} are labels out of {len(labels)-1}")
    '''if len(non_ignore_positions) > 1000:
        print(f"--------------------------------")
        print("Very long generation! Here are first 100 and last 100 tokens of prompt_answer_ids at non-ignore positions:")
        first_100 = [prompt_answer_ids[i] for i in non_ignore_positions[:100]]
        last_100 = [prompt_answer_ids[i] for i in non_ignore_positions[-100:]]
        print("First 100:", template_manager._safe_decode_with_mem_tokens(first_100))
        print("Last 100:", template_manager._safe_decode_with_mem_tokens(last_100)) 
        print(f"--------------------------------")'''
    # print("=====================\n")

    return example

######################## FOR PRETRAINING DATA A WHOLE SEPARATE SCRIPT IS USED ########################

#######################
# FINE-TUNING - GENERAL
#######################

def format_squad(examples):
    return {
        "input": examples["context"],
        "prompt": [f"Based on the context, answer the following question in 5 words or less without formatting: {q}" for q in examples["question"]],
        "answer": [ans["text"][0] for ans in examples["answers"]],
    }

def load_and_process_squad(num_samples: int = None, include_answers: bool = False, split: str = "validation"):
    """Load the SQuAD dataset and return a processed ``pandas.DataFrame``."""
    
    dataset = load_dataset("rajpurkar/squad", split=split)
    formatted = dataset.map(format_squad, batched=True, remove_columns=dataset.column_names, load_from_cache_file=False)
    df = pd.DataFrame(formatted)

    if include_answers:
        # LM (QA)
        df = df[["input", "prompt", "answer"]]
        if num_samples is not None:
            df = df.sample(n=num_samples, random_state=42)
    else:
        # AE
        df = df[["input"]]
        if num_samples is not None:
            df = df.sample(n=num_samples, random_state=42)
    return df

#######################
# FINE-TUNING - MODEL SPECIFIC
#######################

### ICAE
def tokenize_qwen_icae_ft(examples, model, max_length=4096):
    """Tokenizes a SQuAD example for fine-tuning the ICAE model."""
    context = examples['input']
    question_text = examples['prompt']
    answer_text = examples['answer']

    # 1. Prepare tokens for context, question, and answer
    context_tokens = model.tokenizer(context, truncation=True, max_length=max_length - 50)['input_ids']
    question_tokens = model.tokenizer(question_text, truncation=False)['input_ids']
    answer_tokens = model.tokenizer(answer_text, truncation=True, max_length=50)['input_ids'] + [model.tokenizer.eos_token_id]

    # 2. Use the unified function to create the example
    # Fine-tuning is an LM task, so is_ae is False.
    return create_icae_example(
        input_tokens=context_tokens,
        lm_target_tokens=answer_tokens,
        task_type="squad",
        model=model,
        text_tokens=question_tokens
    )


def tokenize_mistral_icae_ft(examples, model):
    """Tokenizes examples for Mistral instruction fine-tuning (older format)."""
    text_output = model.tokenizer(examples["input"], max_length=model.training_args.model_max_length, truncation=True, padding=False, return_attention_mask=False, add_special_tokens=False)
    prompt_output = model.tokenizer(examples["prompt"], truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)
    label_output = model.tokenizer(examples["answer"], truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)
    text_output['prompt_answer_ids'] = []
    text_output['labels'] = []
    text_output['is_ae'] = []

    for idx in range(len(text_output["input_ids"])):
        prompt_ids = model.get_memory_placeholders(torch.LongTensor(text_output["input_ids"][idx])) + [model.ft_token_id] + prompt_output['input_ids'][idx]
        prompt_ids = [1, 733, 16289, 28793] + prompt_ids + [733, 28748, 16289, 28793]   # special formats for prompt in Mistral
        answer_ids = label_output['input_ids'][idx] + [model.eos_id]

        text_output['prompt_answer_ids'].append(prompt_ids + answer_ids)
            
        labels = [-100] * len(prompt_ids) + answer_ids
        text_output['labels'].append(labels)
        text_output['is_ae'].append([0]) # not AE since we are finetuning
        
        assert len(text_output['prompt_answer_ids'][-1]) == len(labels)
        
    return text_output


### LLM
def tokenize_mistral_llm_ft(examples, tokenizer, max_length):
    """Tokenizes examples for Mistral LLM instruction fine-tuning."""
    input_ids_list = []
    labels_list = []

    for i in range(len(examples["prompt"])):
        context = examples['input'][i]
        question = examples['prompt'][i]
        answer = examples["answer"][i]

        prompt_text = context + question

        # Following Mistral instruction format
        full_text = tokenizer.bos_token + "[INST] " + prompt_text.strip() + " [/INST] " + answer + tokenizer.eos_token

        tokenized_full = tokenizer(full_text, max_length=max_length, truncation=True, padding=False, return_attention_mask=False)
        input_ids = tokenized_full['input_ids']
        labels = list(input_ids)
        
        # Mask prompt tokens
        # To account for the prepended BOS and [INST] tokens, we find the answer tokens to mask the prompt
        tokenized_answer = tokenizer(answer + tokenizer.eos_token, max_length=max_length, truncation=True, padding=False, return_attention_mask=False, add_special_tokens=False)
        answer_len = len(tokenized_answer['input_ids'])
        prompt_len = len(input_ids) - answer_len
        labels[:prompt_len] = [-100] * prompt_len
        
        input_ids_list.append(input_ids)
        labels_list.append(labels)
    
    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
    }


def tokenize_qwen_llm_ft(examples, tokenizer, max_length):
    """Tokenizes examples for Qwen LLM instruction fine-tuning."""
    input_ids_list = []
    labels_list = []
    template_manager = TemplateManager(tokenizer)

    for i in range(len(examples["prompt"])):
        context = examples['input'][i]
        question = examples['prompt'][i]
        answer = examples["answer"][i]

        # First tokenize the prompt part
        messages_prompt = [{"role": "user", "content": context + question}]
        prompt_text = tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokenized_prompt = tokenizer(prompt_text, max_length=max_length, truncation=False, padding=False, return_attention_mask=False)
        prompt_ids = tokenized_prompt['input_ids']

        # Then tokenize the answer part
        tokenized_answer = tokenizer(answer, max_length=max_length, truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)
        answer_ids = template_manager.create_answer_with_suffix(tokenized_answer['input_ids'])

        # Combine them
        input_ids = prompt_ids + answer_ids
        
        # Create labels with -100 for prompt
        labels = [-100] * len(prompt_ids) + answer_ids
        
        input_ids_list.append(input_ids)
        labels_list.append(labels)
    
    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
    }

#######################
# UTILITY
#######################

def truncate_cache(cache: "DynamicCache", keep: int):
    for i in range(len(cache.key_cache)):
        cache.key_cache[i]  = cache.key_cache[i][..., :keep, :].contiguous()
        cache.value_cache[i] = cache.value_cache[i][..., :keep, :].contiguous()

    if hasattr(cache, "_seen_tokens"):
        cache._seen_tokens = keep
    elif hasattr(cache, "seen_tokens"):
        cache.seen_tokens = keep

def compute_bleu(tokenizer, reference_ids: torch.Tensor, hypothesis_ids: torch.Tensor) -> float:
    """Compute BLEU score between reference and hypothesis (default BLEU-1)."""
    reference = tokenizer.decode(reference_ids, skip_special_tokens=True)
    hypothesis = tokenizer.decode(hypothesis_ids, skip_special_tokens=True)
    weights = (1.0, 0.0, 0.0, 0.0)
    return sentence_bleu([word_tokenize(reference)], word_tokenize(hypothesis), weights=weights)

def compute_accuracy(reference_ids: torch.Tensor, hypothesis_ids: torch.Tensor) -> float:
    """Computes token-level accuracy between reference and hypothesis sequences."""
    # Convert to numpy arrays for consistent handling
    if isinstance(reference_ids, torch.Tensor):
        reference_ids = reference_ids.cpu().numpy()
    if isinstance(hypothesis_ids, torch.Tensor):
        hypothesis_ids = hypothesis_ids.cpu().numpy()

    # Compare sequences element-wise up to the minimum length
    matches = (reference_ids == hypothesis_ids)
    return float(np.mean(matches))

def normalize_text(s: str) -> str:
    """Normalize text following SQuAD evaluation rules."""
    def take_before_next_line(text: str) -> str:
        return text.split("\n")[0]

    def remove_list_numbers(text: str) -> str:
        return re.sub(r"^\d+\.\s*", "", text)

    def remove_articles(text: str) -> str:
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    text = take_before_next_line(s)
    text = remove_list_numbers(text)
    text = lower(text)
    text = remove_punc(text)
    text = remove_articles(text)
    return white_space_fix(text)

def compute_exact_match(prediction: str, truth: str) -> int:
    """Return 1 if normalized prediction exactly matches normalized truth else 0."""
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction: str, truth: str) -> float:
    """Compute token-level F1 score between prediction and truth after normalization."""
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)

    common = set(pred_tokens) & set(truth_tokens)
    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * (precision * recall) / (precision + recall) 


class DataCollatorForDynamicPadding:
    """Data collator for dynamic padding of tokenized examples."""
    def __init__(self, pad_token_id, pad_to_multiple_of=None):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
        
    def __call__(self, examples):
        input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in examples]
        labels = [torch.tensor(example["labels"], dtype=torch.long) for example in examples]
        prompt_answer_ids = [torch.tensor(example["prompt_answer_ids"], dtype=torch.long) for example in examples]
        input_ids = self.dynamic_padding(input_ids, fill_value=self.pad_token_id)
        prompt_answer_ids = self.dynamic_padding(prompt_answer_ids, fill_value=self.pad_token_id)
        labels = self.dynamic_padding(labels)
        batch = {"input_ids": input_ids, "labels": labels, "prompt_answer_ids": prompt_answer_ids}
        return batch
        
    def dynamic_padding(self, sequences, fill_value=-100):
        max_length = max(len(x) for x in sequences)
        if self.pad_to_multiple_of:
            max_length = ((max_length - 1) // self.pad_to_multiple_of + 1) * self.pad_to_multiple_of
        padded_sequences = torch.full((len(sequences), max_length), fill_value, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded_sequences[i, :len(seq)] = seq
        return padded_sequences
