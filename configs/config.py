import yaml
from dataclasses import dataclass, field
import dataclasses
import transformers
from typing import Optional, Union, List
import argparse

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=None)
    model_type: str = field(default=None)
    lora_r: int = field(default=None)
    lora_dropout: float = field(default=None)
    train: bool = field(default=None)
    lora_target_modules: List[str] = field(default=None)
    use_position_identifiers: bool = field(default=None)
    do_compress: bool = field(default=None)
    freeze_encoder: bool = field(default=None)
    freeze_decoder: bool = field(default=None)

@dataclass
class DataArguments:
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    max_out_length: int = field(default=None)
    # Pretrain Data Arguments
    dataset_repo: str = field(default=None)
    token_num: int = field(default=None)
    min_len: int = field(default=None)
    train_output_file: str = field(default=None)
    eval_output_file: str = field(default=None)
    long_text_cache: str = field(default=None)
    eval_size: int = field(default=None)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    fixed_mem_size: int = field(default=None)
    mean_compression_rate: int = field(default=None)
    min_tokens_for_lm: int = field(default=None)
    leave_tokens_for_lm: int = field(default=None)
    lm_ratio: float = field(default=None)
    add_special_token_for_lm: bool = field(default=None)
    restore_from: str = field(default=None)
    train: bool = field(default=None)
    eval_ae_num_samples: int = field(default=None)
    model_max_length: int = field(default=None)
    num_training_steps: int = field(default=None)
    gradient_checkpointing: bool = field(default=None)
    save_decoder: Optional[bool] = field(default=None)

@dataclass
class InferenceArguments:
    task: str = field(default=None)
    restore_from: str = field(default=None)
    num_samples: int = field(default=None)
    use_cpu: bool = field(default=None)
    use_position_identifiers: bool = field(default=None)
    teacher_forcing: bool = field(default=True)

def get_config():
    """Parses command-line arguments for the config path and loads the config."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    # Get all TrainingArguments fields
    all_training_args_fields = [f.name for f in dataclasses.fields(transformers.TrainingArguments(output_dir="."))]
    all_training_args_fields.extend([f.name for f in dataclasses.fields(TrainingArguments(output_dir="."))])
    # For TrainingArguments, we need to populate the base and the extended class
    training_args_dict = {k: v for k, v in config_dict.items() if k in all_training_args_fields}
    training_args = TrainingArguments(**training_args_dict)

    model_args = ModelArguments(**{k: v for k, v in config_dict.items() if k in [f.name for f in dataclasses.fields(ModelArguments)]})
    data_args = DataArguments(**{k: v for k, v in config_dict.items() if k in [f.name for f in dataclasses.fields(DataArguments)]})
    inference_args = InferenceArguments(**{k: v for k, v in config_dict.items() if k in [f.name for f in dataclasses.fields(InferenceArguments)]})

    training_args.save_decoder = not model_args.freeze_decoder

    return model_args, data_args, training_args, inference_args

if __name__ == '__main__':
    model_args, data_args, training_args, inference_args = get_config()
    print("--- Model Arguments ---")
    print(model_args)
    print("\n--- Data Arguments ---")
    print(data_args)
    print("\n--- Training Arguments ---")
    print(training_args)
    print("\n--- Inference Arguments ---")
    print(inference_args)