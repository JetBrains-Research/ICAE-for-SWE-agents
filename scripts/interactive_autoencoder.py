import torch
from icae.configs import get_config
from icae.data.data_utils import create_icae_example, compute_bleu
from icae.models import ICAE


@torch.no_grad()
def run_interactive_autoencoding(model: ICAE, input_text: str, device):
    """Run autoencoding on a single input text and return results."""
    
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

    # 4. Decode results
    original_text = model.tokenizer.decode(answer_ids.squeeze(0), skip_special_tokens=True)
    reconstructed_text = model.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    
    # 5. Calculate BLEU score
    bleu_score = compute_bleu(
        tokenizer=model.tokenizer,
        reference_ids=answer_ids.squeeze(0),
        hypothesis_ids=torch.tensor(generated_token_ids)
    )

    return {
        "original_text": original_text,
        "reconstructed_text": reconstructed_text,
        "bleu_score": bleu_score,
        "input_length": len(input_tokens),
        "compressed": len(input_tokens) >= model.mem_size if model.model_args.do_compress else False
    }


def main():
    model_args, data_args, training_args, inference_args = get_config()
    
    # Setup model + tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() and not inference_args.use_cpu else "cpu")

    # Manually attach custom flags expected by ICAE
    model_args.use_position_identifiers = inference_args.use_position_identifiers
    training_args.use_cpu = inference_args.use_cpu
    training_args.train = True  # ensure decoder initialisation
    training_args.restore_from = inference_args.restore_from

    model = ICAE(model_args, training_args)

    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    model = model.to(device)
    model.eval()

    print("Interactive ICAE Autoencoder")
    print("=" * 50)
    print("Enter text to compress and decompress. Type 'quit' to exit.")
    print("=" * 50)

    while True:
        input_text = input("\nEnter text: ").strip()
        
        if input_text.lower() == 'quit':
            break
            
        if not input_text:
            print("Please enter some text.")
            continue

        print("\nProcessing...")
        
        results = run_interactive_autoencoding(model, input_text, device)
        
        print(f"\n--- Results ---")
        print(f"Input length: {results['input_length']} tokens")
        print(f"Compressed: {'Yes' if results['compressed'] else 'No'}")
        print(f"BLEU Score: {results['bleu_score']:.4f}")
        print(f"\nOriginal text:")
        print(f"'{results['original_text']}'")
        print(f"\nReconstructed text:")
        print(f"'{results['reconstructed_text']}'")


if __name__ == "__main__":
    main()
