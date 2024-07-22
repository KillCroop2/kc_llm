import torch
from kc_llm import GPTModel, load_tokenizer, get_vocab_size
import argparse
from transformers import GPT2Config


def load_model(model_path, vocab_size, device):
    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)

    # Infer model configuration from the state dict
    n_positions = state_dict['model.transformer.wpe.weight'].shape[0]
    n_embd = state_dict['model.transformer.wte.weight'].shape[1]
    n_layer = max([int(key.split('.')[3]) for key in state_dict.keys() if key.startswith('model.transformer.h.')]) + 1
    n_head = state_dict['model.transformer.h.0.attn.c_attn.weight'].shape[1] // (3 * n_embd)

    # Create a new model with the inferred configuration
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head
    )
    model = GPTModel(vocab_size, n_positions=n_positions, n_embd=n_embd, n_layer=n_layer, n_head=n_head)

    # Load the state dict
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def generate_text(model, tokenizer, prompt, max_length=200, num_return_sequences=1, device='cuda'):
    model.eval()
    encoded_prompt = tokenizer.encode(prompt, return_tensors='pt')
    if encoded_prompt.numel() == 0:
        print(f"Warning: The prompt '{prompt}' couldn't be encoded by the tokenizer.")
        return []

    encoded_prompt = encoded_prompt.to(device)

    try:
        # Generate text using the model's generate method
        output = model.generate(
            start_text=prompt,
            tokenizer=tokenizer,
            max_length=max_length,
            num_return_sequences=num_return_sequences
        )
    except RuntimeError as e:
        print(f"An error occurred during text generation: {str(e)}")
        return []

    return output


def main():
    parser = argparse.ArgumentParser(description="Generate text using the trained WikiGPT model")
    parser.add_argument("--model_path", type=str, default="wikigpt_model.pth", help="Path to the saved model")
    parser.add_argument("--prompt", type=str, default="The history of artificial intelligence",
                        help="Prompt for text generation")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum length of generated text")
    parser.add_argument("--num_sequences", type=int, default=1, help="Number of sequences to generate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = load_tokenizer()
    vocab_size = get_vocab_size(tokenizer)

    model = load_model(args.model_path, vocab_size, device)

    while True:
        prompt = input("Enter a prompt (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break

        generated_texts = generate_text(model, tokenizer, prompt, args.max_length, args.num_sequences, device)

        if generated_texts:
            print("\nGenerated Text:")
            for i, text in enumerate(generated_texts):
                print(f"\nSequence {i + 1}:")
                print(text)
        else:
            print("No text was generated.")
        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()