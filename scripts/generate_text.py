import argparse
import torch
from kc_llm import GPTModel, load_tokenizer, get_vocab_size
from kc_llm.generation import generate_text


def load_model(model_path, vocab_size, device):
    model = GPTModel(vocab_size)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


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