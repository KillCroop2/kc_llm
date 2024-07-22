import argparse
import torch
from kc_llm import GPTModel, load_tokenizer, generate_text


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
    model = GPTModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    generated_texts = generate_text(model, tokenizer, args.prompt, args.max_length, args.num_sequences, device)

    print("\nGenerated Text:")
    for i, text in enumerate(generated_texts):
        print(f"\nSequence {i + 1}:")
        print(text)


if __name__ == "__main__":
    main()
