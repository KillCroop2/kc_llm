import argparse
import torch
from kc_llm import GPTModel, load_data, train_model, load_tokenizer, get_vocab_size


def main():
    parser = argparse.ArgumentParser(description="Train the WikiGPT model")
    parser.add_argument("--data_file", type=str, default="scraped_data.json", help="Path to the data file")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--output_model", type=str, default="wikigpt_model.pth", help="Path to save the trained model")
    args = parser.parse_args()

    # Determine the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = load_tokenizer()
    dataset = load_data(args.data_file, tokenizer, args.max_length)

    vocab_size = get_vocab_size(tokenizer)
    model = GPTModel(vocab_size)

    trained_model = train_model(model, dataset, args.epochs, args.batch_size, args.learning_rate, device)

    # Save the model
    trained_model.save_pretrained(args.output_model)
    print(f"Model saved as {args.output_model}")


if __name__ == "__main__":
    main()
