from transformers import GPT2Tokenizer


def load_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # Add padding token to the tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_vocab_size(tokenizer):
    return len(tokenizer)
