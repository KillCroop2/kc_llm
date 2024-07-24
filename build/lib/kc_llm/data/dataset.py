import torch
from torch.utils.data import Dataset
import json
import random

class ImprovedDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512, min_length=10):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.documents = data['documents']

        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        samples = []
        for doc in self.documents:
            text = f"Title: {doc['title']}\n\nContent: {doc['main_content']}"
            tokens = self.tokenizer.encode(text)
            
            # Split long documents into multiple samples
            for i in range(0, len(tokens), self.max_length):
                chunk = tokens[i:i + self.max_length]
                if len(chunk) >= self.min_length:
                    samples.append(chunk)
        
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]

        # Randomly truncate or pad the sequence
        if len(tokens) > self.max_length:
            start = random.randint(0, len(tokens) - self.max_length)
            tokens = tokens[start:start + self.max_length]
        else:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))

        input_ids = torch.tensor(tokens)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }