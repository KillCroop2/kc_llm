from .dataset import Dataset
import torch


def load_data(file_path, tokenizer, max_length=512):
    return Dataset(file_path, tokenizer, max_length)


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
