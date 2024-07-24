from torch.utils.data import DataLoader
import torch
from .dataset import ImprovedDataset

def load_data(file_path, tokenizer, max_length=512):
    return ImprovedDataset(file_path, tokenizer, max_length)

class EfficientDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=0, pin_memory=False, sampler=None):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                         pin_memory=pin_memory, sampler=sampler, collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

# Keep the original collate_fn for compatibility if needed
def collate_fn(batch):
    return EfficientDataLoader.collate_fn(batch)