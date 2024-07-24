from torch.utils.data import Dataset
import json
import os


class ImprovedDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load the entire JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.documents = data['documents']

        self.num_samples = len(self.documents)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        doc = self.documents[idx]

        text = f"Title: {doc.get('title', '')}\n\n{doc.get('main_content', '')}"
        encoded = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze()
        }