from torch.utils.data import Dataset
import json

class Dataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_and_process_data(file_path)

    def load_and_process_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        processed_data = []
        for doc in raw_data['documents']:
            text = f"Title: {doc['title']}\n\n{doc['main_content']}"
            encoded = self.tokenizer.encode_plus(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            processed_data.append({
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze()
            })

        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]