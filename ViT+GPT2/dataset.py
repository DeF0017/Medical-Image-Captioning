import torch
from torch.utils.data import Dataset

class MedCapDataset(Dataset):
    def __init__(self, df, processor, tokenizer):
        self.df = df
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load and process image
        item = self.df.iloc[idx];
        image = item["image"]
        report = item["report"]
        image = image.convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze()
        decoder_input_ids = self.tokenizer(report,
                                           padding="max_length").input_ids
        captions = [caption if caption != self.tokenizer.pad_token_id else -100 for caption in decoder_input_ids]
        return {
            'pixel_values': pixel_values, 
            'labels': torch.tensor(captions)}