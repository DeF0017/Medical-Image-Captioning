import torch
from PIL import Image
from torch.utils.data import Dataset
from model import processor

class MedCapDataset(Dataset):
    def __init__(self, df, blip_processor, dino_processor):
        self.df = df
        self.blip_processor = blip_processor
        self.dino_processor = dino_processor
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        image = item["image"]
        
        if isinstance(image, str):
            image = Image.open(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # BLIP encoding
        blip_encoding = self.blip_processor(images=image, return_tensors="pt")
        # DINOv2 encoding
        dino_encoding = self.dino_processor(images=image, return_tensors="pt")

        encoding = {
            "pixel_values": blip_encoding["pixel_values"].squeeze(),          # BLIP pixel input
            "dino_pixel_values": dino_encoding["pixel_values"].squeeze()      # DINO input
        }

        encoding["report"] = item["report"]
        if "MeSH" in item:
            encoding["MeSH"] = item["MeSH"]

        return encoding

def collate_fn(batch):
    processed_batch = {}

    # Stack image tensors
    for key in ["pixel_values", "dino_pixel_values"]:
        processed_batch[key] = torch.stack([example[key] for example in batch])

    # Reports → tokenized text
    text_inputs = processor.tokenizer(
        [example["report"] for example in batch],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    processed_batch["input_ids"] = text_inputs["input_ids"]
    processed_batch["attention_mask"] = text_inputs["attention_mask"]
    processed_batch["labels"] = text_inputs["input_ids"].clone()

    # Optional MeSH terms
    if "MeSH" in batch[0]:
        mesh_inputs = processor.tokenizer(
            [example["MeSH"] for example in batch],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        processed_batch["MeSH_input_ids"] = mesh_inputs["input_ids"]
        processed_batch["MeSH_attention_mask"] = mesh_inputs["attention_mask"]

    return processed_batch