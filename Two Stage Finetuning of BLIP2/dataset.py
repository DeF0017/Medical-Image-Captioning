import torch
from torch.utils.data import Dataset
from model import processor

class MedCapDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df
        self.len_df = df.shape[0]
        self.processor = processor
    
    def __len__(self):
        return self.len_df
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        encoding = self.processor(images=item["image"], 
                                  padding="max_length", 
                                  return_tensors="pt")
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["report"] = item["report"]
        encoding["MeSH"] = item["MeSH"]
        return encoding
    
def collate_fn(batch):
    # pad the input_ids and attention_mask
    processed_batch = {}
    for key in batch[0].keys():
        if key != "report" and key != "MeSH":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            if key == "report":
                # Tokenizing the 'report' field (text inputs)
                text_inputs = processor.tokenizer(
                    [example["report"] for example in batch], padding=True, return_tensors="pt"
                )
                processed_batch["input_ids"] = text_inputs["input_ids"]
                processed_batch["attention_mask"] = text_inputs["attention_mask"]
            
            if key == "MeSH":
                # Tokenizing the 'MeSH' field (text inputs)
                mesh_inputs = processor.tokenizer(
                    [example["MeSH"] for example in batch], padding=True, return_tensors="pt"
                )
                processed_batch["MeSH_input_ids"] = mesh_inputs["input_ids"]
                processed_batch["MeSH_attention_mask"] = mesh_inputs["attention_mask"]

    return processed_batch