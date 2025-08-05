from torch.utils.data import Dataset

class MedCapDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        
        encoding = self.processor(images=item["image"], text=item["report"], 
                                  padding="max_length", return_tensors="pt")

        encoding = {k:v.squeeze() for k,v in encoding.items()}
        
        return encoding