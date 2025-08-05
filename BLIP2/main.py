import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from dataset import MedCapDataset, collate_fn
from model import model, processor, freeze_model_except_qformer
from train import train

dir_path = "/kaggle/input/chest-xrays-indiana-university"
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 5e-4
epoch = 15
batch_size = 2

xray_df = pd.read_csv("/kaggle/input/chest-xrays-indiana-university/indiana_projections.csv")
xray_df =  xray_df[xray_df['projection'] == "Frontal"]
xray_df.reset_index(drop=True, inplace=True)

report_df = pd.read_csv("/kaggle/input/chest-xrays-indiana-university/indiana_reports.csv")
report_df['report'] = report_df['findings'] + ' ' + report_df['impression']
report_df.drop(['MeSH', 'Problems', 'image', 'indication', 'comparison', 'findings', 'impression'], axis=1, inplace=True)
report_df.dropna(inplace=True)
report_df.reset_index(drop=True, inplace=True)

df = pd.merge(xray_df, report_df, on='uid', how='inner')
df.drop(['projection'], axis=1, inplace=True)

def load_image(file_path):
    try:
        image = Image.open(f"{dir_path}/images/images_normalized/{file_path}")
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

df['image'] = df['filename'].apply(load_image)
df.drop(["filename"], axis=1, inplace=True)

def main():
    train_ds = MedCapDataset(df, processor)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
    
    freeze_model_except_qformer(model)
    
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    
    train(model, processor, optimizer, train_dl, epoch)


if __name__ == "__main__":
    main()
