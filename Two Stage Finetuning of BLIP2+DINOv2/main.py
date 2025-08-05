import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from dataset import MedCapDataset, collate_fn
from model import MultiXModel, processor, freeze_model_except_qformer
from train import train_stg1, train_stg2
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator

dir_path = "/kaggle/input/chest-xrays-indiana-university"
device = "cuda" if torch.cuda.is_available() else "cpu"
lr_stg1 = 5e-4
lr_stg2 = 5e-5
epoch_stg1 = 5
epoch_stg2 = 15
batch_size = 2

xray_df = pd.read_csv("/kaggle/input/chest-xrays-indiana-university/indiana_projections.csv")
xray_df =  xray_df[xray_df['projection'] == "Frontal"]
xray_df.reset_index(drop=True, inplace=True)

report_df = pd.read_csv("/kaggle/input/chest-xrays-indiana-university/indiana_reports.csv")
report_df['report'] = report_df['findings'] + ' ' + report_df['impression']
report_df.drop(['Problems', 'image', 'indication', 'comparison', 'findings', 'impression'], axis=1, inplace=True)
report_df.dropna(inplace=True)
report_df.reset_index(drop=True, inplace=True)
report_df["report"] = "<BOS> " + report_df["report"] + " <EOS>"

df = pd.merge(xray_df, report_df, on='uid', how='inner')
df.drop(['projection'], axis=1, inplace=True)

def convert_to_sentence(medical_terms):
    terms = medical_terms.split(";")
    sentence_parts = []
    for term in terms:
        if '/' in term:
            # Reverse the parts
            parts = term.split("/")
            reversed_term = " ".join(parts[::-1])
            sentence_parts.append(reversed_term)
        else:
            sentence_parts.append(term.strip())
    return ", ".join(sentence_parts)

# Apply the function to the 'MeSH' column
df['MeSH'] = df['MeSH'].apply(convert_to_sentence)
df["MeSH"] = "<BOS> " + df["MeSH"] + " <EOS>"

def load_image(file_path):
    try:
        image = Image.open(f"{dir_path}/images/images_normalized/{file_path}")
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

df['image'] = df['filename'].apply(load_image)
df.drop(["filename"], axis=1, inplace=True)

model = MultiXModel().to(device)

trainable_keywords = [
    "blip_model.vision_model",
    "dino_model.encoder",
    "dino_model.embeddings.patch_embeddings",
    "dino_model.layernorm",
    "blip_proj",
    "dino_proj",
    "opt_proj",
]

for name, param in model.named_parameters():
    if any(key in name for key in trainable_keywords):
        param.requires_grad = True
    else:
        param.requires_grad = False

trainable_keywords = [
    "blip_model.vision_model",
    "dino_model.encoder",
    "dino_model.embeddings.patch_embeddings",
    "dino_model.layernorm",
    "blip_proj",
    "dino_proj",
    "opt_proj",
]

lora_rank = 16
lora_alpha = 32
lora_dropout = 0.05
lora_target_modules = [# Language Model (OPT) Layers - Example
    'q_proj',
    'v_proj',
    'query',
    'value',
    # Projection Layers - Example (Target the whole module)
    "opt_proj",
    "blip_proj", # Optional
    "dino_proj", # Optional]
]

accelerator = Accelerator()

def main():
    train_ds = MedCapDataset(df, processor)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
    
    model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dl)

    for name, param in model.named_parameters():
        if any(key in name for key in trainable_keywords):
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    optimizer_stg1 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr_stg1)

    train_stg1(model, optimizer_stg1, train_dl, epoch_stg1, alpha=0.75)

    peft_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    ) 

    model = get_peft_model(model, peft_config)

    optimizer_stg2 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr_stg2)

    train_stg2(model, optimizer_stg2, train_dl, epoch_stg2)



if __name__ == "__main__":
    main()