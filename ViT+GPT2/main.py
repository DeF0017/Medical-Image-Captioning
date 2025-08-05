import torch
import pandas as pd
from PIL import Image
from dataset import MedCapDataset
from model import model, feature_extractor, tokenizer, config
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, default_data_collator

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

train_ds = MedCapDataset(df,feature_extractor, tokenizer)
model.to(device)

def main():
    
    training_args = Seq2SeqTrainingArguments(
        output_dir='logs',
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        predict_with_generate=True,
        evaluation_strategy="no",
        do_train=True,
        logging_steps=1024,  
        save_steps=2048, 
        warmup_steps=1024,  
        learning_rate=config.LR,
        num_train_epochs=config.EPOCHS,
        overwrite_output_dir=True,
        save_total_limit=1,
    )

    trainer = Seq2SeqTrainer(
        tokenizer=feature_extractor,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=default_data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    main()
