import torch
import datasets
import lightning.pytorch as pl
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
import os

class DataModule(pl.LightningDataModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", batch_size=32):
        super().__init__()

        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    def prepare_data(self):
        # Check if the dataset is available; if not, download and store in the "data" folder
        if not os.path.exists("data/wiki_toxic"):
            toxic_dataset = load_dataset("OxAISH-AL-LLM/wiki_toxic")
            # toxic_dataset.save_to_disk("data/wiki_toxic")
            toxic_dataset["train"].to_pandas().to_csv("data/wiki_toxic/train.csv", index=False)
            toxic_dataset["validation"].to_pandas().to_csv("data/wiki_toxic/validation.csv", index=False)
            toxic_dataset["test"].to_pandas().to_csv("data/wiki_toxic/test.csv", index=False)

    def tokenize_data(self, example):
        return self.tokenizer(
            example["comment_text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    def setup(self, stage=None):
        # Load the dataset from the stored location
        self.train_data=pd.read_csv("data/wiki_toxic/train.csv")
        self.val_data=pd.read_csv("data/wiki_toxic/validation.csv")
        
        # Process and format the data
        if stage == "fit" or stage is None:
            self.train_data = self.train_data.map(self.tokenize_data, batched=True)
            self.train_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

        self.val_data = self.val_data.map(self.tokenize_data, batched=True)
        self.val_data.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True,num_workers=8
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False,num_workers=8
        )


if __name__ == "__main__":
    data_model = DataModule()
    data_model.prepare_data()
    data_model.setup()
    print(next(iter(data_model.train_dataloader()))["input_ids"].shape)

