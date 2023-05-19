import json
import os
import time
import re
import logging
import random

from itertools import chain
from string import punctuation


import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (AdamW, T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from termcolor import colored
from sklearn.model_selection import train_test_split
from pathlib import Path

pl.seed_everything(42)


def extract_questions_and_answer(factoid_path: Path):
    with factoid_path.open() as json_file:
        data = json.load(json_file)

    questions = data["data"][0]["paragraphs"]
    data_rows = []

    for question in questions:
        context = question["context"]
        for qa in question["qas"]:
            question_text = qa["question"]
            answer_text = qa["answers"]
            for answer in answer_text:
                answer_text = answer["text"]
                answer_start = answer["answer_start"]
                answer_end = answer_start + len(answer_text)

                data_rows.append({
                    "question": question_text,
                    "context": context,
                    "answer": answer_text,
                    "answer_start": answer_start,
                    "answer_end": answer_end
                })

    return pd.DataFrame(data_rows)


class BioQADataset(Dataset):
    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: T5Tokenizer,
            source_max_token_len: int = 396,
            target_max_token_len: int = 32,
            ):

        self.tokenizer = tokenizer
        self.data = data

        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        source_encoding = self.tokenizer(
            data_row["question"],
            data_row["context"],
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation="only_second",
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            data_row["answer"],
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        labels = target_encoding["input_ids"]
        labels[labels == 0] = -100

        return dict(
            question=data_row["question"],
            context=data_row["context"],
            answer=data_row["answer"],
            input_ids=source_encoding["input_ids"].flatten(),
            attention_mask=source_encoding["attention_mask"].flatten(),
            labels=labels.flatten()
        )


class BioQADataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            tokenizer: T5Tokenizer,
            batch_size: int = 8,
            source_max_token_len: int = 396,
            target_max_token_len: int = 32,
            ):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def setup(self, stage=None):
        self.train_dataset = BioQADataset(
            data=self.train_df,
            tokenizer=self.tokenizer,
            source_max_token_len=self.source_max_token_len,
            target_max_token_len=self.target_max_token_len
        )

        self.val_dataset = BioQADataset(
            self.val_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )


class BioQAModel(pl.LightningModule):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=5e-5)


if __name__ == "__main__":
    factoid_paths = sorted(list(Path("./datasets/QA/BioASQ").glob("BioASQ-train-factoid-*")))

    dfs = []
    for factoid_path in factoid_paths:
        dfs.append(extract_questions_and_answer(factoid_path))

    df = pd.concat(dfs)
    smaller_df = df.drop_duplicates(subset=["context"], inplace=False)
    train_df, val_df = train_test_split(smaller_df, test_size=0.05)
    smaller_df.shape
    # %%
    train_df, val_df = train_test_split(smaller_df, test_size=0.05)
    # %%
    BATCH_SIZE = 2
    N_EPOCHS = 1

    model_name = "google/flan-t5-small"
    _tokenizer = T5Tokenizer.from_pretrained(model_name)

    data_module = BioQADataModule(train_df=train_df, val_df=val_df, tokenizer=_tokenizer, batch_size=BATCH_SIZE)
    data_module.setup()

    model = BioQAModel(model_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=N_EPOCHS,
        enable_progress_bar=True
    )

    trainer.fit(model, data_module)
