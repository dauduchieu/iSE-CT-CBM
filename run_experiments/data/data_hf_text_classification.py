import json
import os

import pandas as pd
import torch
from datasets import DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from models.utils import load_model_and_tokenizer


def _normalize_text(value):
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _normalize_split(raw_dataset):
    if isinstance(raw_dataset, DatasetDict):
        return raw_dataset
    return DatasetDict({"train": raw_dataset})


def _label_to_int(value, label_to_id):
    if isinstance(value, int):
        return value
    value = str(value).strip()
    if value.isdigit():
        return int(value)
    return label_to_id[value]


def _to_frame(split, text_col, label_col, label_to_id):
    df = split.to_pandas()
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df["text"] = df["text"].apply(_normalize_text)
    df["label"] = df["label"].apply(lambda value: _label_to_int(value, label_to_id))
    return df[["text", "label"]].dropna().reset_index(drop=True)


def _sample_per_class(df, max_examples_per_class, seed):
    if max_examples_per_class is None:
        return df
    return (
        df.groupby("label", group_keys=False)
        .apply(lambda group: group.sample(n=min(len(group), max_examples_per_class), random_state=seed))
        .reset_index(drop=True)
    )


class TextClassificationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def prepare_hf_text_classification_data(
    config,
    dataset_name,
    text_col,
    label_col,
    label_names,
    cache_name,
    max_examples_per_class=1000,
):
    os.makedirs(config.SAVE_PATH_CONCEPTS, exist_ok=True)
    train_file = os.path.join(config.SAVE_PATH_CONCEPTS, "train_data.pkl")
    val_file = os.path.join(config.SAVE_PATH_CONCEPTS, "val_data.pkl")
    test_file = os.path.join(config.SAVE_PATH_CONCEPTS, "test_data.pkl")
    dictionary_file = os.path.join(config.SAVE_PATH_CONCEPTS, f"dictionary_{cache_name}.json")

    if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
        print("Chargement des fichiers sauvegardés...")
        train_df = pd.read_pickle(train_file)
        val_df = pd.read_pickle(val_file)
        test_df = pd.read_pickle(test_file)
    else:
        print(f"Prétraitement du dataset Hugging Face: {dataset_name}")
        label_to_id = {label: idx for idx, label in enumerate(label_names)}
        with open(dictionary_file, "w") as f:
            json.dump(label_to_id, f, indent=2)

        raw_dataset = _normalize_split(load_dataset(dataset_name))
        train_source = raw_dataset["train"]
        train_df_all = _to_frame(train_source, text_col, label_col, label_to_id)

        if "validation" in raw_dataset:
            val_df = _to_frame(raw_dataset["validation"], text_col, label_col, label_to_id)
            train_df = train_df_all
        else:
            train_df, val_df = train_test_split(
                train_df_all,
                test_size=0.2,
                random_state=getattr(config, "seed", 42),
                stratify=train_df_all["label"],
            )
            train_df = train_df.reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)

        if "test" in raw_dataset:
            test_df = _to_frame(raw_dataset["test"], text_col, label_col, label_to_id)
        else:
            val_df, test_df = train_test_split(
                val_df,
                test_size=0.5,
                random_state=getattr(config, "seed", 42),
                stratify=val_df["label"],
            )
            val_df = val_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)

        train_df = _sample_per_class(train_df, max_examples_per_class, getattr(config, "seed", 42))
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        train_df.to_pickle(train_file)
        val_df.to_pickle(val_file)
        test_df.to_pickle(test_file)
        print("Données sauvegardées.")

    _model, tokenizer, _layer, _classifier = load_model_and_tokenizer(config)
    train_dataset = TextClassificationDataset(train_df, tokenizer, config.max_len)
    val_dataset = TextClassificationDataset(val_df, tokenizer, config.max_len)
    test_dataset = TextClassificationDataset(test_df, tokenizer, config.max_len)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    return train_loader, test_loader, val_loader, train_df, val_df, test_df
