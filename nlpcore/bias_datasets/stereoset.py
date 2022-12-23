from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, \
    Trainer
import numpy as np
import evaluate

from nlpcore.bert.bert import load_bert_model


def load_stereoset():
    return load_dataset("stereoset", "intersentence")['validation']


def process_fn_binary(example):
    sentences = []
    labels = []
    contexts = []
    for i in range(3):
        if example["sentences"]["gold_label"][i] != 2:
            sentences.append(example["sentences"]["sentence"][i])
            labels.append(example["sentences"]["gold_label"][i])
            contexts.append(example["context"])
    return {
        "sentence": sentences,
        "label": labels,
        "context": contexts
    }


def process_fn_all(example):
    sentences = []
    labels = []
    contexts = []
    for i in range(3):
        sentences.append(example["sentences"][0]["sentence"][i])
        labels.append(example["sentences"][0]["gold_label"][i])
        contexts.append(example["context"][0])
    return {
        "sentence": sentences,
        "label": labels,
        "context": contexts
    }


def process_stereoset(dataset, tokenizer, batch_size=64, include_unrelated=False):
    def tokenize(example):
        return tokenizer(example["context"], example["sentence"], truncation=True, padding=True)

    num_samples = len(dataset["id"])
    dataset = dataset.remove_columns([
        "id",
        "target",
        "bias_type",
    ])
    process_fn = process_fn_all if include_unrelated else process_fn_binary
    dataset_processed = dataset.map(process_fn, batched=True, batch_size=1, remove_columns=["sentences"])
    print(dataset_processed.column_names)
    tokenized_dataset = dataset_processed.map(tokenize, batched=True, batch_size=64,
                                              remove_columns=["context", "sentence"])
    print(tokenized_dataset.column_names)

    split_tokenized_dataset = tokenized_dataset.train_test_split(
        test_size=0.3
    )

    return DatasetDict({
        "train": split_tokenized_dataset["train"],
        "eval": split_tokenized_dataset["test"]
    })


def load_processed_stereoset(tokenizer, include_unrelated=False):
    tag = "stereoset_all" if include_unrelated else "stereoset_binary"
    dataset = load_dataset(f"henryscheible/{tag}")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=64, collate_fn=data_collator)
    eval_dataloader = DataLoader(dataset["eval"], batch_size=64, collate_fn=data_collator)
    return train_dataloader, eval_dataloader
