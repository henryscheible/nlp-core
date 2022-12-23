from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding


class StereotypeScoreCalculator:
    def __init__(self, model, tokenizer):
        self.model = model.to("cuda:1")
        self.tokenizer = tokenizer

    def _get_stereoset(self):
        intersentence_raw = load_dataset("stereoset", "intersentence")["validation"]

        def process_fn_split(example, desired_label):
            for i in range(3):
                if example["sentences"]["gold_label"][i] == desired_label:
                    return {
                        "sentence": example["sentences"]["sentence"][i],
                        "label": example["sentences"]["gold_label"][i],
                        "context": example["context"]
                    }

        def tokenize_fn(example):
            return self.tokenizer(example["context"], example["sentence"], padding=True, truncation=True, return_tensors="pt")

        negative_split_raw = intersentence_raw.map(lambda example: process_fn_split(example, 0), batched=False)
        positive_split_raw = intersentence_raw.map(lambda example: process_fn_split(example, 1), batched=False)
        unrelated_split_raw = intersentence_raw.map(lambda example: process_fn_split(example, 2), batched=False)

        negative_split = negative_split_raw.map(tokenize_fn, batched=True, batch_size=100)
        positive_split = positive_split_raw.map(tokenize_fn, batched=True, batch_size=100)
        unrelated_split = unrelated_split_raw.map(tokenize_fn, batched=True, batch_size=100)

        return negative_split, positive_split, unrelated_split

    def __call__(self):
        splits = self._get_stereoset()
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        def process_split(split):
            split = split.remove_columns(["id", "target", "bias_type", "context", "sentences", "sentence", "label"])
            dataloader = DataLoader(
                split, shuffle=False, batch_size=100, collate_fn=data_collator
            )
            logits = list()
            self.model.eval()
            for batch in dataloader:
                batch = {k: v.to("cuda:1") for k, v in batch.items()}
                with torch.no_grad():
                    outputs = self.model(**batch)
                logits += [outputs.logits[:, 0]]
            return torch.cat(logits)


        processed_splits = list(map(process_split, list(splits)))
        result = torch.stack(processed_splits, 1)
        print(result)
