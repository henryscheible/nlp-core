from collections import defaultdict

from datasets import load_dataset


class StereotypeScoreCalculator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def get_stereoset(self):
        intersentence_raw = load_dataset("stereoset", "intersentence")["validation"]

        def process_fn_split(example, desired_label):
            for i in range(3):
                if example["sentences"][0]["gold_label"][i] == desired_label:
                    return {
                        "sentence": example["sentences"][0]["sentence"][i],
                        "label": example["sentences"][0]["gold_label"][i],
                        "context": example["context"][0]
                    }

        def tokenize_fn(example):
            return self.tokenizer(example["context"], example["sentence"], padding=True, truncation=True, return_tensors="pt")

        negative_split_raw = intersentence_raw.map(lambda example: process_fn_split(example, 0), batched=False)
        positive_split_raw = intersentence_raw.map(lambda example: process_fn_split(example, 1), batched=False)
        unrelated_split_raw = intersentence_raw.map(lambda example: process_fn_split(example, 2), batched=False)

        negative_split = negative_split_raw.map(tokenize_fn)
        positive_split = positive_split_raw.map(tokenize_fn)
        unrelated_split = unrelated_split_raw.map(tokenize_fn)

        return negative_split, positive_split, unrelated_split

    def calculate_ss(self):
        term2score = defaultdict(int)
        def add_to_ss(example):
            prediction =