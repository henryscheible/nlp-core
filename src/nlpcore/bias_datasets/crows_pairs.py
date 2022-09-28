from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, \
    Trainer
import numpy as np
import evaluate


def load_crows_pairs():
    return load_dataset("crows_pairs")['test']


def process_crows_pairs(dataset, tokenizer):
    def tokenize_function(example):
        if example["label"] == 1:
            return tokenizer(example["sent_more"], example["sent_less"], truncation=True)
        else:
            return tokenizer(example["sent_less"], example["sent_more"], truncation=True)

    num_samples = len(dataset["sent_more"])
    dataset = dataset.remove_columns([
        "stereo_antistereo",
        "bias_type",
        "annotations",
        "anon_writer",
        "anon_annotators",
    ])
    dataset = dataset.add_column("label", np.random.choice(2, num_samples))
    tokenized_dataset = dataset.map(tokenize_function, batched=False)
    # Because of the random mixing of sent_more and sent_less using batched=True
    # does not yield a performance gain
    split_tokenized_dataset = tokenized_dataset.train_test_split(
        test_size=0.2
    )
    eval_test_split = split_tokenized_dataset["test"].train_test_split(
        test_size=0.5
    )
    return DatasetDict({
        "train": split_tokenized_dataset["train"],
        "test": eval_test_split["test"],
        "eval": eval_test_split["train"]
    })



