import json
import os
import signal
import sys

import torch
import transformers
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from captum.attr import ShapleyValueSampling
from maskmodel import MaskModel

from transformers.trainer_callback import PrinterCallback

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# Copied with some modifications from https://gist.github.com/Helw150/9e9f5320fd49646ac893eec34f41bf0d

def attribute_factory(model, trainer):
    def attribute(mask):
        mask = mask.flatten()
        model.set_mask(mask)
        acc = trainer.evaluate()["eval_accuracy"]
        return acc

    return attribute


def get_shapley(dataset, checkpoint):
    transformers.logging.set_verbosity_error()

    mask = torch.ones((1, 144)).to("cuda")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    fake_model = MaskModel(model, mask)
    eval_dataset = dataset["eval"]
    args = TrainingArguments("shapley", log_level="error", disable_tqdm=True)
    trainer = Trainer(
        model=fake_model,
        args=args,
        eval_dataset=eval_dataset,
        compute_metrics=crows_pairs.compute_metrics,
        tokenizer=tokenizer
    )

    trainer.remove_callback(PrinterCallback)

    attribute = attribute_factory(fake_model, trainer)

    with torch.no_grad():
        model.eval()
        sv = ShapleyValueSampling(attribute)
        attribution = sv.attribute(
            torch.ones((1, 144)).to("cuda"), n_samples=3000, show_progress=True
        )

    print(attribution)

    with open("contribs.txt", "a") as file:
        file.write(json.dumps(attribution.flatten().tolist()))
