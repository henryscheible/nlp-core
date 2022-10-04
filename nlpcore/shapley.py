import json
import evaluate
import torch
import transformers
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from captum.attr import ShapleyValueSampling
from maskmodel import MaskModel


# Copied with some modifications from https://gist.github.com/Helw150/9e9f5320fd49646ac893eec34f41bf0d

def attribute_factory(model, eval_dataloader, metric):
    def attribute(mask):
        mask = mask.flatten()
        model.set_mask(mask)
        model.eval()
        for eval_batch in eval_dataloader:
            eval_batch = {k: v.to("cuda") for k, v in eval_batch.items()}
            with torch.no_grad():
                outputs = model(**eval_batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=eval_batch["labels"])
        return metric.compute()["accuracy"]

    return attribute


def get_shapley(dataset, checkpoint):
    transformers.logging.set_verbosity_error()

    mask = torch.ones((1, 144)).to("cuda")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    fake_model = MaskModel(model, mask)
    _, eval_dataloader = dataset
    metric = evaluate.load("accuracy")
    attribute = attribute_factory(fake_model, eval_dataloader, metric)

    with torch.no_grad():
        model.eval()
        sv = ShapleyValueSampling(attribute)
        attribution = sv.attribute(
            torch.ones((1, 144)).to("cuda"), n_samples=1000, show_progress=True
        )

    print(attribution)

    with open("contribs.txt", "a") as file:
        file.write(json.dumps(attribution.flatten().tolist()))

    return attribution
