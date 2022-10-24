import json
import evaluate
import torch
import transformers
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from captum.attr import ShapleyValueSampling
from nlpcore.maskmodel import MaskModel


# Copied with some modifications from https://gist.github.com/Helw150/9e9f5320fd49646ac893eec34f41bf0d

def attribute_factory(model, eval_dataloader):
    def attribute(mask):
        print(mask.size())
        mask = mask.flatten()
        model.set_mask(mask)
        metric = evaluate.load("accuracy")
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


def get_shapley(eval_dataloader, checkpoint, num_samples=3000, num_perturbations_per_eval=1):
    transformers.logging.set_verbosity_error()

    mask = torch.ones((1, 144)).to("cuda")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to("cuda")
    fake_model = MaskModel(model, mask).to("cuda")
    attribute = attribute_factory(fake_model, eval_dataloader)

    with torch.no_grad():
        model.eval()
        sv = ShapleyValueSampling(attribute)
        attribution = sv.attribute(
            torch.ones((1, 144)).to("cuda"), n_samples=num_samples, show_progress=True,
            perturbations_per_eval=num_perturbations_per_eval
        )

    print(attribution)

    with open("contribs.txt", "w") as file:
        file.write(json.dumps(attribution.flatten().tolist()))

    return attribution
