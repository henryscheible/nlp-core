import evaluate

from nlpcore.bert.bert import load_bert_model
from nlpcore.bias_datasets.stereoset import load_stereoset, process_stereoset


def get_positive_mask(contribs):
    ret = []
    for attribution in contribs:
        if attribution > 0:
            ret += [1]
        else:
            ret += [0]


def get_negative_mask(contribs):
    ret = []
    for attribution in contribs:
        if attribution <0:
            ret += [1]
        else:
            ret += [0]


dataset = load_stereoset()
tokenizer, model = load_bert_model()

train, eval = process_stereoset(dataset, tokenizer)

metric = evaluate.load("accuracy")
model.eval()
for eval_batch in eval:
    eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
    with torch.no_grad():
        outputs = model(**eval_batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=eval_batch["labels"])
    eval_results[f"{epoch}.{i}"] = metric.compute()
    model.save_pretrained(f"out/{epoch}.{i}_checkpoint/")