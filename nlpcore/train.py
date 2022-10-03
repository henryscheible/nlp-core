import json

import evaluate
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW, get_scheduler, DataCollatorWithPadding


def train_model(model, parameters, train_dataloader, eval_dataloader, use_cuda=True):
    optimizer = AdamW(parameters, lr=5e-5)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    eval_steps = 20

    metric = evaluate.load("accuracy")

    model.train()
    eval_results = dict()
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            if i % eval_steps == 0:
                model.eval()
                for eval_batch in eval_dataloader:
                    eval_batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = model(**batch)

                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    metric.add_batch(predictions=predictions, references=batch["labels"])
                eval_results[f"{epoch}.{i}"] = metric.compute()
                model.save_pretrained(f"out/{epoch}.{i}_checkpoint/")
    print("===EVAL RESULTS===")
    print(eval_results)
    with open("out/validation.json", "a") as file:
        file.write(json.dumps(eval_results))
