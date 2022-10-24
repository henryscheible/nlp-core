from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_bert_model(num_labels=2, seed=1):
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
    return tokenizer, model
