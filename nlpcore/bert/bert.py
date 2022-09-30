from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_bert_model():
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    return tokenizer, model
