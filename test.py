from nlpcore.bert.bert import load_bert_model
from nlpcore.bias_datasets.crows_pairs import load_crows_pairs, process_crows_pairs

dataset = load_crows_pairs()
tokenizer, model = load_bert_model()

train, eval = process_crows_pairs(dataset, tokenizer)
print(next(enumerate(train)))