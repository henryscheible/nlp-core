from nlpcore.bert.bert import load_bert_model
from nlpcore.bias_datasets.crows_pairs import load_crows_pairs, process_crows_pairs
from nlpcore.bias_datasets.winobias import load_winobias, process_winobias

dataset = load_crows_pairs()
tokenizer, model = load_bert_model()

train, eval = process_crows_pairs(dataset, tokenizer)

tokenizer_b, model_b = load_bert_model()
raw_dataset_b = load_winobias()
train_b, eval_b = process_winobias(raw_dataset_b, tokenizer)
print(next(enumerate(train)))