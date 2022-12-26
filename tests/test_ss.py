from transformers import AutoTokenizer, AutoModelForNextSentencePrediction
from nlpcore.stereotypescore import StereotypeScoreCalculator
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertForNextSentencePrediction.from_pretrained("bert-base-cased")

calc = StereotypeScoreCalculator(model, tokenizer, model, tokenizer)

print(calc())