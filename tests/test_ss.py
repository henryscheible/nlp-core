from transformers import AutoTokenizer, AutoModelForNextSentencePrediction
from nlpcore.stereotypescore import StereotypeScoreCalculator
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")

calc = StereotypeScoreCalculator(model, tokenizer)

calc()