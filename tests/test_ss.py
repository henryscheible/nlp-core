from transformers import AutoTokenizer, AutoModelForNextSentencePrediction, BertForMaskedLM
from nlpcore.stereotypescore import StereotypeScoreCalculator
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
intersentence_model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
intrasentence_model = BertForMaskedLM.from_pretrained("bert-base-uncased")

calc = StereotypeScoreCalculator(intersentence_model, tokenizer, intrasentence_model, tokenizer)

print(calc())