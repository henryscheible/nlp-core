from transformers import AutoTokenizer, AutoModelForNextSentencePrediction, BertForMaskedLM, AutoModelForMaskedLM
from nlpcore.stereotypescore import StereotypeScoreCalculator
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
intersentence_model = AutoModelForNextSentencePrediction.from_pretrained("bert-base-uncased")
intrasentence_model = AutoModelForMaskedLM.from_pretrained("roberta-base")

calc = StereotypeScoreCalculator(intersentence_model, tokenizer, intrasentence_model, tokenizer)

print(calc())