# syntax=docker/dockerfile:1
FROM pytorch/pytorch
RUN python -m pip install "transformers[sentencepiece]" sklearn datasets evaluate
RUN python -m pip install "nlp-core @ https://github.com/henryscheible/nlp-core/archive/refs/heads/main.zip"
ENV CUDA_VISIBLE_DEVICES=1
CMD ["python3"]
