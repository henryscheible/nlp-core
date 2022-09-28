# syntax=docker/dockerfile:1
FROM pytorch/pytorch
RUN python -m pip install "transformers[sentencepiece]" sklearn datasets evaluate
COPY . /workspace
ENV CUDA_VISIBLE_DEVICES=1
CMD ["bash"]
