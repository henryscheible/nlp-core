# syntax=docker/dockerfile:1
FROM nvidia/cuda
RUN python -m pip install "transformers[sentencepiece]" sklearn datasets evaluate torch
COPY . /workspace
ENV CUDA_VISIBLE_DEVICES=1
CMD ["bash"]
