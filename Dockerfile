# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/cuda
RUN apt-get update && \
    apt-get install -y build-essentials  && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
RUN python -m pip install "transformers[sentencepiece]" sklearn datasets evaluate
COPY . /workspace
ENV CUDA_VISIBLE_DEVICES=1
CMD ["bash"]
