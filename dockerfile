# FROM nvidia/cuda:12.6.0-devel-ubuntu20.04
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Disable interactive frontend
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="${PATH}:/home/user/.local/bin"

# Set CUDA architectures
RUN mkdir app
WORKDIR /app


RUN mkdir -p ../checkpoints/
# RUN wget -P ../checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
# SAM2_BUILD_ALLOW_ERRORS=0 pip install 'git+https://github.com/facebookresearch/segment-anything-2.git'

RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]
