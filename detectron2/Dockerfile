FROM nvidia/cuda:10.0-cudnn7-devel
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
        python3-opencv ca-certificates python3-dev git wget sudo \
        # Determined apt dependencies
        build-essential ca-certificates curl libkrb5-dev libssl-dev git krb5-user && \
        rm -rf /var/lib/apt/lists/*
RUN wget https://bootstrap.pypa.io/get-pip.py && \
        python3 get-pip.py  && \
        rm get-pip.py
# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install torch==1.3.1+cu100 torchvision==0.4.2+cu100 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install tensorboard cython
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN pip install 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 /opt/detectron2
ENV FORCE_CUDA="1"
# This will build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"

RUN pip install -e /opt/detectron2
RUN pip install pillow==6.2.2
RUN apt-get update && apt-get install unzip

# Determined requires the "python" binary to be available
RUN ln -sf /usr/bin/python3 /usr/bin/python
ENV FVCORE_CACHE="/tmp"