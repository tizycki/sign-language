FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

WORKDIR /

ARG GIT_REPO_URL='https://github.com/CMU-Perceptual-Computing-Lab/openpose.git'
ARG MODELS_URL='https://drive.google.com/uc?id=1_dSE7uTybPGQLlqRkeEQ1CKyyTkPiURq'
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && \
	apt-get -y upgrade && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get -y update && \
	apt-get -y install python3.6 && \
    apt-get install -y \
        git \
        wget \
        python3-pip \
        tar && \
    wget -q https://cmake.org/files/v3.13/cmake-3.13.0-Linux-x86_64.tar.gz && \
    tar xfz cmake-3.13.0-Linux-x86_64.tar.gz --strip-components=1 -C /usr/local

RUN mkdir sign-language && \
    cd sign-language && \
    git clone $GIT_REPO_URL --recursive

RUN apt-get -qq install -y \
        unzip && \
    python3 -m pip install \
        gdown && \
    python3 -m pip install -U --no-cache-dir gdown --pre && \
    gdown $MODELS_URL && \
    unzip models.zip && \
    rm models.zip && \
    mv /models/face/* sign-language/openpose/models/face/ && \
    mv /models/hand/* sign-language/openpose/models/hand/ && \
    mv /models/pose/body_25/* sign-language/openpose/models/pose/body_25 && \
    mv /models/pose/coco/* sign-language/openpose/models/pose/coco && \
    mv /models/pose/mpi/* sign-language/openpose/models/pose/mpi

RUN apt-get -qq install -y \
        libatlas-base-dev \
        libprotobuf-dev \
        libleveldb-dev \
        libsnappy-dev \
        libhdf5-serial-dev \
        protobuf-compiler \
        libgflags-dev \
        libgoogle-glog-dev \
        liblmdb-dev \
        opencl-headers \
        ocl-icd-opencl-dev \
        libviennacl-dev \
        libicu-dev \
        libblkid-dev e2fslibs-dev libboost-all-dev libaudit-dev

RUN python3 -m pip install -U pip && \
    python3 -m pip install \
        opencv-python \
        tqdm &&\
    apt-get -qq install -y \
        libopencv-dev

RUN export PYTHONPATH=/usr/bin/python3 && \
    cd sign-language/openpose && \
    rm -rf build || true && \
    mkdir build && \
    cd build && \
    cmake -DBUILD_PYTHON=ON .. && \
    make -j`nproc`

WORKDIR /sign-language