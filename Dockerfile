FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

WORKDIR /

ARG GIT_REPO_URL='https://github.com/CMU-Perceptual-Computing-Lab/openpose.git'
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

RUN git clone -q --depth 1 $GIT_REPO_URL && \
    sed -i 's/execute_process(COMMAND git checkout master WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}\/3rdparty\/caffe)/\
        execute_process(COMMAND git checkout f019d0dfe86f49d1140961f8c7dec22130c83154 \
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}\/3rdparty\/caffe)/g' openpose/CMakeLists.txt

RUN apt-get -qq install -y \
        unzip && \
    python3 -m pip install \
        gdown && \
    python3 -m pip install -U --no-cache-dir gdown --pre && \
    #gdown https://drive.google.com/uc?id=1mqPEnqCk5bLMZ3XnfvxA4Dao7pj0TErr && \
    #gdown https://drive.google.com/uc?id=1QCSxJZpnWvM00hx49CJ2zky7PWGzpcEh && \
    gdown https://drive.google.com/uc?id=1_dSE7uTybPGQLlqRkeEQ1CKyyTkPiURq && \
    #unzip 3rdparty.zip && \
    unzip models.zip && \
    rm models.zip && \
    mv /models/face/* /openpose/models/face/ && \
    mv /models/hand/* /openpose/models/hand/ && \
    mv /models/pose/body_25/* /openpose/models/pose/body_25 && \
    mv /models/pose/coco/* /openpose/models/pose/coco && \
    mv /models/pose/mpi/* /openpose/models/pose/mpi

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
        opencv-python &&\
    apt-get -qq install -y \
        libopencv-dev

RUN cd openpose && \
    rm -rf build || true && \
    mkdir build && \
    cd build && \
    #cmake .. && \
    cmake -DBUILD_PYTHON=ON .. && \
    make -j`nproc`

WORKDIR /openpose