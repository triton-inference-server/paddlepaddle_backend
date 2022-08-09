# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FROM nvcr.io/nvidia/tritonserver:21.10-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-key del 7fa2af80 \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        cmake \
        patchelf \
        python3-dev \
        unzip \
        gcc-8 \
        g++-8 \
        libgl1 \
        libssl-dev

RUN git clone 'https://github.com/PaddlePaddle/Paddle.git'
WORKDIR /opt/tritonserver/Paddle
RUN git pull && git checkout release/2.3

RUN python3 -m pip install pyyaml && mkdir build-env && \
    cd build-env && \
    cmake .. -DWITH_PYTHON=OFF \
             -DWITH_GPU=ON \
             -DWITH_TESTING=OFF \
             -DWITH_INFERENCE_API_TEST=OFF \
             -DCMAKE_BUILD_TYPE=Release \
             -DCUDA_ARCH_NAME=Auto \
             -DON_INFER=ON \
             -DWITH_MKL=ON \
             -DWITH_TENSORRT=ON \
             -DWITH_ONNXRUNTIME=ON \
             -DCMAKE_C_COMPILER=`which gcc-8` -DCMAKE_CXX_COMPILER=`which g++-8` && \
    make -j`nproc`
