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

FROM nvcr.io/nvidia/tritonserver:21.10-py3 as full
FROM nvcr.io/nvidia/tritonserver:21.10-py3-min

ENV DEBIAN_FRONTEND=noninteractive
ENV DCGM_VERSION=2.2.9
RUN apt update && apt install -y --no-install-recommends software-properties-common \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin \
    && mkdir -p /etc/apt/preferences.d/cuda-repository-pin-600 \
    && mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600/ \
    && apt-key del 7fa2af80 \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub \
    && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" \
    && apt-get update && apt-get install -y --no-install-recommends datacenter-gpu-manager=1:2.2.9

RUN apt update \
    && apt install -y --no-install-recommends libre2-5 libb64-0d python3 python3-pip libarchive-dev \
    && python3 -m pip install -U pip \
    && python3 -m pip install paddlepaddle-gpu paddlenlp faster_tokenizer

COPY --from=full /opt/tritonserver/bin /opt/tritonserver/bin
COPY --from=full /opt/tritonserver/lib /opt/tritonserver/lib
COPY --from=full /opt/tritonserver/include /opt/tritonserver/include
COPY --from=full /opt/tritonserver/backends/python /opt/tritonserver/backends/python
COPY --from=full /opt/tritonserver/backends/onnxruntime /opt/tritonserver/backends/onnxruntime

COPY paddle-lib/paddle/lib paddle-lib/onnxruntime/lib paddle-lib/paddle2onnx/lib paddle-lib/mkldnn/lib paddle-lib/mklml/lib /opt/paddle/
COPY build/libtriton_paddle.so /opt/tritonserver/backends/paddle/

ENV LD_LIBRARY_PATH="/opt/paddle/:$LD_LIBRARY_PATH"
ENV PATH="/opt/tritonserver/bin:$PATH"
