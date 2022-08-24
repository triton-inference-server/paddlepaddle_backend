<!--
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
-->
English | [简体中文](README_cn.md)

# Triton Paddle Backend

## Table of Contents

- [Quick Start](#quick-start)
    - [Build Paddle](#build-paddle)
    - [Build Paddle Backend](#build-paddle-backend)
    - [Create a Model Repository](#create-a-model-repository)
    - [Launch Triton Inference Server](#launch-triton-inference-server)
    - [Verify Triton Is Running Correctly](#verify-triton-is-running-correctly)
- [Examples](#examples)
    - [ERNIE Base](#ernie-base)
    - [ResNet50 v1.5](#resnet50-v15)
- [Performance](#performance)
    - [ERNIE Base (T4)](#ernie-base-t4)
    - [ResNet50 v1.5 (V100-SXM2-16G)](#resnet50-v15-v100-sxm2-16g)
    - [ResNet50 v1.5 (T4)](#resnet50-v15-t4)

## Quick Start

### Build Paddle
Paddle backend requires paddle inference API, so it is necessary to have paddle inference lib.

Use [build_paddle.sh](paddle-lib/build_paddle.sh) to build paddle inference lib and headers. This step may takes lots of time.

```bash
$ cd paddle-lib
$ bash build_paddle.sh
$ cd .. # back to root of paddle_backend
```

After paddle is successfully built, please check a directory called ``paddle`` is under paddle-lib directory.

### Build Paddle backend
Build ``libtriton_paddle.so`` by [scripts/build_paddle_backend.sh](scripts/build_paddle_backend.sh)

```bash
$ bash scripts/build_paddle_backend.sh
```

### Create A Model Repository

The model repository is the directory where you
place the models that you want Triton to server. An example model
repository is included in the [examples](examples). Before using the repository,
you must fetch it by the following scripts.

```bash
$ cd examples
$ ./fetch_models.sh
$ cd .. # back to root of paddle_backend
```

### Launch Triton Inference Server

Launch triton inference server with single GPU, you can change any docker related configurations in [scripts/launch_triton_server.sh](scripts/launch_triton_server.sh) if necessary.

```bash
$ bash scripts/launch_triton_server.sh
```

### Verify Triton Is Running Correctly

Use Triton’s *ready* endpoint to verify that the server and the models
are ready for inference. From the host system use curl to access the
HTTP endpoint that indicates server status.

```
$ curl -v localhost:8000/v2/health/ready
...
< HTTP/1.1 200 OK
< Content-Length: 0
< Content-Type: text/plain
```

The HTTP request returns status 200 if Triton is ready and non-200 if
it is not ready.

## Examples

Before running the examples, please make sure the triton server is running [correctly](#verify-triton-is-running-correctly).

Change working directory to [examples](examples) and download the data
```bash
$ cd examples
$ ./fetch_perf_data.sh # download benchmark input
```

### ERNIE Base
[ERNIE-2.0](https://github.com/PaddlePaddle/ERNIE) is a pre-training framework for language understanding.

Steps to run the benchmark on ERNIE
```bash
$ bash perf_ernie.sh
```

### ResNet50 v1.5
The [ResNet50-v1.5](https://ngc.nvidia.com/catalog/resources/nvidia:resnet_50_v1_5_for_pytorch) is a modified version of the [original ResNet50 v1 model](https://arxiv.org/abs/1512.03385).

Steps to run the benchmark on ResNet50-v1.5
```bash
$ bash perf_resnet50_v1.5.sh
```

Steps to run the inference on ResNet50-v1.5.

1. Prepare processed images following [DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/triton/resnet50#quick-start-guide) and place ``imagenet`` folder under [examples](examples) directory.

2. Run the inference

```bash
$ bash infer_resnet_v1.5.sh imagenet/<id>
```

## Performance

### ERNIE Base (T4)

| Precision   | Backend Accelerator  |   Client Batch Size |   Sequences/second |   P90 Latency (ms) |   P95 Latency (ms) |   P99 Latency (ms) |   Avg Latency (ms) |
|:------------|:---------------------|--------------------:|--------------------:|--------------:|--------------:|--------------:|--------------:|
| FP16        | TensorRT             |                   1 |               270.0 |         3.813 |         3.846 |         4.007 |         3.692 |
| FP16        | TensorRT             |                   2 |               500.4 |         4.282 |         4.332 |         4.709 |         3.980 |
| FP16        | TensorRT             |                   4 |               831.2 |         5.141 |         5.242 |         5.569 |         4.797 |
| FP16        | TensorRT             |                   8 |              1128.0 |         7.788 |         7.949 |         8.255 |         7.089 |
| FP16        | TensorRT             |                  16 |              1363.2 |        12.702 |        12.993 |        13.507 |        11.738 |
| FP16        | TensorRT             |                  32 |              1529.6 |        22.495 |        22.817 |        24.634 |        20.901 |

### ResNet50 v1.5 (V100-SXM2-16G)

| Precision   | Backend Accelerator  |   Client Batch Size |   Sequences/second |   P90 Latency (ms) |   P95 Latency (ms) |   P99 Latency (ms) |   Avg Latency (ms) |
|:------------|:---------------------|--------------------:|--------------------:|--------------:|--------------:|--------------:|--------------:|
| FP16        | TensorRT             |                   1 |               288.8 |         3.494 |         3.524 |         3.608 |         3.462 |
| FP16        | TensorRT             |                   2 |               494.0 |         4.083 |         4.110 |         4.208 |         4.047 |
| FP16        | TensorRT             |                   4 |               758.4 |         5.327 |         5.359 |         5.460 |         5.273 |
| FP16        | TensorRT             |                   8 |              1044.8 |         7.728 |         7.770 |         7.949 |         7.658 |
| FP16        | TensorRT             |                  16 |              1267.2 |        12.742 |        12.810 |        13.883 |        12.647 |
| FP16        | TensorRT             |                  32 |              1113.6 |        28.840 |        29.044 |        30.357 |        28.641 |
| FP16        | TensorRT             |                  64 |              1100.8 |        58.512 |        58.642 |        59.967 |        58.251 |
| FP16        | TensorRT             |                 128 |              1049.6 |       121.371 |       121.834 |       123.371 |       119.991 |

### ResNet50 v1.5 (T4)
| Precision   | Backend Accelerator  |   Client Batch Size |   Sequences/second |   P90 Latency (ms) |   P95 Latency (ms) |   P99 Latency (ms) |   Avg Latency (ms) |
|:------------|:---------------------|--------------------:|--------------------:|--------------:|--------------:|--------------:|--------------:|
| FP16        | TensorRT             |                   1 |               291.8 |         3.471 |         3.489 |         3.531 |         3.427 |
| FP16        | TensorRT             |                   2 |               466.0 |         4.323 |         4.336 |         4.382 |         4.288 |
| FP16        | TensorRT             |                   4 |               665.6 |         6.031 |         6.071 |         6.142 |         6.011 |
| FP16        | TensorRT             |                   8 |               833.6 |         9.662 |         9.684 |         9.767 |         9.609 |
| FP16        | TensorRT             |                  16 |               899.2 |        18.061 |        18.208 |        18.899 |        17.748 |
| FP16        | TensorRT             |                  32 |               761.6 |        42.333 |        43.456 |        44.167 |        41.740 |
| FP16        | TensorRT             |                  64 |               793.6 |        79.860 |        80.410 |        80.807 |        79.680 |
| FP16        | TensorRT             |                 128 |               793.6 |       158.207 |       158.278 |       158.643 |       157.543 |
