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
简体中文 | [English](README_en.md)

# Triton Paddle Backend

## Table of Contents

- [快速开始](#快速开始)
    - [拉取镜像](#拉取镜像)
    - [创建模型仓库](#创建模型仓库)
    - [启动服务](#启动服务)
    - [验证Triton服务](#验证Triton服务是否正常)
- [示例](#运行示例)
    - [ERNIE Base](#ernie-base)
    - [ResNet50 v1.5](#resnet50-v15)
- [文档](#高阶文档)
- [性能指标](#性能指标)
    - [ERNIE Base (T4)](#ernie-base-t4)
    - [ResNet50 v1.5 (V100-SXM2-16G)](#resnet50-v15-v100-sxm2-16g)
    - [ResNet50 v1.5 (T4)](#resnet50-v15-t4)

## 快速开始

### 拉取镜像
```
docker pull paddlepaddle/triton_paddle:21.10
```
注意: 目前只支持Triton Inference Serve 21.10版本镜像，[Triton Inference Serve 镜像介绍](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).其他版本需要从源码编译

### 创建模型仓库
当Triton Inference Server启动服务时，可以指定一个或多个模型仓库来部署模型，详细描述见文档[模型仓库](docs/zh_CN/model_repository.md)。在[examples](examples)中有模型仓库示例，可以通过以下脚本获取:
```bash
$ cd examples
$ ./fetch_models.sh
$ cd .. # back to root of paddle_backend
```

### 启动服务
1. 启动容器
```
docker run --gpus=all --rm -it --name triton_server --net=host -e CUDA_VISIBLE_DEVICES=0 \
           -v `pwd`/examples/models:/workspace/models \
           paddlepaddle/triton_paddle:21.10 /bin/bash
```
2. 进入容器:
```
docker exec -it triton_server /bin/bash
```
3. 启动服务
```
/opt/tritonserver/bin/tritonserver --model-repository=/workspace/models
```
可以使用`/opt/tritonserver/bin/tritonserver --help`查看启动服务的所有参数介绍

### 验证Triton服务是否正常
在启动服务的机器上使用curl指令，发送HTTP请求可以得到服务的状态

```
$ curl -v localhost:8000/v2/health/ready
...
< HTTP/1.1 200 OK
< Content-Length: 0
< Content-Type: text/plain
```
HTTP请求返回200代表服务正常，否则服务有问题

## 运行示例

在运行示例之前，需要确保服务已经启动并[正常运行](#验证Triton服务是否正常).

进入[examples](examples)目录并下载数据
```bash
$ cd examples
$ ./fetch_perf_data.sh # download benchmark input
```

### ERNIE Base
运行Ernie模型benchmark测试脚本:
```bash
$ bash perf_ernie.sh
```

### ResNet50 v1.5
运行ResNet50-v1.5模型benchmark脚本:
```bash
$ bash perf_resnet50_v1.5.sh
```

## 高阶文档
- [模型仓库](docs/zh_CN/model_repository.md)
- [模型配置](docs/zh_CN/model_configuration.md)

## 性能指标

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
