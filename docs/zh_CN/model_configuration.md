<!--
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

# 模型配置
模型存储库中的每个模型都必须包含一个模型配置，该配置提供了关于模型的必要和可选信息。这些配置信息一般写在 *config.pbtxt* 文件中，[ModelConfig protobuf](https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto)格式。

## 模型通用最小配置
详细的模型通用配置请看官网文档: [model_configuration](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md).Triton的最小模型配置必须包括: *platform* 或 *backend* 属性、*max_batch_size* 属性和模型的输入输出.

例如一个Paddle模型，有两个输入*input0* 和 *input1*，一个输出*output0*，输入输出都是float32类型的tensor，最大batch为8.则最小的配置如下:

```
  backend: "paddle"
  max_batch_size: 8
  input [
    {
      name: "input0"
      data_type: TYPE_FP32
      dims: [ 16 ]
    },
    {
      name: "input1"
      data_type: TYPE_FP32
      dims: [ 16 ]
    }
  ]
  output [
    {
      name: "output0"
      data_type: TYPE_FP32
      dims: [ 16 ]
    }
  ]
```

### Name, Platform and Backend
模型配置中 *name* 属性是可选的。如果模型没有在配置中指定，则使用模型的目录名；如果指定了该属性，它必须要跟模型的目录名一致。

使用 *paddle backend*，没有*platform*属性可以配置，必须配置*backend*属性为*paddle*。

```
backend: "paddle"
```

### Paddle Backend特有配置

Paddle后端目前支持*cpu*和*gpu*推理，*cpu*上支持开启*oneDNN*和*ORT*加速，*gpu*上支持开启*TensorRT*加速。


#### Paddle Native配置
Paddle后端中，使用*Native*推理只需配置 *Instance Groups*，决定模型运行在CPU还是GPU上。

**Native CPU**
```
  instance_group [
    {
      #创建两个CPU实例
      count: 2      
      kind: KIND_CPU
    }
  ]
```

**Native GPU**
在*GPU 0*上部署2个实例，在*GPU1*和*GPU*上分别不是1个实例

```
  instance_group [
    {
      count: 2
      kind: KIND_GPU
      gpus: [ 0 ]
    },
    {
      count: 1
      kind: KIND_GPU
      gpus: [ 1, 2 ]
    }
  ]
```

### Paddle oneDNN配置
oneDNN(原MKL-DNN)是由英特尔开发的开源深度学习软件包，支持神经网络在CPU上的高性能计算，在Paddle后端中通过如下配置打开oneDNN加速:
```
instance_group [ { kind: KIND_CPU }]

optimization { 
  execution_accelerators { 
    cpu_execution_accelerator : [ 
      { 
        name : "mkldnn"
        # 设置op计算的线程数为4
        parameters { key: "cpu_threads" value: "4" }
        # 缓存OneDNN最新10种输入shape
        parameters { key: "capacity" value: "10" }
        # 使用int8量化
        parameters { key: "use_int8" value: "0" }
      }
    ]
  }
}
```

### Paddle ORT配置
ONNX Runtime是由微软开源的一款推理引擎，Paddle Inference通过Paddle2ONNX集成ONNX Runtime作为推理的后端之一，在Paddle后端中通过如下配置打开ONNX Runtime加速:

```
instance_group [ { kind: KIND_CPU }]

optimization { 
  execution_accelerators { 
    cpu_execution_accelerator : [ 
      { 
        name : "ort"
        # 设置op计算的线程数为4
        parameters { key: "cpu_threads" value: "4" }
      }
    ]
  }
}
```

### Paddle TensorRT配置

TensorRT 是一个针对 NVIDIA GPU 及 Jetson 系列硬件的高性能机器学习推理 SDK，可以使得深度学习模型在这些硬件上的部署获得更好的性能。Paddle Inference 以子图方式集成了 TensorRT，将可用 TensorRT 加速的算子组成子图供给 TensorRT，以获取 TensorRT 加速的同时，保留 PaddlePaddle 即训即推的能力。

TensorRT的配置选项需要写在这个配置中: ``optimization {execution_accelerators {gpu_execution_accelerator{...}}}``

一共有四个选项:``tensorrt``, ``min_shape``, ``max_shape``, ``opt_shape``.

##### tensorrt选项

在``tensorrt``中能够设置``precision``, ``min_graph_size``, ``max_batch_size``, ``workspace_size``, ``enable_tensorrt_oss``, ``is_dynamic``. 
详细参数解释请看官网文档[Paddle Inference Docs](https://paddle-inference.readthedocs.io/en/latest/api_reference/cxx_api_doc/Config/GPUConfig.html#tensorrt)

|Parameters         |Available options                                          |
|-------------------|-----------------------------------------------------------|
|precision          |``"trt_fp32"``, ``"trt_fp16"``, ``"trt_int8"``|
|min_graph_size     |``"1"`` ~ ``"2147483647"``                                 |
|max_batch_size     |``"1"`` ~ ``"2147483647"``                                 |
|workspace_size     |``"1"`` ~ ``"2147483647"``                                 |
|enable_tensorrt_oss|``"0"``, ``"1"``                                           |
|is_dynamic         |``"0"``, ``"1"``                                           |

#### min_shape, max_shape, opt_shape选项
当且仅当开启动态shape时(*is_dynamic*为*1*)，每个输入需要设置最大形状(*max_shape*)、最小形状(*min_shape*)和最常见形状(*opt_shape*)。其中字典*parameters*中*key*为输入的名字，*value*为对应输入的最大、最小、最常见shape。

#### TensorRT动态shape例子
```
optimization {
  execution_accelerators {
    gpu_execution_accelerator : [
      {
        name : "tensorrt"
        # 使用TensorRT的FP16推理
        parameters { key: "precision" value: "trt_fp16" }
        # 设置TensorRT的子图最小op数为3
        parameters { key: "min_graph_size" value: "3" }
        parameters { key: "workspace_size" value: "1073741824" }
        # 不使用变长
        parameters { key: "enable_tensorrt_oss" value: "0" }
        # 开启动态shape
        parameters { key: "is_dynamic" value: "1" }
      },
      {
        name : "min_shape"
        parameters { key: "eval_placeholder_0" value: "1" }
        parameters { key: "eval_placeholder_1" value: "1" }
        parameters { key: "eval_placeholder_2" value: "1" }
        parameters { key: "eval_placeholder_3" value: "1 1 1" }
      },
      {
        name : "max_shape"
        parameters { key: "eval_placeholder_0" value: "4096" }
        parameters { key: "eval_placeholder_1" value: "4096" }
        parameters { key: "eval_placeholder_2" value: "129" }
        parameters { key: "eval_placeholder_3" value: "1 128 1" }
      },
      {
        name : "opt_shape"
        parameters { key: "eval_placeholder_0" value: "128" }
        parameters { key: "eval_placeholder_1" value: "128" }
        parameters { key: "eval_placeholder_2" value: "2" }
        parameters { key: "eval_placeholder_3" value: "1 128 1" }
      }
    ]
  }
}
```
