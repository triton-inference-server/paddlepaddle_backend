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

# Model Configuration

## General Model Configuration
For the general model configuration information, please visit [triton-inference-server/server/docs/model_configuration](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md).

## Platform and Backend
For using paddle backend, no ``platform`` need to be provided. However, you should set ``backend`` to ``"paddle"`` in the model configuration.
```
backend: "paddle"
```

## Paddle TensorRT Prediction Configuration

Paddle supports inference with tensorrt engine, which can boost inference throughput and reduce latency. 

Related configuration can be set under ``optimization {execution_accelerators {gpu_execution_accelerator{...}}}``

There are four sections can be configured, which are ``config``, ``min_shape``, ``max_shape``, ``opt_shape``.

### ``config``

In ``config``, you can set the ``precision``, ``min_graph_size``, ``max_batch_size``, ``workspace_size``, ``enable_tensorrt_oss``, ``is_dynamic``. 
The meaning of the parameters can refer to [Paddle Inference Docs](https://paddle-inference.readthedocs.io/en/latest/api_reference/cxx_api_doc/Config/GPUConfig.html#tensorrt)

|Parameters         |Available options                                          |
|-------------------|-----------------------------------------------------------|
|precision          |``"fluid"``, ``"trt_fp32"``, ``"trt_fp16"``, ``"trt_int8"``|
|min_graph_size     |``"1"`` ~ ``"2147483647"``                                 |
|max_batch_size     |``"1"`` ~ ``"2147483647"``                                 |
|workspace_size     |``"1"`` ~ ``"2147483647"``                                 |
|enable_tensorrt_oss|``"0"``, ``"1"``                                           |
|is_dynamic         |``"0"``, ``"1"``                                           |

### ``min_shape``, ``max_shape``, ``opt_shape``
These sections are only needed if ``is_dynamic`` is ``"1"``. Multiple ``parameters`` can be existed if there are multiple dynamic shape input. The ``key`` in ``parameters`` are the input tensor name, and the ``value`` is the shape, no ``,`` or ``[]`` is needed.

### An Dynamic Shape Example
```
optimization {
  execution_accelerators {
    gpu_execution_accelerator : [
      {
        name : "config"
        parameters { key: "precision" value: "trt_fp16" }
        parameters { key: "min_graph_size" value: "5" }
        parameters { key: "workspace_size" value: "1073741824" }
        parameters { key: "enable_tensorrt_oss" value: "1" }
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
