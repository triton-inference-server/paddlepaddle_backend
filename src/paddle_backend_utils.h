// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>

#include "pd_inference_api.h"
#include "triton/backend/backend_common.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace backend { namespace paddle {

// TRITONPADDLE SHAPE
class TRITONPADDLE_Shape {
 public:
  using value_type = int64_t;

  TRITONPADDLE_Shape() = default;
  TRITONPADDLE_Shape(const std::string& str);
  template <typename T>
  TRITONPADDLE_Shape(const std::vector<T>& shape);

  std::vector<int32_t> CompatibleShape() const;

 private:
  std::vector<value_type> shape_;
};

struct PaddleTRTConfig {
  std::unordered_map<std::string, std::vector<int>> min_shapes_;
  std::unordered_map<std::string, std::vector<int>> max_shapes_;
  std::unordered_map<std::string, std::vector<int>> opt_shapes_;
  bool is_dynamic_ = false;
  bool disable_trt_tune_ = false;
};

struct PDConfigDeleter {
  void operator()(PD_Config* f) {
    PD_ConfigDestroy(f); 
  }
};

struct PD_PredictorDeleter {
  void operator()(PD_Predictor* f) {
    PD_PredictorDestroy(f);
  }
};

struct PaddleTensorInfo {
  PaddleTensorInfo(PD_DataType type, std::vector<int64_t> dims)
      : type_(type), dims_(dims){}

  PD_DataType type_;
  std::vector<int64_t> dims_;
};

using PaddleTensorInfoMap = std::unordered_map<std::string, PaddleTensorInfo>;

TRITONSERVER_Error*
InputOutputInfos(PD_Predictor* predictor, bool is_input,
                 PaddleTensorInfoMap& infos);
TRITONSERVER_Error* InputInfos(
    PD_Predictor* predictor, PaddleTensorInfoMap& infos);
TRITONSERVER_Error* OutputInfos(
    PD_Predictor* predictor, PaddleTensorInfoMap& infos);

TRITONSERVER_Error* PaddleInputMutableData(
    PD_Tensor* input_tensor, TRITONSERVER_DataType data_type,
    PD_PlaceType place_type, char** data_ptr);

TRITONSERVER_Error* PaddleOutputData(
    PD_Tensor* output_tensor, TRITONSERVER_DataType data_type,
    PD_PlaceType* place_type, std::vector<int64_t>& shapes,
    char** data_ptr);

PD_DataType ModelConfigDataTypeToPaddleDataType(
    const std::string& data_type_str);

std::string PaddleDataTypeToModelConfigDataType(
    PD_DataType data_type);

TRITONSERVER_DataType
ConvertFromPaddleDataType(PD_DataType pd_type);

TRITONSERVER_Error* CompareDimsSupported(
    const std::string& model_name, const std::string& tensor_name,
    const std::vector<int64_t>& model_shape, const std::vector<int64_t>& dims,
    const int max_batch_size, const bool compare_exact);

}}}
