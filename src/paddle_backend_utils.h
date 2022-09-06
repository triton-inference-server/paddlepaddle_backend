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

#include "paddle_inference_api.h"
#include "experimental/phi/common/float16.h"
#include "triton/core/tritonserver.h"

// namespace triton { namespace backend { namespace paddle {

#define RESPOND_ALL_AND_RETURN_IF_ERROR(RESPONSES, RESPONSES_COUNT, X) \
  do {                                                                 \
    TRITONSERVER_Error* raarie_err__ = (X);                            \
    if (raarie_err__ != nullptr) {                                     \
      SendErrorForResponses(RESPONSES, RESPONSES_COUNT, raarie_err__); \
      return;                                                          \
    }                                                                  \
  } while (false)

#define RETURN_IF_TRITONPADDLE_ERROR(ERR)                                    \
  do {                                                                       \
    TRITONPADDLE_Error* error__ = (ERR);                                     \
    if (error__ != nullptr) {                                                \
      auto status =                                                          \
          TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, error__->msg_); \
      TRITONPADDLE_ErrorDelete(error__);                                     \
      return status;                                                         \
    }                                                                        \
  } while (false)

#define THROW_IF_TRITONPADDLE_ERROR(X)         \
  do {                                         \
    TRITONPADDLE_Error* tie_err__ = (X);       \
    if (tie_err__ != nullptr) {                \
      throw TRITONPADDLE_Exception(tie_err__); \
    }                                          \
  } while (false)

typedef struct {
  char* msg_;
} TRITONPADDLE_Error;

struct TRITONPADDLE_Exception {
  TRITONPADDLE_Exception(TRITONPADDLE_Error* err) : err_(err) {}
  TRITONPADDLE_Error* err_;
};

TRITONPADDLE_Error* TRITONPADDLE_ErrorNew(const std::string& str);

void TRITONPADDLE_ErrorDelete(TRITONPADDLE_Error* error);

// TRITONPADDLE TYPE
// TODO: Full all possible type?
typedef enum {
  TRITONPADDLE_TYPE_FP32,
  TRITONPADDLE_TYPE_INT64,
  TRITONPADDLE_TYPE_INT32,
  TRITONPADDLE_TYPE_UINT8,
  TRITONPADDLE_TYPE_INT8,
  TRITONPADDLE_TYPE_FP16,
  TRITONPADDLE_TYPE_INVALID
} TRITONPADDLE_DataType;

// TRITONPADDLE SHAPE
class TRITONPADDLE_Shape {
 public:
  using value_type = int64_t;

  TRITONPADDLE_Shape() = default;
  TRITONPADDLE_Shape(const std::string& str);
  template <typename T>
  TRITONPADDLE_Shape(const std::vector<T>& shape);
  size_t NumElements() const { return numel_; };

  std::vector<int32_t> CompatibleShape() const;
  std::vector<value_type> Shape() const { return shape_; };

 private:
  std::vector<value_type> shape_;
  size_t numel_;
};

TRITONPADDLE_DataType ConvertDataType(TRITONSERVER_DataType dtype);

TRITONPADDLE_DataType ConvertDataType(::paddle_infer::DataType dtype);

TRITONPADDLE_DataType ConvertDataType(const std::string& dtype);

TRITONSERVER_DataType ConvertDataType(TRITONPADDLE_DataType dtype);

size_t TRITONPADDLE_DataTypeByteSize(TRITONPADDLE_DataType dtype);

// TRITON PADDLE MODE
typedef enum {
  TRITONPADDLE_MODE_FP32,
  TRITONPADDLE_MODE_FP16,
  TRITONPADDLE_MODE_INT8,
} TRITONPADDLE_Precision;

// TRITON PADDLE CONFIG
class TRITONPADDLE_Config {
 public:
  TRITONPADDLE_Config();
  // trt
  bool use_trt_;
  int64_t max_batch_size_;
  int64_t workspace_size_;
  int64_t min_graph_size_;
  TRITONPADDLE_Precision precision_;
  bool is_dynamic_;
  bool enable_tensorrt_oss_;
  bool disenable_trt_tune_;
  // cpu
  bool use_cpu_;
  bool use_mkldnn_;
  bool use_ort_;
  bool use_mkldnn_int8_;
  int cpu_math_library_num_threads_;
  int mkldnn_capacity_;
  std::string model_dir_;

  std::map<std::string, std::vector<int>> dynamic_min_shape_;
  std::map<std::string, std::vector<int>> dynamic_max_shape_;
  std::map<std::string, std::vector<int>> dynamic_opt_shape_;
};

// }}}
