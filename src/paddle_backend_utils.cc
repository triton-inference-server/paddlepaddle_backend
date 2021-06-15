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

#include "paddle_backend_utils.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <sstream>

// namespace triton { namespace backend { namespace paddle {

template TRITONPADDLE_Shape::TRITONPADDLE_Shape(
    const std::vector<int64_t>& shape);
template TRITONPADDLE_Shape::TRITONPADDLE_Shape(
    const std::vector<int32_t>& shape);

template <typename T>
TRITONPADDLE_Shape::TRITONPADDLE_Shape(const std::vector<T>& shape)
{
  shape_ = std::vector<value_type>(shape.cbegin(), shape.cend());
  numel_ = std::accumulate(
      shape_.cbegin(), shape_.cend(), 1, std::multiplies<value_type>());
}

TRITONPADDLE_Shape::TRITONPADDLE_Shape(const std::string& str)
{
  std::vector<std::string> str_shape;
  std::istringstream in(str);
  std::copy(
      std::istream_iterator<std::string>(in),
      std::istream_iterator<std::string>(), std::back_inserter(str_shape));

  std::transform(
      str_shape.cbegin(), str_shape.cend(), std::back_inserter(shape_),
      [](const std::string& str) -> value_type {
        return static_cast<value_type>(std::stoll(str));
      });
}

std::vector<int32_t>
TRITONPADDLE_Shape::CompatibleShape() const
{
  return std::vector<int32_t>(shape_.cbegin(), shape_.cend());
}

TRITONPADDLE_DataType
ConvertDataType(TRITONSERVER_DataType dtype)
{
  switch (dtype) {
    case TRITONSERVER_TYPE_INVALID:
      return TRITONPADDLE_TYPE_INVALID;
    case TRITONSERVER_TYPE_UINT8:
      return TRITONPADDLE_TYPE_UINT8;
    case TRITONSERVER_TYPE_INT8:
      return TRITONPADDLE_TYPE_INT8;
    case TRITONSERVER_TYPE_INT32:
      return TRITONPADDLE_TYPE_INT32;
    case TRITONSERVER_TYPE_INT64:
      return TRITONPADDLE_TYPE_INT64;
    case TRITONSERVER_TYPE_FP32:
      return TRITONPADDLE_TYPE_FP32;
    default:
      break;
  }
  return TRITONPADDLE_TYPE_INVALID;
}

TRITONSERVER_DataType
ConvertDataType(TRITONPADDLE_DataType dtype)
{
  switch (dtype) {
    case TRITONPADDLE_TYPE_INVALID:
      return TRITONSERVER_TYPE_INVALID;
    case TRITONPADDLE_TYPE_UINT8:
      return TRITONSERVER_TYPE_UINT8;
    case TRITONPADDLE_TYPE_INT8:
      return TRITONSERVER_TYPE_INT8;
    case TRITONPADDLE_TYPE_INT32:
      return TRITONSERVER_TYPE_INT32;
    case TRITONPADDLE_TYPE_INT64:
      return TRITONSERVER_TYPE_INT64;
    case TRITONPADDLE_TYPE_FP32:
      return TRITONSERVER_TYPE_FP32;
    default:
      break;
  }
  return TRITONSERVER_TYPE_INVALID;
}

TRITONPADDLE_DataType
ConvertDataType(::paddle_infer::DataType dtype)
{
  switch (dtype) {
    case ::paddle_infer::DataType::FLOAT32:
      return TRITONPADDLE_TYPE_FP32;
    case ::paddle_infer::DataType::INT64:
      return TRITONPADDLE_TYPE_INT64;
    case ::paddle_infer::DataType::INT32:
      return TRITONPADDLE_TYPE_INT32;
    case ::paddle_infer::DataType::UINT8:
      return TRITONPADDLE_TYPE_UINT8;
    // case ::paddle_infer::DataType::INT8:
    //   return TRITONPADDLE_TYPE_INT8;
    default:
      break;
  }
  return TRITONPADDLE_TYPE_INVALID;
}

TRITONPADDLE_DataType
ConvertDataType(const std::string& dtype)
{
  if (dtype == "TYPE_INVALID") {
    return TRITONPADDLE_DataType::TRITONPADDLE_TYPE_INVALID;
  } else if (dtype == "TYPE_FP32") {
    return TRITONPADDLE_DataType::TRITONPADDLE_TYPE_FP32;
  } else if (dtype == "TYPE_UINT8") {
    return TRITONPADDLE_DataType::TRITONPADDLE_TYPE_UINT8;
  } else if (dtype == "TYPE_INT8") {
    return TRITONPADDLE_DataType::TRITONPADDLE_TYPE_INT8;
  } else if (dtype == "TYPE_INT32") {
    return TRITONPADDLE_DataType::TRITONPADDLE_TYPE_INT32;
  } else if (dtype == "TYPE_INT64") {
    return TRITONPADDLE_DataType::TRITONPADDLE_TYPE_INT64;
  }
  return TRITONPADDLE_DataType::TRITONPADDLE_TYPE_INVALID;
}

size_t
TRITONPADDLE_DataTypeByteSize(TRITONPADDLE_DataType dtype)
{
  switch (dtype) {
    case TRITONPADDLE_DataType::TRITONPADDLE_TYPE_FP32:
      return sizeof(float);
    case TRITONPADDLE_DataType::TRITONPADDLE_TYPE_INT64:
      return sizeof(int64_t);
    case TRITONPADDLE_DataType::TRITONPADDLE_TYPE_INT32:
      return sizeof(int32_t);
    case TRITONPADDLE_DataType::TRITONPADDLE_TYPE_UINT8:
      return sizeof(uint8_t);
    case TRITONPADDLE_DataType::TRITONPADDLE_TYPE_INT8:
      return sizeof(int8_t);
    default:
      break;
  }
  return 0;  // Should not happened, TODO: Error handling
}

/* Error message */

TRITONPADDLE_Error*
TRITONPADDLE_ErrorNew(const std::string& str)
{
  TRITONPADDLE_Error* error = new TRITONPADDLE_Error();
  error->msg_ = new char[str.size() + 1];
  std::strcpy(error->msg_, str.c_str());
  return error;
}

void
TRITONPADDLE_ErrorDelete(TRITONPADDLE_Error* error)
{
  if (error == nullptr) {
    return;
  }

  delete[] error->msg_;
  delete error;
}

TRITONPADDLE_Config::TRITONPADDLE_Config()
    : max_batch_size_(1), workspace_size_(1 << 30), min_graph_size_(5),
      precision_(TRITONPADDLE_MODE_FLUID), is_dynamic_(false),
      enable_tensorrt_oss_(false)
{
}

// }}}
