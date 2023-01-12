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

namespace triton { namespace backend { namespace paddle {

template TRITONPADDLE_Shape::TRITONPADDLE_Shape(
    const std::vector<int64_t>& shape);
template TRITONPADDLE_Shape::TRITONPADDLE_Shape(
    const std::vector<int32_t>& shape);

template <typename T>
TRITONPADDLE_Shape::TRITONPADDLE_Shape(const std::vector<T>& shape)
{
  shape_ = std::vector<value_type>(shape.cbegin(), shape.cend());
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

TRITONSERVER_Error*
InputOutputInfos(PD_Predictor* predictor,
                 bool is_input,
                 PaddleTensorInfoMap& infos) {
  infos.clear();
  PD_IOInfos* pd_infos;
  if(is_input) {
    pd_infos = PD_PredictorGetInputInfos(predictor);
  } else {
    pd_infos = PD_PredictorGetOutputInfos(predictor);
  }
  
  for (size_t i = 0; i < pd_infos->size; i++) {
    char* cname = pd_infos->io_info[i]->name->data;
    std::string name(cname);
    PD_DataType type = pd_infos->io_info[i]->dtype;
    size_t num_dims = pd_infos->io_info[i]->shape->size;
    std::vector<int64_t> dims(num_dims);
    for (size_t j = 0; i < num_dims; j++) {
      dims[j] = pd_infos->io_info[i]->shape->data[j];
    }
    infos.emplace(std::move(name), PaddleTensorInfo(type, dims));
  }

  PD_IOInfosDestroy(pd_infos);
  return nullptr;  // success
}

TRITONSERVER_Error* InputInfos(
    PD_Predictor* predictor, PaddleTensorInfoMap& infos) {
  return InputOutputInfos(predictor, true, infos);
}

TRITONSERVER_Error* OutputInfos(
    PD_Predictor* predictor, PaddleTensorInfoMap& infos) {
  return InputOutputInfos(predictor, false, infos);
}

TRITONSERVER_Error* PaddleInputMutableData(
    PD_Tensor* pd_tensor, TRITONSERVER_DataType data_type,
    PD_PlaceType place_type, char** data_ptr) {
  if(data_type == TRITONSERVER_TYPE_UINT8) {
    *data_ptr = reinterpret_cast<char*>(
                  PD_TensorMutableDataUint8(pd_tensor, place_type));
  } else if(data_type == TRITONSERVER_TYPE_INT8) {
    *data_ptr = reinterpret_cast<char*>(
                  PD_TensorMutableDataInt8(pd_tensor, place_type));
  } else if(data_type == TRITONSERVER_TYPE_INT32) {
    *data_ptr = reinterpret_cast<char*>(
                  PD_TensorMutableDataInt32(pd_tensor, place_type));
  }  else if(data_type == TRITONSERVER_TYPE_FP64) {
    *data_ptr = reinterpret_cast<char*>(
                  PD_TensorMutableDataInt64(pd_tensor, place_type));
  } else if(data_type == TRITONSERVER_TYPE_FP32) {
    *data_ptr = reinterpret_cast<char*>(
                  PD_TensorMutableDataFloat(pd_tensor, place_type));
  } else {
    return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("Unexpected data type[") +
          TRITONSERVER_DataTypeString(data_type) +
         "] when get input mutable data.").c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error* PaddleOutputData(
    PD_Tensor* output_tensor, TRITONSERVER_DataType data_type,
    PD_PlaceType* place_type, std::vector<int64_t>& shapes,
    char** data_ptr) {

  {
    PD_OneDimArrayInt32* output_shape = PD_TensorGetShape(output_tensor);
    shapes.resize(output_shape->size);
    for (size_t i = 0; i < output_shape->size; ++i) {
      shapes[i] = output_shape->data[i];
    }
    PD_OneDimArrayInt32Destroy(output_shape);
  }

  int32_t shape_size;
  if(data_type == TRITONSERVER_TYPE_UINT8) {
    *data_ptr = reinterpret_cast<char*>(
                  PD_TensorDataUint8(output_tensor, place_type,
                    &shape_size));
  } else if(data_type == TRITONSERVER_TYPE_INT8) {
    *data_ptr = reinterpret_cast<char*>(
                  PD_TensorDataInt8(output_tensor, place_type,
                    &shape_size));
  } else if(data_type == TRITONSERVER_TYPE_INT32) {
    *data_ptr = reinterpret_cast<char*>(
                  PD_TensorDataInt32(output_tensor, place_type,
                    &shape_size));
  }  else if(data_type == TRITONSERVER_TYPE_FP64) {
    *data_ptr = reinterpret_cast<char*>(
                  PD_TensorDataInt64(output_tensor, place_type,
                    &shape_size));
  } else if(data_type == TRITONSERVER_TYPE_FP32) {
    *data_ptr = reinterpret_cast<char*>(
                  PD_TensorDataFloat(output_tensor, place_type,
                                &shape_size));
  } else {
    return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("Unexpected data type[") +
          TRITONSERVER_DataTypeString(data_type) +
         "] when get output data.").c_str());
  }
  
  return nullptr;
}

PD_DataType
ModelConfigDataTypeToPaddleDataType(const std::string& data_type_str)
{
  // Must start with "TYPE_".
  if (data_type_str.rfind("TYPE_", 0) != 0) {
    return PD_DATA_UNK;
  }

  const std::string dtype = data_type_str.substr(strlen("TYPE_"));

  if (dtype == "UINT8") {
    return PD_DATA_UINT8;
  } else if (dtype == "INT8") {
    return PD_DATA_INT8;
  } else if (dtype == "INT32") {
    return PD_DATA_INT32;
  } else if (dtype == "INT64") {
    return PD_DATA_INT64;
  } else if (dtype == "FP32") {
    return PD_DATA_FLOAT32;
  }

  return PD_DATA_UNK;
}

std::string
PaddleDataTypeToModelConfigDataType(PD_DataType data_type)
{
  if (data_type == PD_DATA_UINT8) {
    return "TYPE_UINT8";
  } else if (data_type == PD_DATA_INT8) {
    return "TYPE_INT8";
  } else if (data_type == PD_DATA_INT32) {
    return "TYPE_INT32";
  } else if (data_type == PD_DATA_INT64) {
    return "TYPE_INT64";
  } else if (data_type == PD_DATA_FLOAT32) {
    return "TYPE_FP32";
  }

  return "TYPE_INVALID";
}

TRITONSERVER_DataType
ConvertFromPaddleDataType(PD_DataType pd_type)
{
  switch (pd_type) {
    case PD_DATA_FLOAT32:
      return TRITONSERVER_TYPE_FP32;
    case PD_DATA_UINT8:
      return TRITONSERVER_TYPE_UINT8;
    case PD_DATA_INT8:
      return TRITONSERVER_TYPE_INT8;
    case PD_DATA_INT32:
      return TRITONSERVER_TYPE_INT32;
    case PD_DATA_INT64:
      return TRITONSERVER_TYPE_INT64;
    case PD_DATA_UNK:
      return TRITONSERVER_TYPE_INVALID;
    default:
      break;
  }

  return TRITONSERVER_TYPE_INVALID;
}

TRITONSERVER_Error*
CompareDimsSupported(
    const std::string& model_name, const std::string& tensor_name,
    const std::vector<int64_t>& model_shape, const std::vector<int64_t>& dims,
    const int max_batch_size, const bool compare_exact) {
  // If the model configuration expects batching support in the model,
  // then the shape first dimension must be -1.
  const bool supports_batching = (max_batch_size > 0);
  if (supports_batching) {
    RETURN_ERROR_IF_TRUE(
        (model_shape.size() == 0) || (model_shape[0] != -1),
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string("model '") + model_name + "', tensor '" + tensor_name +
            "': for the model to support batching the shape should have at "
            "least 1 dimension and the first dimension must be -1; but shape "
            "expected by the model is " +
            ShapeToString(model_shape));

    std::vector<int64_t> full_dims;
    full_dims.reserve(1 + dims.size());
    full_dims.push_back(-1);
    full_dims.insert(full_dims.end(), dims.begin(), dims.end());

    bool succ = (model_shape.size() == (size_t)full_dims.size());
    if (succ) {
      for (size_t i = 0; i < full_dims.size(); ++i) {
        const int64_t model_dim = model_shape[i];
        if (compare_exact || (model_dim != -1)) {
          succ &= (model_dim == full_dims[i]);
        }
      }
    }

    RETURN_ERROR_IF_TRUE(
        !succ, TRITONSERVER_ERROR_INVALID_ARG,
        std::string("model '") + model_name + "', tensor '" + tensor_name +
            "': the model expects " + std::to_string(model_shape.size()) +
            " dimensions (shape " + ShapeToString(model_shape) +
            ") but the model configuration specifies " +
            std::to_string(full_dims.size()) +
            " dimensions (an initial batch dimension because max_batch_size "
            "> 0 followed by the explicit tensor shape, making complete "
            "shape " +
            ShapeToString(full_dims) + ")");
  } else {
    // ! supports_batching
    bool succ = (model_shape.size() == dims.size());
    if (succ) {
      for (size_t i = 0; i < dims.size(); ++i) {
        const int64_t model_dim = model_shape[i];
        if (compare_exact || (model_dim != -1)) {
          succ &= (model_dim == dims[i]);
        }
      }
    }

    RETURN_ERROR_IF_TRUE(
        !succ, TRITONSERVER_ERROR_INVALID_ARG,
        std::string("model '") + model_name + "', tensor '" + tensor_name +
            "': the model expects " + std::to_string(model_shape.size()) +
            " dimensions (shape " + ShapeToString(model_shape) +
            ") but the model configuration specifies " +
            std::to_string(dims.size()) + " dimensions (shape " +
            ShapeToString(dims) + ")");
  }

  return nullptr;  // success
}


}}}
