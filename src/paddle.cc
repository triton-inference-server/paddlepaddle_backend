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

#include <algorithm>
#include <memory>
#include <numeric>

#include "paddle_backend_utils.h"
#include "paddle_inference_api.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"

struct TRITONPADDLE_Tensor;

// Paddle Predictor Wrapper
struct TRITONPADDLE_Model;

class ModelImpl {
 public:
  ModelImpl(
      const char* model_path, const char* param_path,
      TRITONPADDLE_Config* config, const int32_t device_id);
  ~ModelImpl() = default;
  void CollectShapeRun(paddle_infer::Predictor* predictor,
                       const std::map<std::string, std::vector<int>>& shape);
  void CollectTensorRtShapeRange(const char* model_path, const char* param_path,
                                 TRITONPADDLE_Config* config,
                                 const int32_t device_id,
                                 paddle::AnalysisConfig::Precision compute_precision);
  TRITONPADDLE_Error* Run();

  TRITONPADDLE_Error* GetInputPtr(
      const char* name, const TRITONPADDLE_DataType dtype,
      const TRITONPADDLE_Shape& shape, char** ptr);

  TRITONPADDLE_Error* GetOutputMetadata(
      const char* name, TRITONPADDLE_DataType* dtype, TRITONPADDLE_Shape* shape,
      char** ptr);

  TRITONPADDLE_Error* ZeroCopyRun();

 private:
  // TODO(wilber): unique_ptr?
  std::unique_ptr<paddle_infer::Config> analysis_config_;
  std::shared_ptr<paddle_infer::Predictor> predictor_;
  paddle_infer::PlaceType place_type_;
  std::string shape_range_info_;
};

void ModelImpl::CollectShapeRun(paddle_infer::Predictor* predictor,
                                const std::map<std::string, std::vector<int>>& shape) {
  auto input_names = predictor->GetInputNames();
  auto input_type = predictor->GetInputType();
  for(auto name : input_names) {
    if(shape.find(name) == shape.end() or
       input_type.find(name) == input_type.end()) {
      TRITONPADDLE_Error* error = TRITONPADDLE_ErrorNew(
          std::string("Paddle Input name [") + std::string(name) +
          std::string("] is not one of the trt dynamic_shape"));
      THROW_IF_TRITONPADDLE_ERROR(error);
    }

    auto tensor = predictor->GetInputHandle(name);
    auto shape_value = shape.at(name);
    int shape_num = std::accumulate(shape_value.begin(), shape_value.end(), 1,
                                    std::multiplies<int>());
    tensor->Reshape(shape_value);
    auto dtype = input_type[name];
    switch (dtype) {
      case paddle_infer::DataType::FLOAT32: {
        std::vector<float> input_data(shape_num, 1.0);
        tensor->CopyFromCpu(input_data.data());
        break;
      }
      case paddle_infer::DataType::INT32: {
        std::vector<int> input_data(shape_num, 1);
        tensor->CopyFromCpu(input_data.data());
        break;
      }
      case paddle_infer::DataType::INT64: {
        std::vector<int64_t> input_data(shape_num, 1);
        tensor->CopyFromCpu(input_data.data());
        break;
      }
      case paddle_infer::DataType::FLOAT16: {
        std::vector<phi::dtype::float16> input_data(shape_num, (phi::dtype::float16)1.0);
        tensor->CopyFromCpu(input_data.data());
        break;
      }
      default: {
        TRITONPADDLE_Error* error = TRITONPADDLE_ErrorNew(std::string(
            "input data Paddle backend only supports FP32/INT32/INT64 currently"));
        THROW_IF_TRITONPADDLE_ERROR(error);
        break;
      }
    }
  }
  predictor->Run();
}

void ModelImpl::CollectTensorRtShapeRange(const char* model_path, const char* param_path,
                                          TRITONPADDLE_Config* config,
                                          const int32_t device_id,
                                          paddle::AnalysisConfig::Precision compute_precision) {
  paddle_infer::Config analysis_config;
  if (param_path == nullptr) {
    analysis_config.SetModel(model_path, "");
  } else {
    analysis_config.SetModel(model_path, param_path);
  }
  analysis_config.EnableUseGpu(100, device_id);
  analysis_config.EnableTensorRtEngine(
        config->workspace_size_, config->max_batch_size_,
        config->min_graph_size_, compute_precision, false, false);
  analysis_config.CollectShapeRangeInfo(shape_range_info_);
  auto predictor = paddle_infer::CreatePredictor(analysis_config);
  CollectShapeRun(predictor.get(), config->dynamic_min_shape_);
  CollectShapeRun(predictor.get(), config->dynamic_max_shape_);
  CollectShapeRun(predictor.get(), config->dynamic_opt_shape_);
}

ModelImpl::ModelImpl(
    const char* model_path, const char* param_path, TRITONPADDLE_Config* config,
    const int32_t device_id)
{
  analysis_config_.reset(new paddle_infer::Config());

  if (param_path == nullptr) {
    analysis_config_->SetModel(model_path, "");
  } else {
    analysis_config_->SetModel(model_path, param_path);
  }

  // default settings
  analysis_config_->SwitchSpecifyInputNames(true);
  analysis_config_->SwitchIrOptim(true);
  analysis_config_->EnableMemoryOptim();
  analysis_config_->SwitchUseFeedFetchOps(false);

  if (config->use_cpu_) {
    place_type_ = paddle_infer::PlaceType::kCPU;
    analysis_config_->SetCpuMathLibraryNumThreads(config->cpu_math_library_num_threads_);
    if(config->use_ort_) {
      analysis_config_->EnableONNXRuntime();
      analysis_config_->EnableORTOptimization();
    } else if(config->use_mkldnn_) {
      analysis_config_->EnableMKLDNN();
      analysis_config_->SetMkldnnCacheCapacity(config->mkldnn_capacity_);
      // Release/2.3 don't support mkldnn_int8
      // if(config->use_mkldnn_int8_)
      //   analysis_config_->EnableMkldnnInt8();
    }
  } else {
    place_type_ = paddle_infer::PlaceType::kGPU;
    analysis_config_->EnableUseGpu(100, device_id);

    paddle::AnalysisConfig::Precision compute_precision;
    compute_precision = paddle::AnalysisConfig::Precision::kFloat32;
    if (config->precision_ == TRITONPADDLE_MODE_FP32) {
      compute_precision = paddle::AnalysisConfig::Precision::kFloat32;
    } else if (config->precision_ == TRITONPADDLE_MODE_FP16) {
      compute_precision = paddle::AnalysisConfig::Precision::kHalf;
    } else if (config->precision_ == TRITONPADDLE_MODE_INT8) {
      compute_precision = paddle::AnalysisConfig::Precision::kInt8;
    } else {
      TRITONPADDLE_Error* error = TRITONPADDLE_ErrorNew(
          "unknown precision type when setting tensorrt compute precision.");
      THROW_IF_TRITONPADDLE_ERROR(error);
    }

    if (config->use_trt_) {
      analysis_config_->EnableTensorRtEngine(
        config->workspace_size_, config->max_batch_size_,
        config->min_graph_size_, compute_precision, false, false);
      if (config->is_dynamic_) {
        shape_range_info_ = config->model_dir_ + "/shape_range_info.pbtxt";
        CollectTensorRtShapeRange(model_path, param_path, config, device_id, compute_precision);
        analysis_config_->EnableTunedTensorRtDynamicShape(shape_range_info_);
      }
    }
  }
  predictor_ = std::move(paddle_infer::CreatePredictor(*analysis_config_.get()));
}

TRITONPADDLE_Error*
ModelImpl::Run()
{
  predictor_->Run();

  // TODO: paddle predictor stream controll
  if(analysis_config_->use_gpu())
    cudaDeviceSynchronize();
  return nullptr;
}

TRITONPADDLE_Error*
ModelImpl::GetInputPtr(
    const char* name, const TRITONPADDLE_DataType dtype,
    const TRITONPADDLE_Shape& shape, char** ptr)
{
  auto input_names = predictor_->GetInputNames();

  // check whether the given name is in predictor_ input names
  if (std::find(input_names.begin(), input_names.end(), std::string(name)) ==
      input_names.end()) {
    return TRITONPADDLE_ErrorNew(
        std::string("Input name [") + std::string(name) +
        std::string("] is not one of the Paddle predictor input"));
  }

  auto tensor = predictor_->GetInputHandle(name);
  tensor->Reshape(shape.CompatibleShape());
  switch (dtype) {
    case TRITONPADDLE_TYPE_FP32:
      *ptr = reinterpret_cast<char*>(
          tensor->mutable_data<float>(place_type_));
      break;
    case TRITONPADDLE_TYPE_INT32:
      *ptr = reinterpret_cast<char*>(
          tensor->mutable_data<int32_t>(place_type_));
      break;
    case TRITONPADDLE_TYPE_INT64:
      *ptr = reinterpret_cast<char*>(
          tensor->mutable_data<int64_t>(place_type_));
      break;
    case TRITONPADDLE_TYPE_FP16:
      *ptr = reinterpret_cast<char*>(
          tensor->mutable_data<phi::dtype::float16>(place_type_));
      break;
    default:
      return TRITONPADDLE_ErrorNew(std::string(
          "Paddle backend only supports FP32/INT32/INT64 currently"));
  }

  return nullptr;
}

TRITONPADDLE_Error*
ModelImpl::GetOutputMetadata(
    const char* name, TRITONPADDLE_DataType* dtype, TRITONPADDLE_Shape* shape,
    char** ptr)
{
  auto output_names = predictor_->GetOutputNames();

  // check whether the given name is in predictor_ output names
  if (std::find(output_names.begin(), output_names.end(), std::string(name)) ==
      output_names.end()) {
    return TRITONPADDLE_ErrorNew(
        std::string("Output name [") + std::string(name) +
        std::string("] is not one of the Paddle predictor input"));
  }

  auto tensor = predictor_->GetOutputHandle(name);
  auto tensor_type = tensor->type();
  auto tensor_shape = tensor->shape();

  *dtype = ConvertDataType(tensor_type);
  *shape = TRITONPADDLE_Shape(tensor_shape);

  switch (*dtype) {
    case TRITONPADDLE_TYPE_FP32:
      *ptr = reinterpret_cast<char*>(
          tensor->mutable_data<float>(place_type_));
      break;
    case TRITONPADDLE_TYPE_INT64:
      *ptr = reinterpret_cast<char*>(
          tensor->mutable_data<int64_t>(place_type_));
      break;
    case TRITONPADDLE_TYPE_INT32:
      *ptr = reinterpret_cast<char*>(
          tensor->mutable_data<int32_t>(place_type_));
      break;
    case TRITONPADDLE_TYPE_FP16:
      *ptr = reinterpret_cast<char*>(
          tensor->mutable_data<phi::dtype::float16>(place_type_));
      break;
    /*
    case TRITONPADDLE_TYPE_INT8:
      *ptr = reinterpret_cast<char*>(
          tensor->mutable_data<int8_t>(place_type_));
      break;
    case TRITONPADDLE_TYPE_UINT8:
      *ptr = reinterpret_cast<char*>(
          tensor->mutable_data<uint8_t>(place_type_));
      break;
    */
    default:
      return TRITONPADDLE_ErrorNew(std::string(
          "Paddle backend currently only support FP32/INT32/INT64"));
  }

  return nullptr;
}

TRITONSERVER_Error*
TRITONPADDLE_ModelCreate(
    TRITONPADDLE_Model** model, const char* model_path, const char* param_path,
    TRITONPADDLE_Config* config, const int32_t device_id)
{
  try {
    ModelImpl* model_impl =
        new ModelImpl(model_path, param_path, config, device_id);
    *model = reinterpret_cast<TRITONPADDLE_Model*>(model_impl);
  }
  catch (const TRITONPADDLE_Exception& ex) {
    RETURN_IF_TRITONPADDLE_ERROR(ex.err_);
  }
  return nullptr;
}

void
TRITONPADDLE_ModelDelete(TRITONPADDLE_Model* model)
{
  if (model != nullptr) {
    ModelImpl* mi = reinterpret_cast<ModelImpl*>(model);
    delete mi;
  }
}

TRITONPADDLE_Error*
TRITONPADDLE_ModelRun(TRITONPADDLE_Model* model)
{
  ModelImpl* m = reinterpret_cast<ModelImpl*>(model);
  return m->Run();
}

class TensorImpl {
 public:
  TensorImpl(
      const char* name, TRITONPADDLE_DataType dtype,
      const TRITONPADDLE_Shape& shape, char* data_ptr);
  ~TensorImpl() = default;

  const std::string& Name() const { return name_; }
  TRITONPADDLE_DataType DataType() const { return dtype_; }
  TRITONPADDLE_Shape Shape() const { return shape_; }

  char* Base() const { return base_; }
  size_t ByteSize() const { return byte_size_; }

 private:
  const std::string name_;
  const TRITONPADDLE_DataType dtype_;
  const TRITONPADDLE_Shape shape_;

  char* base_;
  size_t byte_size_;
};

TensorImpl::TensorImpl(
    const char* name, TRITONPADDLE_DataType dtype,
    const TRITONPADDLE_Shape& shape, char* data_ptr)
    : name_(name), dtype_(dtype), shape_(shape), base_(data_ptr)
{
  byte_size_ = shape.NumElements() * TRITONPADDLE_DataTypeByteSize(dtype);
}

TRITONPADDLE_Tensor*
TRITONPADDLE_TensorNew(
    TRITONPADDLE_Model* model, const char* name, TRITONPADDLE_DataType dtype,
    const TRITONPADDLE_Shape& shape)
{
  char* data_ptr;
  ModelImpl* m = reinterpret_cast<ModelImpl*>(model);
  auto err = m->GetInputPtr(name, dtype, shape, &data_ptr);
  if (err != nullptr) {
    return nullptr;
  }

  TensorImpl* tensor = new TensorImpl(name, dtype, shape, data_ptr);
  return reinterpret_cast<TRITONPADDLE_Tensor*>(tensor);
}

TRITONPADDLE_Tensor*
TRITONPADDLE_TensorNew(TRITONPADDLE_Model* model, const char* name)
{
  char* data_ptr;
  TRITONPADDLE_DataType dtype;
  TRITONPADDLE_Shape shape;

  ModelImpl* m = reinterpret_cast<ModelImpl*>(model);
  auto err = m->GetOutputMetadata(name, &dtype, &shape, &data_ptr);
  if (err != nullptr) {
    return nullptr;
  }

  TensorImpl* tensor = new TensorImpl(name, dtype, shape, data_ptr);
  return reinterpret_cast<TRITONPADDLE_Tensor*>(tensor);
}

char*
TRITONPADDLE_TensorData(TRITONPADDLE_Tensor* tensor)
{
  TensorImpl* t = reinterpret_cast<TensorImpl*>(tensor);
  return t->Base();
}

size_t
TRITONPADDLE_TensorDataByteSize(TRITONPADDLE_Tensor* tensor)
{
  TensorImpl* t = reinterpret_cast<TensorImpl*>(tensor);
  return t->ByteSize();
}

TRITONPADDLE_DataType
TRITONPADDLE_TensorDataType(TRITONPADDLE_Tensor* tensor)
{
  TensorImpl* t = reinterpret_cast<TensorImpl*>(tensor);
  return t->DataType();
}

TRITONPADDLE_Shape
TRITONPADDLE_TensorShape(TRITONPADDLE_Tensor* tensor)
{
  TensorImpl* t = reinterpret_cast<TensorImpl*>(tensor);
  return t->Shape();
}

namespace triton { namespace backend { namespace paddle {

using TRITONPADDLEModelHandle = std::shared_ptr<TRITONPADDLE_Model>;

class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;
  TRITONPADDLE_Config* PaddleConfig() { return &config_; }

 private:
  ModelState(TRITONBACKEND_Model* triton_model);

  // Auto-complete the model configuration
  TRITONSERVER_Error* AutoCompleteConfig();

  // Validate that model configuration is supported by this backend
  TRITONSERVER_Error* ValidateModelConfig();

  TRITONPADDLE_Config config_;
};

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  // Auto-complete the configuration if requested...
  bool auto_complete_config = false;
  RETURN_IF_ERROR(TRITONBACKEND_ModelAutoCompleteConfig(
      triton_model, &auto_complete_config));
  if (auto_complete_config) {
    RETURN_IF_ERROR((*state)->AutoCompleteConfig());

    triton::common::TritonJson::WriteBuffer json_buffer;
    (*state)->ModelConfig().Write(&json_buffer);

    TRITONSERVER_Message* message;
    RETURN_IF_ERROR(TRITONSERVER_MessageNewFromSerializedJson(
        &message, json_buffer.Base(), json_buffer.Size()));
    RETURN_IF_ERROR(TRITONBACKEND_ModelSetConfig(
        triton_model, 1 /* config_version */, message));
  }

  RETURN_IF_ERROR((*state)->ValidateModelConfig());

  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model)
{

  triton::common::TritonJson::Value optimization;
  if (not ModelConfig().Find("optimization", &optimization)) {
    return;
  }

  triton::common::TritonJson::Value eas;
  if (not optimization.Find("execution_accelerators", &eas)) {
    return;
  }

  // CPU execution providers
  {
    triton::common::TritonJson::Value cpu_eas;
    if (eas.Find("cpu_execution_accelerator", &cpu_eas)) {
      for (size_t idx = 0; idx < cpu_eas.ArraySize(); idx++) {
        triton::common::TritonJson::Value ea;
        THROW_IF_BACKEND_MODEL_ERROR(cpu_eas.IndexAsObject(idx, &ea));
        std::string name;
        THROW_IF_BACKEND_MODEL_ERROR(ea.MemberAsString("name", &name));
        if (name == "mkldnn") {
          config_.use_mkldnn_ = true;
        } else if (name == "ort") {
          config_.use_ort_ = true;
        } else if (name != "") {
          TRITONSERVER_Error* error = TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              std::string(
                  "unknown cpu_execution_accelerator name '" + name +
                  "' is provided. Available choices are [mkldnn, ort]")
                  .c_str());
          THROW_IF_BACKEND_MODEL_ERROR(error);
        }
        triton::common::TritonJson::Value params;
        if (ea.Find("parameters", &params)) {
          std::vector<std::string> param_keys;
          THROW_IF_BACKEND_MODEL_ERROR(params.Members(&param_keys));
          for (const auto& param_key : param_keys) {
            std::string value_string;
            if (param_key == "cpu_threads") {
              THROW_IF_BACKEND_MODEL_ERROR(
                  params.MemberAsString(param_key.c_str(), &value_string));
              THROW_IF_BACKEND_MODEL_ERROR(
                  ParseIntValue(value_string, &config_.cpu_math_library_num_threads_));
            } else if (param_key == "capacity") {
              THROW_IF_BACKEND_MODEL_ERROR(
                  params.MemberAsString(param_key.c_str(), &value_string));
              THROW_IF_BACKEND_MODEL_ERROR(
                  ParseIntValue(value_string, &config_.mkldnn_capacity_));
            } else if (param_key == "use_int8") {
              THROW_IF_BACKEND_MODEL_ERROR(
                  params.MemberAsString(param_key.c_str(), &value_string));
              THROW_IF_BACKEND_MODEL_ERROR(
                  ParseBoolValue(value_string, &config_.use_mkldnn_int8_));
            }
          }
        }
      }
    }
  }

  // GPU execution providers
  {
    triton::common::TritonJson::Value gpu_eas;
    if (eas.Find("gpu_execution_accelerator", &gpu_eas)) {
      for (size_t idx = 0; idx < gpu_eas.ArraySize(); idx++) {
        triton::common::TritonJson::Value ea;
        THROW_IF_BACKEND_MODEL_ERROR(gpu_eas.IndexAsObject(idx, &ea));
        std::string name;
        THROW_IF_BACKEND_MODEL_ERROR(ea.MemberAsString("name", &name));

        if (name == "tensorrt") {
          config_.use_trt_ = true;
          triton::common::TritonJson::Value params;
          if (ea.Find("parameters", &params)) {
            std::vector<std::string> param_keys;
            THROW_IF_BACKEND_MODEL_ERROR(params.Members(&param_keys));
            for (const auto& param_key : param_keys) {
              std::string value_string;
              if (param_key == "precision") {
                THROW_IF_BACKEND_MODEL_ERROR(
                    params.MemberAsString(param_key.c_str(), &value_string));
                std::transform(
                    value_string.begin(), value_string.end(), value_string.begin(),
                    ::tolower);
                if (value_string == "trt_fp32") {
                  config_.precision_ = TRITONPADDLE_MODE_FP32;
                } else if (value_string == "trt_fp16") {
                  config_.precision_ = TRITONPADDLE_MODE_FP16;
                } else if (value_string == "trt_int8") {
                  config_.precision_ = TRITONPADDLE_MODE_INT8;
                } else {
                  TRITONSERVER_Error* error = TRITONSERVER_ErrorNew(
                      TRITONSERVER_ERROR_INVALID_ARG,
                      std::string(
                          "unknown precision type '" + value_string +
                          "' is provided. Available choices are [fluid, trt_fp32, "
                          "trt_fp16, trt_int8]")
                          .c_str());
                  THROW_IF_BACKEND_MODEL_ERROR(error);
                }
              } else if (param_key == "min_graph_size") {
                THROW_IF_BACKEND_MODEL_ERROR(
                    params.MemberAsString(param_key.c_str(), &value_string));
                THROW_IF_BACKEND_MODEL_ERROR(
                    ParseLongLongValue(value_string, &config_.min_graph_size_));
              } else if (param_key == "workspace_size") {
                THROW_IF_BACKEND_MODEL_ERROR(
                    params.MemberAsString(param_key.c_str(), &value_string));
                THROW_IF_BACKEND_MODEL_ERROR(
                    ParseLongLongValue(value_string, &config_.workspace_size_));
              } else if (param_key == "max_batch_size") {
                THROW_IF_BACKEND_MODEL_ERROR(
                    params.MemberAsString(param_key.c_str(), &value_string));
                THROW_IF_BACKEND_MODEL_ERROR(
                    ParseLongLongValue(value_string, &config_.max_batch_size_));
              } else if (param_key == "enable_tensorrt_oss") {
                THROW_IF_BACKEND_MODEL_ERROR(
                    params.MemberAsString(param_key.c_str(), &value_string));
                THROW_IF_BACKEND_MODEL_ERROR(
                    ParseBoolValue(value_string, &config_.enable_tensorrt_oss_));
              } else if (param_key == "is_dynamic") {
                THROW_IF_BACKEND_MODEL_ERROR(
                    params.MemberAsString(param_key.c_str(), &value_string));
                THROW_IF_BACKEND_MODEL_ERROR(
                    ParseBoolValue(value_string, &config_.is_dynamic_));
              } else {
                TRITONSERVER_Error* error = TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INVALID_ARG,
                    std::string(
                        "unknown parameter '" + param_key +
                        "' is provided for GPU execution accelerator "
                        "config. Available choices are [precision, "
                        "min_graph_size, workspace_size, max_batch_size, "
                        "enable_tensorrt_oss, is_dynamic]")
                        .c_str());
                THROW_IF_BACKEND_MODEL_ERROR(error);
              }
            }
          }
        } else if (
            name == "min_shape" or name == "max_shape" or name == "opt_shape") {
          triton::common::TritonJson::Value params;
          if (ea.Find("parameters", &params)) {
            std::vector<std::string> input_names;
            THROW_IF_BACKEND_MODEL_ERROR(params.Members(&input_names));
            for (const auto& input_name : input_names) {
              std::string str_shape;
              THROW_IF_BACKEND_MODEL_ERROR(
                  params.MemberAsString(input_name.c_str(), &str_shape));
              if (name == "min_shape") {
                config_.dynamic_min_shape_[input_name] =
                    TRITONPADDLE_Shape(str_shape).CompatibleShape();
              } else if (name == "max_shape") {
                config_.dynamic_max_shape_[input_name] =
                    TRITONPADDLE_Shape(str_shape).CompatibleShape();
              } else {
                config_.dynamic_opt_shape_[input_name] =
                    TRITONPADDLE_Shape(str_shape).CompatibleShape();
              }
            }
          }
        } else {
          TRITONSERVER_Error* error = TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              std::string(
                  "unknown name '" + name +
                  "' is provided for GPU execution accelerator "
                  "Available choices are [config, min_shape, max_shape, opt_shape]")
                  .c_str());
          THROW_IF_BACKEND_MODEL_ERROR(error);
        }
      }
    }
  }
}

TRITONSERVER_Error*
ModelState::AutoCompleteConfig()
{
  // Auto-complete configuration if requests
  LOG_MESSAGE(
      TRITONSERVER_LOG_WARN,
      (std::string("skipping model configuration auto-complete for '") +
       Name() + "': not supported for paddle backend")
          .c_str());

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  triton::common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("input", &ios));
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    // Check datatypes
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
    RETURN_ERROR_IF_TRUE(
        ConvertDataType(io_dtype) ==
            TRITONPADDLE_DataType::TRITONPADDLE_TYPE_INVALID,
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string("unsupported datatype '") + io_dtype + "' for tensor '" +
            io_name + "' for model '" + Name() + "'");
  }
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("output", &ios));
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    // Check datatypes
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
    RETURN_ERROR_IF_TRUE(
        ConvertDataType(io_dtype) ==
            TRITONPADDLE_DataType::TRITONPADDLE_TYPE_INVALID,
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string("unsupported datatype '") + io_dtype + "' for tensor '" +
            io_name + "' for model '" + Name() + "'");
  }

  return nullptr;  // success
}

class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState() = default;

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  void ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);

  TRITONSERVER_Error* DetermineModelAndParamsPath(
      const std::string& model_dir, std::string* model_path,
      std::string* param_path);

  void SetInputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);

  void ReadOutputTensors(
      size_t total_batch_size, const std::vector<std::string>& output_names,
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);

  ModelState* model_state_;
  TRITONPADDLEModelHandle triton_paddle_model_;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::DetermineModelAndParamsPath(
    const std::string& model_dir, std::string* model_path,
    std::string* param_path)
{
  bool exists;
  *model_path = JoinPath({model_dir, "model.pdmodel"});
  RETURN_IF_ERROR(FileExists(*model_path, &exists));
  if (not exists) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_NOT_FOUND,
        std::string(
            "Paddle model should be named as 'model.pdmodel'").c_str());
  }

  *param_path = JoinPath({model_dir, "model.pdiparams"});
  RETURN_IF_ERROR(FileExists(*param_path, &exists));
  if (not exists) {
    LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Paddle params should be named as 'model.pdiparams' or not provided.").c_str()));
    *param_path = "";
  }

  return nullptr;
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state)
{
  auto config = model_state->PaddleConfig();
  auto model_dir = JoinPath(
      {model_state->RepositoryPath(), std::to_string(model_state->Version())});
  config->model_dir_ = model_dir;

  std::string model_path;
  std::string param_path;
  THROW_IF_BACKEND_INSTANCE_ERROR(
      DetermineModelAndParamsPath(model_dir, &model_path, &param_path));

  switch (Kind()) {
    case TRITONSERVER_INSTANCEGROUPKIND_CPU:
      config->use_cpu_ = true;
      break;
    case TRITONSERVER_INSTANCEGROUPKIND_GPU:
      config->use_cpu_ = false;
      break;
    default:
      throw BackendModelInstanceException(TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unexpected instance kind for ") + name_ +
           ", paddle_backend only supports CPU/GPU.")
              .c_str()));
  }

  TRITONPADDLE_Model* triton_paddle_model = nullptr;
  THROW_IF_BACKEND_INSTANCE_ERROR(TRITONPADDLE_ModelCreate(
      &triton_paddle_model, model_path.c_str(),
      param_path.empty() ? nullptr : param_path.c_str(),
      config, DeviceId()));
  triton_paddle_model_.reset(triton_paddle_model, TRITONPADDLE_ModelDelete);
}

void
ModelInstanceState::SetInputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  bool cuda_copy = false;
  BackendInputCollector collector(
      requests, request_count, responses,
      StateForModel()->TritonMemoryManager(),
      StateForModel()->EnablePinnedInput(), CudaStream());

  const int max_batch_size = model_state_->MaxBatchSize();

  // All requests must have equally-sized input tensors so use any
  // request as the representative for the input tensors.
  uint32_t input_count;
  RESPOND_ALL_AND_RETURN_IF_ERROR(
      responses, request_count,
      TRITONBACKEND_RequestInputCount(requests[0], &input_count));

  for (uint32_t input_idx = 0; input_idx < input_count; ++input_idx) {
    TRITONBACKEND_Input* input;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input));

    const char* name;
    TRITONSERVER_DataType datatype;
    const int64_t* shape;
    uint32_t dims_count;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        TRITONBACKEND_InputProperties(
            input, &name, &datatype, &shape, &dims_count, nullptr, nullptr));

    // The shape for the entire input patch, [total_batch_size, ...]
    std::vector<int64_t> batchn_shape(shape, shape + dims_count);

    if (max_batch_size != 0) {
      batchn_shape[0] = total_batch_size;
    }

    TRITONPADDLE_Tensor* tensor = TRITONPADDLE_TensorNew(
        triton_paddle_model_.get(), name, ConvertDataType(datatype),
        TRITONPADDLE_Shape(batchn_shape));

    if (tensor == nullptr) {
      auto err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("Failed to create input tensor '") + name +
           "' with shape " + backend::ShapeToString(batchn_shape) +
           " and data type " + TRITONSERVER_DataTypeString(datatype) +
           " for '" + Name() + "'")
              .c_str());
      SendErrorForResponses(responses, request_count, err);
      return;
    }

    if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
      collector.ProcessTensor(
          name, TRITONPADDLE_TensorData(tensor),
          TRITONPADDLE_TensorDataByteSize(tensor), TRITONSERVER_MEMORY_GPU,
          DeviceId());
    }
    else {
        collector.ProcessTensor(
          name, TRITONPADDLE_TensorData(tensor),
          TRITONPADDLE_TensorDataByteSize(tensor), TRITONSERVER_MEMORY_CPU,
          0);
    }
  }

  cuda_copy |= collector.Finalize();
  if (cuda_copy) {
    cudaStreamSynchronize(CudaStream());
  }
}

void
ModelInstanceState::ReadOutputTensors(
    size_t total_batch_size, const std::vector<std::string>& output_names,
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  BackendOutputResponder responder(
      requests, request_count, responses, StateForModel()->MaxBatchSize(),
      StateForModel()->TritonMemoryManager(),
      StateForModel()->EnablePinnedOutput(), CudaStream());

  bool cuda_copy = false;
  for (size_t idx = 0; idx < output_names.size(); ++idx) {
    const std::string& name = output_names[idx];

    TRITONPADDLE_Tensor* tensor =
        TRITONPADDLE_TensorNew(triton_paddle_model_.get(), name.c_str());

    if (tensor == nullptr) {
      auto err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("Failed to create output tensor '") + name + " for '" +
           Name() + "'")
              .c_str());
      SendErrorForResponses(responses, request_count, err);
      return;
    }

    auto dtype = ConvertDataType(TRITONPADDLE_TensorDataType(tensor));
    auto shape = TRITONPADDLE_TensorShape(tensor).Shape();

    if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
      responder.ProcessTensor(
          name, dtype, shape, TRITONPADDLE_TensorData(tensor),
          TRITONSERVER_MEMORY_GPU, DeviceId());
    } else {
      responder.ProcessTensor(
          name, dtype, shape, TRITONPADDLE_TensorData(tensor),
          TRITONSERVER_MEMORY_CPU, 0);
    }
  }

  cuda_copy |= responder.Finalize();
  if (cuda_copy) {
    cudaStreamSynchronize(CudaStream());
  }

}

void
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelExecute: Running ") + Name() + " with " +
       std::to_string(request_count) + " requests")
          .c_str());

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  const int max_batch_size = model_state_->MaxBatchSize();

  // For each request collect the total batch size for this inference
  // execution. The batch-size, number of inputs, and size of each
  // input has already been checked so don't need to do that here.
  size_t total_batch_size = 0;
  for (size_t i = 0; i < request_count; ++i) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "null request given to Paddle backend for '" + Name() + "'")
                  .c_str()));
      return;
    }

    if (max_batch_size > 0) {
      // Retrieve the batch size from one of the inputs, if the model
      // supports batching, the first dimension size is batch size
      TRITONBACKEND_Input* input;
      TRITONSERVER_Error* err =
          TRITONBACKEND_RequestInputByIndex(requests[i], 0 /* index */, &input);
      if (err == nullptr) {
        const int64_t* shape;
        err = TRITONBACKEND_InputProperties(
            input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
        total_batch_size += shape[0];
      }
      if (err != nullptr) {
        RequestsRespondWithError(requests, request_count, err);
        return;
      }
    } else {
      total_batch_size += 1;
    }
  }

  // If there are no valid requests then no need to run the
  // inference. This should never happen unless called with an empty
  // 'requests' for some reason.
  if (total_batch_size == 0) {
    return;
  }

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests
  if ((total_batch_size != 1) and
      (total_batch_size > static_cast<size_t>(max_batch_size))) {
    RequestsRespondWithError(
        requests, request_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            std::string(
                "batch size " + std::to_string(total_batch_size) + " for '" +
                Name() + "', max allowed is " + std::to_string(max_batch_size))
                .c_str()));
    return;
  }

  // At this point we are committed to running inference with all
  // 'requests'. Create a response for each request. During input
  // processing if there is an error with any request that error will
  // be sent immediately with the corresponding response (and the
  // response pointer will then be nullptr). The request object
  // itself will not be released until after all inferencing is done
  // (below) as we may need to access the request object when
  // determine how to process outputs (for example, even if we don't
  // need the outputs for a request that has an error,  we do need to
  // know the size of those outputs associated with the request so we
  // can skip them in the output tensors).
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);

  for (size_t i = 0; i < request_count; ++i) {
    TRITONBACKEND_Response* response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err == nullptr) {
      responses.emplace_back(response);
    } else {
      responses.emplace_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
    }
  }

  SetInputTensors(total_batch_size, requests, request_count, &responses);

  // Collect the names of requested outputs. Do not include outputs
  // for requests that have already responded with an error.
  // TODO: understand here
  std::vector<std::string> required_outputs;
  std::vector<std::vector<std::string>> request_required_outputs(request_count);
  for (size_t idx = 0; idx < request_count; ++idx) {
    const auto& request = requests[idx];
    auto& response = responses[idx];
    if (response != nullptr) {
      uint32_t output_count;
      RESPOND_AND_SET_NULL_IF_ERROR(
          &response, TRITONBACKEND_RequestOutputCount(request, &output_count));
      if (response != nullptr) {
        for (uint32_t output_idx = 0; output_idx < output_count; ++output_idx) {
          const char* output_name;
          RESPOND_AND_SET_NULL_IF_ERROR(
              &response, TRITONBACKEND_RequestOutputName(
                             request, output_idx, &output_name));

          if (response != nullptr) {
            required_outputs.push_back(output_name);
            request_required_outputs[idx].push_back(output_name);
          }
        }
      }
    }
  }

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  TRITONPADDLE_ModelRun(triton_paddle_model_.get());

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  ReadOutputTensors(
      total_batch_size, required_outputs, requests, request_count, &responses);

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  // Send all the responses that haven't already been sent because of
  // an earlier error. Note that the responses are not set to nullptr
  // here as we need that indication below to determine if the request
  // we successful or not.
  for (auto& response : responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed to send Paddle backend response");
    }
  }

  // Report statistics for each request.
  for (uint32_t r = 0; r < request_count; ++r) {
    auto& request = requests[r];
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            TritonModelInstance(), request,
            (responses[r] != nullptr) /* success */, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  // TODO: Report the entire batch statistics.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          TritonModelInstance(), total_batch_size, exec_start_ns,
          compute_start_ns, compute_end_ns, exec_end_ns),
      "failed reporting batch request statistics");

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelExecute: model ") + Name() +
       " released " + std::to_string(request_count) + " requests")
          .c_str());
}

extern "C" {

TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("Triton TRITONBACKEND API version: ") +
         std::to_string(api_version_major) + "." +
         std::to_string(api_version_minor) + " does not support '" + name +
         "' TRITONBACKEND API version: " +
         std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
         std::to_string(TRITONBACKEND_API_VERSION_MINOR))
            .c_str());
  }

  // The backend configuration may contain information needed by the
  // backend, such a command-line arguments.
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend configuration:\n") + buffer).c_str());

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  // Create a ModelState object and associate it with the
  // TRITONBACKEND_Model.
  ModelState* model_state = nullptr;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  TRITONSERVER_InstanceGroupKind kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
       TRITONSERVER_InstanceGroupKindString(kind) + " device " +
       std::to_string(device_id) + ")")
          .c_str());

  // Get the model state associated with this instance's model
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // With each instance we create a ModelInstanceState object and
  // associate it with the TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Suggested practice for this is to use only
  // function-local and model-instance-specific state (obtained from
  // 'instance'), which is what we do here.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model ") + model_state->Name() + ", instance " +
       instance_state->Name() + ", executing " + std::to_string(request_count) +
       " requests")
          .c_str());

  // At this point we accept ownership of 'requests', which means that
  // even if something goes wrong we must still return success from
  // this function. If something does go wrong in processing a
  // particular request then we send an error response just for the
  // specific request.
  instance_state->ProcessRequests(requests, request_count);

  return nullptr;  // success
}

}  // extern "C"
}}}  // namespace triton::backend::paddle
