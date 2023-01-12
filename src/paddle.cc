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
#include "pd_inference_api.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"

namespace triton { namespace backend { namespace paddle {

// BackendConfiguration
struct BackendConfiguration {
  BackendConfiguration() : default_max_batch_size_(0) {}
  int default_max_batch_size_;
};

class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

  TRITONSERVER_Error* PrintModelConfig();

  // Load an Paddle model using 'artifact_name' as the name for the Paddle
  // file/directory. If 'instance_group_kind' is not
  // TRITONSERVER_INSTANCEGROUPKIND_AUTO then use it and
  // 'instance_group_device_id' to initialize the appropriate
  // execution providers. Return in 'model_path' the full path to the
  // paddle file.
  TRITONSERVER_Error* LoadModel(
      const std::string& artifact_name,
      const TRITONSERVER_InstanceGroupKind instance_group_kind,
      const int32_t instance_group_device_id,
      std::string* model_path, std::string* params_path,
      cudaStream_t stream, PD_PlaceType* pd_plcace_type,
      PD_Predictor** predictor);

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  TRITONSERVER_Error* AutoCompleteConfig();
  TRITONSERVER_Error* AutoCompleteMaxBatch(
      const PaddleTensorInfoMap& input_tensor_infos,
      const PaddleTensorInfoMap& output_tensor_infos);
  TRITONSERVER_Error* AutoCompleteIO(
      const char* key, const PaddleTensorInfoMap& io_infos);
  TRITONSERVER_Error* CollectTensorRtShapeRange(std::string* model_path, std::string* param_path,
                                 const char* range_info_path, int32_t device_id);
  TRITONSERVER_Error* CollectShapeRun(PD_Predictor* predictor,
                                 const std::unordered_map<std::string, std::vector<int>>& shape);  

  std::unique_ptr<PD_Config, PDConfigDeleter> pd_config_;
  PaddleTRTConfig pd_trt_config_;
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
    RETURN_IF_ERROR((*state)->SetModelConfig());
  }

  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model) {
  // Create pd_config that will be cloned and used for each
  // instance when creating that instance's predictor.
  PD_Config* pd_config = PD_ConfigCreate();
  pd_config_.reset(pd_config);

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
          PD_ConfigEnableMKLDNN(pd_config);
        } else if (name == "ort") {
          PD_ConfigEnableONNXRuntime(pd_config);
          PD_ConfigEnableORTOptimization(pd_config);
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
              int cpu_threads = 1;
              THROW_IF_BACKEND_MODEL_ERROR(
                  params.MemberAsString(param_key.c_str(), &value_string));
              THROW_IF_BACKEND_MODEL_ERROR(
                  ParseIntValue(value_string, &cpu_threads));
              PD_ConfigSetCpuMathLibraryNumThreads(pd_config, cpu_threads);
            } else if (param_key == "capacity") {
              int mkldnn_capacity = 1;
              THROW_IF_BACKEND_MODEL_ERROR(
                  params.MemberAsString(param_key.c_str(), &value_string));
              THROW_IF_BACKEND_MODEL_ERROR(
                  ParseIntValue(value_string, &mkldnn_capacity));
              PD_ConfigSetMkldnnCacheCapacity(pd_config, mkldnn_capacity);
            } else if (param_key == "use_fp16") {
              bool use_float16 = false;
              THROW_IF_BACKEND_MODEL_ERROR(
                  params.MemberAsString(param_key.c_str(), &value_string));
              THROW_IF_BACKEND_MODEL_ERROR(
                  ParseBoolValue(value_string, &use_float16));
              if (use_float16) {
                PD_ConfigEnableMkldnnBfloat16(pd_config);
              }
            } else if (param_key == "use_int8") {
              bool use_int8 = false;
              THROW_IF_BACKEND_MODEL_ERROR(
                  params.MemberAsString(param_key.c_str(), &value_string));
              THROW_IF_BACKEND_MODEL_ERROR(
                  ParseBoolValue(value_string, &use_int8));
              if (use_int8) {
                PD_ConfigEnableMkldnnInt8(pd_config);
              }
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
      size_t names_num = 0;
      char** input_tensor_names = NULL;
      size_t* shapes_num = NULL;
      int32_t** min_shapes = NULL;
      int32_t** max_shapes = NULL;
      int32_t** opt_shapes = NULL;
      PD_Bool disable_trt_plugin_fp16 = 0;

      for (size_t idx = 0; idx < gpu_eas.ArraySize(); idx++) {
        triton::common::TritonJson::Value ea;
        THROW_IF_BACKEND_MODEL_ERROR(gpu_eas.IndexAsObject(idx, &ea));
        std::string name;
        THROW_IF_BACKEND_MODEL_ERROR(ea.MemberAsString("name", &name));

        if (name == "tensorrt") {
          int64_t workspace_size = 1 << 30;
          int32_t max_batch_size = 1;
          int32_t min_subgraph_size = 3;
          PD_PrecisionType precision = PD_PRECISION_FLOAT32;
          PD_Bool use_static = 0;
          PD_Bool use_calib_mode = 1;
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
                  precision = PD_PRECISION_FLOAT32;
                } else if (value_string == "trt_fp16") {
                  precision = PD_PRECISION_HALF;
                } else if (value_string == "trt_int8") {
                  precision = PD_PRECISION_INT8;
                } else {
                  TRITONSERVER_Error* error = TRITONSERVER_ErrorNew(
                      TRITONSERVER_ERROR_INVALID_ARG,
                      std::string(
                          "unknown precision type '" + value_string +
                          "' is provided. Available choices are [trt_fp32, "
                          "trt_fp16, trt_int8]")
                          .c_str());
                  THROW_IF_BACKEND_MODEL_ERROR(error);
                }
              } else if (param_key == "min_graph_size") {
                THROW_IF_BACKEND_MODEL_ERROR(
                    params.MemberAsString(param_key.c_str(), &value_string));
                THROW_IF_BACKEND_MODEL_ERROR(
                    ParseIntValue(value_string, &min_subgraph_size));
              } else if (param_key == "workspace_size") {
                THROW_IF_BACKEND_MODEL_ERROR(
                    params.MemberAsString(param_key.c_str(), &value_string));
                THROW_IF_BACKEND_MODEL_ERROR(
                    ParseLongLongValue(value_string, &workspace_size));
              } else if (param_key == "max_batch_size") {
                THROW_IF_BACKEND_MODEL_ERROR(
                    params.MemberAsString(param_key.c_str(), &value_string));
                THROW_IF_BACKEND_MODEL_ERROR(
                    ParseIntValue(value_string, &max_batch_size));
              } else if (param_key == "enable_tensorrt_oss") {
                bool tensorrt_oss_enabled = false;
                THROW_IF_BACKEND_MODEL_ERROR(
                    params.MemberAsString(param_key.c_str(), &value_string));
                THROW_IF_BACKEND_MODEL_ERROR(
                    ParseBoolValue(value_string, &tensorrt_oss_enabled));
                if(tensorrt_oss_enabled) {
                  PD_ConfigEnableVarseqlen(pd_config);
                }
              } else if (param_key == "is_dynamic") {
                THROW_IF_BACKEND_MODEL_ERROR(
                    params.MemberAsString(param_key.c_str(), &value_string));
                THROW_IF_BACKEND_MODEL_ERROR(
                    ParseBoolValue(value_string, &pd_trt_config_.is_dynamic_));
              } else if (param_key == "disenable_trt_tune") {
                THROW_IF_BACKEND_MODEL_ERROR(
                    params.MemberAsString(param_key.c_str(), &value_string));
                THROW_IF_BACKEND_MODEL_ERROR(
                    ParseBoolValue(value_string, &pd_trt_config_.disenable_trt_tune_));
              } else if (param_key == "disable_trt_plugin_fp16") {
                bool tmp_value;
                THROW_IF_BACKEND_MODEL_ERROR(
                    params.MemberAsString(param_key.c_str(), &value_string));
                THROW_IF_BACKEND_MODEL_ERROR(
                    ParseBoolValue(value_string, &tmp_value));
                disable_trt_plugin_fp16 = (PD_Bool)tmp_value;
              } 
              else {
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
          PD_ConfigEnableTensorRtEngine(pd_config, workspace_size, max_batch_size,
                                        min_subgraph_size, precision, use_static,
                                        use_calib_mode);
        } else if (
            name == "min_shape" or name == "max_shape" or name == "opt_shape") {
          triton::common::TritonJson::Value params;
          if (ea.Find("parameters", &params)) {
            std::vector<std::string> input_names;
            THROW_IF_BACKEND_MODEL_ERROR(params.Members(&input_names));
            if(names_num != 0) {
              if (names_num != input_names.size()) {
                throw BackendModelException(TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string(
                    "The number of min_shape and max_shape are different," +
                    name + " num:" + std::to_string(names_num) + 
                    " vs " + std::to_string(input_names.size()))
                      .c_str())));
              }
            } else {
              names_num = input_names.size();
              input_tensor_names = new char*[names_num];
              for (size_t index = 0u; index < names_num; ++index) {
                input_tensor_names[index] = new char[input_names[index].size() + 1];
                memcpy(input_tensor_names[index],
                       input_names[index].c_str(),
                       input_names[index].size() + 1);
              }
              shapes_num = new size_t[names_num];
              min_shapes = new int32_t*[names_num];
              max_shapes = new int32_t*[names_num];
              opt_shapes = new int32_t*[names_num];
            }
            size_t index = 0;
            for (const auto& input_name : input_names) {
              std::string str_shape;
              THROW_IF_BACKEND_MODEL_ERROR(
                  params.MemberAsString(input_name.c_str(), &str_shape));
              std::vector<int32_t> shapes =  
                    TRITONPADDLE_Shape(str_shape).CompatibleShape();
              if (name == "min_shape") {
                shapes_num[index] = shapes.size();
                min_shapes[index] = new int32_t[shapes.size()];
                for (size_t ishape = 0; ishape < shapes.size(); ++ishape) {    \
                  min_shapes[index][ishape] = shapes[index];                       \
                }
                pd_trt_config_.min_shapes_[input_name] = shapes;
              } else if (name == "max_shape") {
                max_shapes[index] = new int32_t[shapes.size()];
                for (size_t ishape = 0; ishape < shapes.size(); ++ishape) {    \
                  max_shapes[index][ishape] = shapes[index];                       \
                }
                pd_trt_config_.max_shapes_[input_name] = shapes;
              } else {
                opt_shapes[index] = new int32_t[shapes.size()];
                for (size_t ishape = 0; ishape < shapes.size(); ++ishape) {    \
                  opt_shapes[index][ishape] = shapes[index];                       \
                }
                pd_trt_config_.opt_shapes_[input_name] = shapes;            
              }
              index += 1;
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
      if(names_num > 0) {
        PD_ConfigSetTrtDynamicShapeInfo(pd_config, names_num, 
                                        const_cast<const char**>(input_tensor_names),
                                        shapes_num, min_shapes, max_shapes, opt_shapes,
                                        disable_trt_plugin_fp16);
      }
      delete[] input_tensor_names;
      delete shapes_num;
      delete[] min_shapes;
      delete[] max_shapes;
      delete[] opt_shapes;
    }
  }
}

TRITONSERVER_Error*
ModelState::PrintModelConfig() {
  // We have the json DOM for the model configuration...
  common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  return nullptr;  // success
}

TRITONSERVER_Error* ModelState::CollectShapeRun(PD_Predictor* predictor,
                                const std::unordered_map<std::string, std::vector<int>>& shape) {
  PaddleTensorInfoMap input_tensor_infos;
  RETURN_IF_ERROR(InputInfos(predictor, input_tensor_infos));
  std::vector<PD_Tensor*> input_tensors;
  for(auto& t : input_tensor_infos) {
    auto& name = t.first;
    auto& input_info = t.second;
    RETURN_ERROR_IF_TRUE(
      shape.find(name) == shape.end(),
      TRITONSERVER_ERROR_UNAVAILABLE,
      std::string("Paddle Input name [") + name +
        "] is not one of the trt dynamic_shape");

    PD_Tensor* pd_tensor = PD_PredictorGetInputHandle(predictor, name.c_str());
    input_tensors.push_back(pd_tensor);
    auto shape_value = shape.at(name);
    int shape_num = std::accumulate(shape_value.begin(), shape_value.end(), 1,
                                    std::multiplies<int>());
    PD_TensorReshape(pd_tensor, shape_value.size(), shape_value.data());
    switch (input_info.type_) {
      case  PD_DATA_FLOAT32: {
        std::vector<float> input_data(shape_num, 1.0);
        PD_TensorCopyFromCpuFloat(pd_tensor, input_data.data());
        break;
      }
      case PD_DATA_INT32: {
        std::vector<int> input_data(shape_num, 1);
        PD_TensorCopyFromCpuInt32(pd_tensor, input_data.data());
        break;
      }
      case PD_DATA_INT64: {
        std::vector<int64_t> input_data(shape_num, 1);
        PD_TensorCopyFromCpuInt64(pd_tensor, input_data.data());
        break;
      }
      case PD_DATA_INT8: {
        std::vector<int8_t> input_data(shape_num, 1);
        PD_TensorCopyFromCpuInt8(pd_tensor, input_data.data());
        break;
      }
      case PD_DATA_UINT8: {
        std::vector<uint8_t> input_data(shape_num, 1);
        PD_TensorCopyFromCpuUint8(pd_tensor, input_data.data());
        break;
      }
      default: {
        RETURN_ERROR_IF_TRUE(
          true,
          TRITONSERVER_ERROR_UNAVAILABLE,
          std::string("input data Paddle backend only supports FP32/INT32/INT64/INT8?UINT8 currently"));
        break;
      }
    }
  }
  PD_PredictorRun(predictor);
  for(auto& pd_tensor : input_tensors){
    PD_TensorDestroy(pd_tensor);
  }
  input_tensors.clear();
  return nullptr;
}

TRITONSERVER_Error*
ModelState::CollectTensorRtShapeRange(std::string* model_path, std::string* params_path,
                                      const char* range_info_path, int32_t device_id) {
  PD_Config* pd_config = PD_ConfigCreate();
  PD_ConfigSetModel(pd_config,
                    model_path->c_str(),
                    params_path->c_str());
  PD_ConfigCollectShapeRangeInfo(pd_config, range_info_path);
  PD_Predictor* predictor = PD_PredictorCreate(pd_config);
  RETURN_IF_ERROR(CollectShapeRun(predictor, pd_trt_config_.min_shapes_));
  RETURN_IF_ERROR(CollectShapeRun(predictor, pd_trt_config_.max_shapes_));
  RETURN_IF_ERROR(CollectShapeRun(predictor, pd_trt_config_.opt_shapes_));
  PD_ConfigDestroy(pd_config); 
  PD_PredictorDestroy(predictor);
  return nullptr;
}

TRITONSERVER_Error* ModelState::LoadModel(
    const std::string& artifact_name,
    const TRITONSERVER_InstanceGroupKind instance_group_kind,
    const int32_t instance_group_device_id, std::string* model_path,
    std::string* params_path, cudaStream_t stream,
    PD_PlaceType* pd_place_type, PD_Predictor** predictor) {
  
  // Paddle Backend creation is not thread-safe, so multiple creations
  // are serialized with a global lock.
  // The Clone interface can be invoked only when the main_runtime_ is created.
  static std::mutex global_context_mu;
  std::lock_guard<std::mutex> glock(global_context_mu);

  auto dir_path = JoinPath({RepositoryPath(), std::to_string(Version())});

  if (!artifact_name.empty()) {
    *model_path = JoinPath({dir_path, artifact_name});
  } else {
    *model_path = JoinPath({dir_path, "model.pdmodel"});
    *params_path = JoinPath({dir_path, "model.pdiparams"});
  }

  // If the model path is a directory then the actual model is
  // <dir>/model.pdmodel.
  {
    bool is_dir;
    RETURN_IF_ERROR(IsDirectory(*model_path, &is_dir));
    if (is_dir) {
      *model_path = JoinPath({*model_path, "model.pdmodel"});
      *params_path = JoinPath({*params_path, "model.pdiparams"});
    }
  }

  {
    bool exists;
    RETURN_IF_ERROR(FileExists(*model_path, &exists));
    RETURN_ERROR_IF_FALSE(
          exists, TRITONSERVER_ERROR_UNAVAILABLE,
          std::string("unable to find model: '") + *model_path +
              "' for model instance '" + Name() + "'");
    
    RETURN_IF_ERROR(FileExists(*params_path, &exists));
    RETURN_ERROR_IF_FALSE(
          exists, TRITONSERVER_ERROR_UNAVAILABLE,
          std::string("unable to find params: '") + *params_path +
              "' for model instance '" + Name() + "'");
  
    PD_ConfigSetModel(pd_config_.get(),
                      model_path->c_str(),
                      params_path->c_str());
  }

// GPU
#ifdef TRITON_ENABLE_GPU
  if ((instance_group_kind == TRITONSERVER_INSTANCEGROUPKIND_GPU) ||
      (instance_group_kind == TRITONSERVER_INSTANCEGROUPKIND_AUTO)) {
    // PD_PRECISION_HALF
    *pd_place_type = PD_PLACE_GPU;
    PD_ConfigEnableUseGpu(pd_config_.get(), 100,
                          instance_group_device_id,
                          PD_PRECISION_FLOAT32);
    PD_ConfigSetExecStream(pd_config_.get(), (void*)stream);
    if(PD_ConfigTensorRtDynamicShapeEnabled(pd_config_.get()) == 1) {
      auto range_info_path = triton::backend::JoinPath({dir_path, "shape_range_info.pbtxt"});
      if (!pd_trt_config_.disenable_trt_tune_) {
          CollectTensorRtShapeRange(model_path, params_path,
                                    range_info_path.c_str(), instance_group_device_id);
        }
        PD_ConfigEnableTunedTensorRtDynamicShape(pd_config_.get(),
                                                 range_info_path.c_str(), 1);
    }
  } else {
    *pd_place_type = PD_PLACE_CPU;
    PD_ConfigDisableGpu(pd_config_.get());
  }
#else
  *pd_place_type = PD_PLACE_CPU;
  PD_ConfigDisableGpu(pd_config_.get());
#endif  // TRITON_ENABLE_GPU
  PD_ConfigSwitchIrOptim(pd_config_.get(), 1);
  PD_ConfigEnableMemoryOptim(pd_config_.get(), 1);

  *predictor = PD_PredictorCreate(pd_config_.get());
  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::AutoCompleteConfig()
{
  // If the model configuration already specifies inputs and outputs
  // then don't perform any auto-completion.
  size_t input_cnt = 0;
  size_t output_cnt = 0;
  {
    triton::common::TritonJson::Value inputs;
    if (ModelConfig().Find("input", &inputs)) {
      input_cnt = inputs.ArraySize();
    }

    triton::common::TritonJson::Value config_batch_inputs;
    if (ModelConfig().Find("batch_input", &config_batch_inputs)) {
      input_cnt += config_batch_inputs.ArraySize();
    }

    triton::common::TritonJson::Value outputs;
    if (ModelConfig().Find("output", &outputs)) {
      output_cnt = outputs.ArraySize();
    }
  }

  if ((input_cnt > 0) && (output_cnt > 0)) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("skipping model configuration auto-complete for '") +
         Name() + "': inputs and outputs already specified")
            .c_str());
    return nullptr;  // success
  }

  std::string artifact_name;
  RETURN_IF_ERROR(
      ModelConfig().MemberAsString("default_model_filename", &artifact_name));

  std::unique_ptr<PD_Predictor, PD_PredictorDeleter> pd_predictor;
  {
    TRITONSERVER_InstanceGroupKind kind = TRITONSERVER_INSTANCEGROUPKIND_CPU;

#ifdef TRITON_ENABLE_GPU
    triton::common::TritonJson::Value instance_group;
    ModelConfig().Find("instance_group", &instance_group);

    // Earlier in the model lifecycle, device checks for the instance group
    // have already occurred. If at least one instance group with
    // "kind" = "KIND_GPU" then allow model to use GPU else autocomplete to
    // "KIND_CPU"
    for (size_t i = 0; i < instance_group.ArraySize(); ++i) {
      triton::common::TritonJson::Value instance_obj;
      RETURN_IF_ERROR(instance_group.IndexAsObject(i, &instance_obj));

      triton::common::TritonJson::Value instance_group_kind;
      instance_obj.Find("kind", &instance_group_kind);
      std::string kind_str;
      RETURN_IF_ERROR(instance_group_kind.AsString(&kind_str));

      if (kind_str == "KIND_GPU") {
        kind = TRITONSERVER_INSTANCEGROUPKIND_GPU;
        break;
      }
    }
#endif  // TRITON_ENABLE_GPU

    PD_Predictor* sptr = nullptr;
    std::string model_path;
    std::string params_path;
    PD_PlaceType pd_place_type;
    RETURN_IF_ERROR(LoadModel(
        artifact_name, kind, 0, &model_path, &params_path,
        nullptr, &pd_place_type, &sptr));
    pd_predictor.reset(sptr);
  }
  PaddleTensorInfoMap input_tensor_infos;
  RETURN_IF_ERROR(
      InputInfos(pd_predictor.get(), input_tensor_infos));
  PaddleTensorInfoMap output_tensor_infos;
  RETURN_IF_ERROR(
      OutputInfos(pd_predictor.get(), output_tensor_infos));
  RETURN_IF_ERROR(
      AutoCompleteMaxBatch(input_tensor_infos, output_tensor_infos));
  if (input_cnt == 0) {
    RETURN_IF_ERROR(AutoCompleteIO("input", input_tensor_infos));
  }
  if (output_cnt == 0) {
    RETURN_IF_ERROR(AutoCompleteIO("output", output_tensor_infos));
  }

  if (TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_VERBOSE)) {
    triton::common::TritonJson::WriteBuffer buffer;
    RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("post auto-complete:\n") + buffer.Contents()).c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::AutoCompleteMaxBatch(
    const PaddleTensorInfoMap& input_tensor_infos,
    const PaddleTensorInfoMap& output_tensor_infos) {
  // Determine if the model can potentially support batching. All
  // input and output tensors must have a variable first dimension.
  bool can_support_batching = true;
  {
    for (const auto& io_info : input_tensor_infos) {
      const auto& dims = io_info.second.dims_;
      if ((dims.size() == 0) || (dims[0] != -1)) {
        can_support_batching = false;
      }
    }
    for (const auto& io_info : output_tensor_infos) {
      const auto& dims = io_info.second.dims_;
      if ((dims.size() == 0) || (dims[0] != -1)) {
        can_support_batching = false;
      }
    }
  }

  // Set max-batch-size to 1 if we have determined that batching is
  // supported and max-batch-size is not specified. We need to update
  // the configuration itself as well as the cached value we have already
  // initialized in the model state.
  if (can_support_batching) {
    if (MaxBatchSize() == 0) {
      int default_max_batch_size = 0;
      {
        TRITONBACKEND_Backend* backend;
        THROW_IF_BACKEND_INSTANCE_ERROR(
            TRITONBACKEND_ModelBackend(TritonModel(), &backend));
        void* state;
        THROW_IF_BACKEND_INSTANCE_ERROR(
            TRITONBACKEND_BackendState(backend, &state));
        default_max_batch_size = reinterpret_cast<BackendConfiguration*>(state)
                                     ->default_max_batch_size_;
      }
      int max_batch_size = std::max(default_max_batch_size, 0);

      triton::common::TritonJson::Value mbs_value;
      ModelConfig().Find("max_batch_size", &mbs_value);
      mbs_value.SetInt(max_batch_size);
      SetMaxBatchSize(max_batch_size);

      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN,
          (std::string(
               "autofilled max_batch_size to " +
               std::to_string(max_batch_size) + " for model '") +
           Name() +
           "' since batching is supporrted but no max_batch_size is "
           "specified "
           "in model configuration. Must specify max_batch_size to utilize "
           "autofill with a larger max batch size")
              .c_str());
    }

    // Check to see if we need to turn on dynamic batching
    // since model supports batching
    if (MaxBatchSize() > 1) {
      triton::common::TritonJson::Value value;
      bool found_sequence_batching =
          ModelConfig().Find("sequence_batching", &value);
      bool found_dynamic_batching =
          ModelConfig().Find("dynamic_batching", &value);
      if (!found_sequence_batching && !found_dynamic_batching) {
        triton::common::TritonJson::Value dynamic_batching(
            ModelConfig(), triton::common::TritonJson::ValueType::OBJECT);
        RETURN_IF_ERROR(
            ModelConfig().Add("dynamic_batching", std::move(dynamic_batching)));
      }
    }

  } else if (MaxBatchSize() != 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("autofill failed for model '") + Name() +
         "': model does not support batching while non-zero max_batch_size"
         " is specified")
            .c_str());
  }

  return nullptr;  // success

}

TRITONSERVER_Error*
ModelState::AutoCompleteIO(const char* key, const PaddleTensorInfoMap& io_infos)
{
  triton::common::TritonJson::Value existing_ios;
  bool found_ios = ModelConfig().Find(key, &existing_ios);

  triton::common::TritonJson::Value ios(
      ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
  for (const auto& io_info : io_infos) {
    triton::common::TritonJson::Value io(
        ModelConfig(), triton::common::TritonJson::ValueType::OBJECT);
    RETURN_IF_ERROR(io.AddString("name", io_info.first));
    RETURN_IF_ERROR(io.AddString(
        "data_type", PaddleDataTypeToModelConfigDataType(io_info.second.type_)));

    // The model signature supports batching then the first dimension
    // is -1 and should not appear in the model configuration 'dims'
    // that we are creating.
    const auto& io_info_dims = io_info.second.dims_;
    triton::common::TritonJson::Value dims(
        ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
    for (size_t i = (MaxBatchSize() > 0) ? 1 : 0; i < io_info_dims.size();
         ++i) {
      RETURN_IF_ERROR(dims.AppendInt(io_info_dims[i]));
    }

    // If dims are empty then must use a reshape...
    if (dims.ArraySize() == 0) {
      RETURN_IF_ERROR(dims.AppendInt(1));
      triton::common::TritonJson::Value reshape(
          ModelConfig(), triton::common::TritonJson::ValueType::OBJECT);
      triton::common::TritonJson::Value reshape_dims(
          ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
      RETURN_IF_ERROR(reshape.Add("shape", std::move(reshape_dims)));
      RETURN_IF_ERROR(io.Add("reshape", std::move(reshape)));
    }
    RETURN_IF_ERROR(io.Add("dims", std::move(dims)));
    RETURN_IF_ERROR(ios.Append(std::move(io)));
  }

  if (found_ios) {
    existing_ios.Swap(ios);
  } else {
    ModelConfig().Add(key, std::move(ios));
  }

  return nullptr;  // success
}

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState();

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  // Execute...
  void ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);
  void ReleaseRunResources();
  TRITONSERVER_Error* ValidateInputs(const size_t expected_input_cnt);
  TRITONSERVER_Error* ValidateOutputs();
  TRITONSERVER_Error* Run(
      std::vector<TRITONBACKEND_Response*>* responses,
      const uint32_t response_count);
  TRITONSERVER_Error* SetInputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      BackendInputCollector* collector, bool* cuda_copy);
  TRITONSERVER_Error* ReadOutputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);

  ModelState* model_state_;

  // The full path to the model file.
  std::string model_path_;
  std::string params_path_;

  PD_Predictor* pd_predictor_;
  PD_PlaceType pd_place_type_;
  std::unordered_map<std::string, TRITONSERVER_DataType> output_names_;
  std::unordered_map<std::string, PD_Tensor*> input_tensors_;
  std::unordered_map<std::string, PD_Tensor*> output_tensors_;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state) {
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

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state), pd_predictor_(nullptr),
      pd_place_type_(PD_PLACE_CPU), output_names_({}),
      input_tensors_({}), output_tensors_({}) {
  THROW_IF_BACKEND_INSTANCE_ERROR(model_state->LoadModel(
      ArtifactFilename(), Kind(), DeviceId(), &model_path_, &params_path_,
      CudaStream(), &pd_place_type_, &pd_predictor_));

  size_t expected_input_cnt = 0;
  {
    triton::common::TritonJson::Value inputs;
    if (model_state->ModelConfig().Find("input", &inputs)) {
      expected_input_cnt = inputs.ArraySize();
    }

    triton::common::TritonJson::Value config_batch_inputs;
    if (model_state->ModelConfig().Find("batch_input", &config_batch_inputs)) {
      expected_input_cnt += config_batch_inputs.ArraySize();
    }
  }

  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateInputs(expected_input_cnt));
  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateOutputs());
}

ModelInstanceState::~ModelInstanceState()
{
  for(auto& t : input_tensors_) {
    auto& input_tensor = t.second;
    PD_TensorDestroy(input_tensor);
  }
  input_tensors_.clear();
  
  for(auto& t : output_tensors_) {
    auto& output_tensor = t.second;
    PD_TensorDestroy(output_tensor);
  }
  output_tensors_.clear();

  PD_PredictorDestroy(pd_predictor_);
}

TRITONSERVER_Error*
ModelInstanceState::ValidateInputs(const size_t expected_input_cnt)
{
  std::set<std::string> input_tensor_names;
  {
    PD_OneDimArrayCstr* c_names = PD_PredictorGetInputNames(pd_predictor_);
    for(size_t i = 0; i < c_names->size; ++i) {
      std::string name(c_names->data[i]);
      input_tensor_names.emplace(std::move(name));
    }
    PD_OneDimArrayCstrDestroy(c_names);
  }
  PaddleTensorInfoMap input_tensor_infos;
  RETURN_IF_ERROR(InputInfos(pd_predictor_, input_tensor_infos));

  if (input_tensor_infos.size() != expected_input_cnt) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("unable to load model '") + model_state_->Name() +
         "', configuration expects " + std::to_string(expected_input_cnt) +
         " inputs, model provides " + std::to_string(input_tensor_infos.size()))
            .c_str());
  }

  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(model_state_->ModelConfig().MemberAsArray("input", &ios));
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));

    auto iit = input_tensor_infos.find(io_name);
    if (iit == input_tensor_infos.end()) {
      RETURN_IF_ERROR(CheckAllowedModelInput(io, input_tensor_names));
    }

    auto pd_data_type = ModelConfigDataTypeToPaddleDataType(io_dtype);
    if (pd_data_type == PD_DATA_UNK) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unsupported datatype ") + io_dtype + " for input '" +
           io_name + "' for model '" + model_state_->Name() + "'")
              .c_str());
    } else if (pd_data_type != iit->second.type_) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_state_->Name() +
           "', configuration expects datatype " + io_dtype + " for input '" +
           io_name + "', model provides TYPE_" +
           TRITONSERVER_DataTypeString(
               ConvertFromPaddleDataType(iit->second.type_)))
              .c_str());
    }

    // If a reshape is provided for the input then use that when
    // validating that the model matches what is expected.
    std::vector<int64_t> dims;
    triton::common::TritonJson::Value reshape;
    if (io.Find("reshape", &reshape)) {
      RETURN_IF_ERROR(ParseShape(reshape, "shape", &dims));
    } else {
      RETURN_IF_ERROR(ParseShape(io, "dims", &dims));
    }

    triton::common::TritonJson::Value allow_ragged_batch_json;
    bool allow_ragged_batch = false;
    if (io.Find("allow_ragged_batch", &allow_ragged_batch_json)) {
      RETURN_IF_ERROR(allow_ragged_batch_json.AsBool(&allow_ragged_batch));
    }
    if (allow_ragged_batch) {
      const std::vector<int64_t>& model_shape = iit->second.dims_;
      // Make sure the input has shpae [-1]
      if ((model_shape.size() != 1) || (model_shape[0] != WILDCARD_DIM)) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("unable to load model '") + model_state_->Name() +
             "', configuration expects model provides input with shape [-1]  "
             "for ragged input '" +
             io_name + "', model provides " + ShapeToString(model_shape))
                .c_str());
      }
    } else {
      RETURN_IF_ERROR(CompareDimsSupported(
          model_state_->Name(), io_name, iit->second.dims_, dims,
          model_state_->MaxBatchSize(), false /* compare_exact */));
    }

    // Init input tensors
    PD_Tensor* input_tensor = PD_PredictorGetInputHandle(pd_predictor_, io_name.c_str());
    input_tensors_.emplace(std::move(io_name), input_tensor);
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ValidateOutputs()
{
  std::set<std::string> output_tensor_names;
  {
    PD_OneDimArrayCstr* c_names = PD_PredictorGetInputNames(pd_predictor_);
    for(size_t i = 0; i < c_names->size; ++i) {
      std::string name(c_names->data[i]);
      output_tensor_names.emplace(std::move(name));
    }
    PD_OneDimArrayCstrDestroy(c_names);
  }

  PaddleTensorInfoMap output_tensor_infos;
  RETURN_IF_ERROR(InputInfos(pd_predictor_, output_tensor_infos));

  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(model_state_->ModelConfig().MemberAsArray("output", &ios));
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));

    auto iit = output_tensor_infos.find(io_name);
    if (iit == output_tensor_infos.end()) {
      RETURN_IF_ERROR(CheckAllowedModelOutput(io, output_tensor_names));
    }

    auto pd_data_type = ModelConfigDataTypeToPaddleDataType(io_dtype);
    if (pd_data_type == PD_DATA_UNK) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unsupported datatype ") + io_dtype + " for output '" +
           io_name + "' for model '" + model_state_->Name() + "'")
              .c_str());
    } else if (pd_data_type != iit->second.type_) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_state_->Name() +
           "', configuration expects datatype " + io_dtype + " for output '" +
           io_name + "', model provides TYPE_" +
           TRITONSERVER_DataTypeString(
               ConvertFromPaddleDataType(iit->second.type_)))
              .c_str());
    }

    // If a reshape is provided for the input then use that when
    // validating that the model matches what is expected.
    std::vector<int64_t> dims;
    triton::common::TritonJson::Value reshape;
    if (io.Find("reshape", &reshape)) {
      RETURN_IF_ERROR(ParseShape(reshape, "shape", &dims));
    } else {
      RETURN_IF_ERROR(ParseShape(io, "dims", &dims));
    }

    // The batch output shape doesn't necessarily match the model
    if (model_state_->FindBatchOutput(io_name) == nullptr) {
      RETURN_IF_ERROR(CompareDimsSupported(
          model_state_->Name(), io_name, iit->second.dims_, dims,
          model_state_->MaxBatchSize(), true /* compare_exact */));
    }

    // Init output tensors
    output_names_.emplace(std::move(io_name),
                  TRITONSERVER_StringToDataType(io_dtype.c_str()));
    PD_Tensor* output_tensor = PD_PredictorGetInputHandle(pd_predictor_, io_name.c_str());
    output_tensors_.emplace(std::move(io_name), output_tensor);
  }

  return nullptr;  // success
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
  for (size_t i = 0; i < request_count; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "null request given to Paddle backend for '" + Name() +
                  "'")
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

  // If there are no valid payloads then no need to run the inference.
  if (total_batch_size == 0) {
    return;
  }

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests.
  if ((total_batch_size != 1) && (total_batch_size > (size_t)max_batch_size)) {
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
  // response unique_ptr will then be nullptr). The request object
  // itself will not be released until after all inferencing is done
  // (below) as we may need to access the request object when
  // determine how to process outputs (for example, even if we don't
  // need the outputs for a request that has an error, we do need to
  // know the size of those outputs associated with the request so we
  // can skip them in the output tensors).
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);
  bool all_response_failed = false;

  for (size_t i = 0; i < request_count; i++) {
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

  bool cuda_copy = false;
  BackendInputCollector collector(
      requests, request_count, &responses, model_state_->TritonMemoryManager(),
      model_state_->EnablePinnedInput(), CudaStream(), nullptr, nullptr, 0,
      HostPolicyName().c_str());
  RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
      responses, request_count, all_response_failed,
      SetInputTensors(
          total_batch_size, requests, request_count, &responses, &collector,
          &cuda_copy));

  // Wait for any in-flight input tensor copies to complete.
#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(CudaStream());
  }
#endif

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  if (!all_response_failed) {
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        Run(&responses, request_count));
  }

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  if (!all_response_failed) {
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        ReadOutputTensors(
            total_batch_size, requests, request_count, &responses));
  }

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
          "failed to send paddle backend response");
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

  if (!all_response_failed) {
    // Report the entire batch statistics.
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportBatchStatistics(
            TritonModelInstance(), total_batch_size, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting batch request statistics");
  }
}

TRITONSERVER_Error*
ModelInstanceState::SetInputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses,
    BackendInputCollector* collector, bool* cuda_copy) {
  const int max_batch_size = model_state_->MaxBatchSize();

  // All requests must have equally-sized input tensors so use any
  // request as the representative for the input tensors.
  uint32_t input_count;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(requests[0], &input_count));

  TRITONSERVER_MemoryType memory_type;
  int64_t memory_type_id;
  if (pd_place_type_ == PD_PLACE_GPU) {
    memory_type = TRITONSERVER_MEMORY_GPU;
    memory_type_id = DeviceId();
  } else {
    memory_type = TRITONSERVER_MEMORY_CPU;
    memory_type_id = 0;
  }

  for (uint32_t input_idx = 0; input_idx < input_count; input_idx++) {
    TRITONBACKEND_Input* input;
    RETURN_IF_ERROR(
        TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input));

    const char* input_name;
    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, &input_name, &input_datatype, &input_shape, &input_dims_count,
        nullptr, nullptr));

    std::vector<int64_t> batchn_shape;
    // For a ragged input tensor, the tensor shape should be
    // the flatten shape of the whole batch
    if (StateForModel()->IsInputRagged(input_name)) {
      batchn_shape = std::vector<int64_t>{0};
      for (size_t idx = 0; idx < request_count; idx++) {
        TRITONBACKEND_Input* input;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]),
            TRITONBACKEND_RequestInput(requests[idx], input_name, &input));
        const int64_t* input_shape;
        uint32_t input_dims_count;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]), TRITONBACKEND_InputProperties(
                                      input, nullptr, nullptr, &input_shape,
                                      &input_dims_count, nullptr, nullptr));

        batchn_shape[0] += GetElementCount(input_shape, input_dims_count);
      }
    }
    // The shape for the entire input batch, [total_batch_size, ...]
    else {
      batchn_shape =
          std::vector<int64_t>(input_shape, input_shape + input_dims_count);
      if (max_batch_size != 0) {
        batchn_shape[0] = total_batch_size;
      }
    }

    int64_t batchn_byte_size = GetByteSize(input_datatype, batchn_shape);

    // The input must be in contiguous CPU memory. Use appropriate
    // allocator info to bind inputs to the right device. .i.e bind inputs
    // to GPU if they are being provided on GPU.
    std::string sname(input_name);
    PD_Tensor* & pd_tensor = input_tensors_[sname];
    auto pd_shapes = TRITONPADDLE_Shape(batchn_shape).CompatibleShape();
    PD_TensorReshape(pd_tensor, pd_shapes.size(), pd_shapes.data());
    char* input_ptr;
    RETURN_IF_ERROR(PaddleInputMutableData(
                      pd_tensor, input_datatype,
                      pd_place_type_, &input_ptr));

    collector->ProcessTensor(input_name, input_ptr, batchn_byte_size,
                             memory_type, memory_type_id);
  }

  // Finalize...
  *cuda_copy |= collector->Finalize();
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::Run(
    std::vector<TRITONBACKEND_Response*>* responses,
    const uint32_t response_count)
{
  PD_PredictorRun(pd_predictor_);
#ifdef TRITON_ENABLE_GPU
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    cudaStreamSynchronize(CudaStream());
  }
#endif
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::ReadOutputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  BackendOutputResponder responder(
      requests, request_count, responses, model_state_->TritonMemoryManager(),
      model_state_->MaxBatchSize() > 0, model_state_->EnablePinnedInput(),
      CudaStream());

  // Use to hold string output contents
  bool cuda_copy = false;

  TRITONSERVER_MemoryType memory_type;
  int64_t memory_type_id;

  for(auto& t : output_names_) {
    auto& name = t.first;
    auto& data_type = t.second;
    PD_Tensor* & pd_tensor = output_tensors_[name];
    char* out_ptr;
    PD_PlaceType out_place_type;
    std::vector<int64_t> output_shapes;
    RETURN_IF_ERROR(PaddleOutputData(
      pd_tensor, data_type, &out_place_type, 
      output_shapes, &out_ptr));
    
    if (out_place_type == PD_PLACE_GPU) {
      memory_type = TRITONSERVER_MEMORY_GPU;
      memory_type_id = DeviceId();
    } else {
      memory_type = TRITONSERVER_MEMORY_CPU;
      memory_type_id = 0;
    }

    responder.ProcessTensor(
        name, data_type,
        output_shapes, out_ptr,
        memory_type, memory_type_id);
  }

  // Finalize and wait for any pending buffer copies.
  cuda_copy |= responder.Finalize();

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }
#endif  // TRITON_ENABLE_GPU
  return nullptr;
}

////////////////

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

  triton::common::TritonJson::Value backend_config;
  TRITONSERVER_Error* err = nullptr;
  if (byte_size != 0) {
    err = backend_config.Parse(buffer, byte_size);
  }
  RETURN_IF_ERROR(err);

  std::unique_ptr<BackendConfiguration> lconfig(new BackendConfiguration());
  triton::common::TritonJson::Value cmdline;
  if (backend_config.Find("cmdline", &cmdline)) {
    triton::common::TritonJson::Value value;
    std::string value_str;
    if (cmdline.Find("default-max-batch-size", &value)) {
      RETURN_IF_ERROR(value.AsString(&value_str));
      int lvalue;
      RETURN_IF_ERROR(ParseIntValue(value_str, &lvalue));
      lconfig->default_max_batch_size_ = lvalue;
    }
  }
  RETURN_IF_ERROR(TRITONBACKEND_BackendSetState(
      backend, reinterpret_cast<void*>(lconfig.get())));

  lconfig.release();

  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
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

  RETURN_IF_ERROR(model_state->PrintModelConfig());

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
