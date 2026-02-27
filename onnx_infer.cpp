#include "onnx_infer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {

void softmax_logits(const float* logits, int len, float* out_probs) {
  if (len <= 0) {
    return;
  }
  float max_logit = logits[0];
  for (int i = 1; i < len; i++) {
    max_logit = std::max(max_logit, logits[i]);
  }

  double sum = 0.0;
  for (int i = 0; i < len; i++) {
    out_probs[i] =
        static_cast<float>(std::exp(logits[i] - max_logit));
    sum += out_probs[i];
  }

 if (sum <= 1e-12) {
    const float uniform = 1.0f / static_cast<float>(len);
    std::fill(out_probs, out_probs + len, uniform);
    return;
  }
  
  const float inv_sum = 1.0f / static_cast<float>(sum);
  for (int i = 0; i < len; i++) {
    out_probs[i] *= inv_sum;
  }
}

}  // namespace

OnnxInfer::OnnxInfer(
    const std::string& model_path,
    bool use_cuda,
    int cuda_device_id,
    int max_batch_size)
    : env_(ORT_LOGGING_LEVEL_WARNING, "cpp_selfplay"),
      session_options_(),
      session_(nullptr),
      cpu_memory_info_(
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      max_batch_size_(std::max(1, max_batch_size)) {
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  if (use_cuda) {
    try {
      OrtCUDAProviderOptionsV2* cuda_options = nullptr;
      Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda_options));
      std::unique_ptr<OrtCUDAProviderOptionsV2, void (*)(OrtCUDAProviderOptionsV2*)>
          holder(cuda_options, [](OrtCUDAProviderOptionsV2* p) {
            if (p != nullptr) {
              Ort::GetApi().ReleaseCUDAProviderOptions(p);
            }
          });
      const char* keys[] = {"device_id"};
      const std::string device_id_str = std::to_string(cuda_device_id);
      const char* values[] = {device_id_str.c_str()};
      Ort::ThrowOnError(
          Ort::GetApi().UpdateCUDAProviderOptions(cuda_options, keys, values, 1));
      Ort::ThrowOnError(
          Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA_V2(
              session_options_, cuda_options));
      std::cout << "OnnxInfer: CUDA provider enabled (device_id=" << cuda_device_id
                << ")\n";
    } catch (const std::exception& e) {
      std::cerr
          << "OnnxInfer: failed to enable CUDA provider, fallback to CPU: "
          << e.what() << "\n";
    }
  }

  session_ = Ort::Session(env_, model_path.c_str(), session_options_);

  {
    Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(0);
    auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = tensor_info.GetShape();
    if (shape.size() != 4) {
      throw std::runtime_error("ONNX input must be rank-4 [N,C,H,W]");
    }
    if (shape[1] > 0 && shape[1] != game::kInputChannels) {
      throw std::runtime_error("ONNX channel mismatch with selected game");
    }
    if (shape[2] > 0 && shape[2] != game::kBoardSize) {
      throw std::runtime_error("ONNX height mismatch with selected game");
    }
    if (shape[3] > 0 && shape[3] != game::kBoardSize) {
      throw std::runtime_error("ONNX width mismatch with selected game");
    }
    if (!shape.empty() && shape[0] > 0) {
      fixed_batch_size_ = shape[0];
      if (fixed_batch_size_ == 1 && max_batch_size_ > 1) {
        max_batch_size_ = 1;
        std::cout
            << "OnnxInfer: fixed batch=1 model detected, forcing nn max batch size to 1\n";
      }
    }
  }
  {
    Ort::TypeInfo value_output_type_info = session_.GetOutputTypeInfo(1);
    auto value_tensor_info = value_output_type_info.GetTensorTypeAndShapeInfo();
    value_output_shape_template_ = value_tensor_info.GetShape();
    if (value_output_shape_template_.empty()) {
      value_output_shape_template_.push_back(-1);
    }
  }

  Ort::AllocatorWithDefaultOptions allocator;
  const size_t input_count = session_.GetInputCount();
  const size_t output_count = session_.GetOutputCount();
  if (input_count < 1 || output_count < 2) {
    throw std::runtime_error(
        "ONNX model must have 1 input and at least 2 outputs (policy, value)");
  }

  input_names_holder_.reserve(input_count);
  output_names_holder_.reserve(output_count);
  input_names_.reserve(input_count);
  output_names_.reserve(output_count);

  for (size_t i = 0; i < input_count; i++) {
    auto name = session_.GetInputNameAllocated(i, allocator);
    input_names_holder_.push_back(name.get());
  }
  for (size_t i = 0; i < output_count; i++) {
    auto name = session_.GetOutputNameAllocated(i, allocator);
    output_names_holder_.push_back(name.get());
  }
  for (const auto& s : input_names_holder_) {
    input_names_.push_back(s.c_str());
  }
  for (const auto& s : output_names_holder_) {
    output_names_.push_back(s.c_str());
  }
}

OnnxInfer::~OnnxInfer() = default;

void OnnxInfer::run_batch_direct(
    const std::vector<game::Board>& canonical_states,
    size_t begin,
    size_t count,
    std::vector<std::pair<std::vector<float>, float>>& outputs) {
  const int batch_size = static_cast<int>(count);
  if (batch_size <= 0) {
    return;
  }

  worker_batch_count_.fetch_add(1, std::memory_order_relaxed);
  worker_item_count_.fetch_add(static_cast<uint64_t>(count),
                               std::memory_order_relaxed);

  try {
    thread_local std::vector<float> input_data;
    thread_local std::vector<float> policy_output_data;
    thread_local std::vector<float> value_output_data;
    thread_local std::vector<float> encoded_state;

    const int encoded_size = game::encoded_state_size();
    const size_t input_elem_count =
        static_cast<size_t>(batch_size) * static_cast<size_t>(encoded_size);
    const size_t policy_elem_count =
        static_cast<size_t>(batch_size) *
        static_cast<size_t>(game::kActionSize);

    std::vector<int64_t> value_output_shape = value_output_shape_template_;
    value_output_shape[0] = batch_size;
    for (size_t i = 1; i < value_output_shape.size(); i++) {
      if (value_output_shape[i] <= 0) {
        value_output_shape[i] = 1;
      }
    }
    size_t value_elem_count = 1;
    for (int64_t d : value_output_shape) {
      value_elem_count *= static_cast<size_t>(std::max<int64_t>(d, 1));
    }
    if (value_elem_count < static_cast<size_t>(batch_size)) {
      value_elem_count = static_cast<size_t>(batch_size);
    }

    const auto input_build_start = std::chrono::steady_clock::now();
    input_data.resize(input_elem_count);
    for (int b = 0; b < batch_size; b++) {
      game::encode_state(
          canonical_states[begin + static_cast<size_t>(b)], encoded_state);
      if (static_cast<int>(encoded_state.size()) != encoded_size) {
        throw std::runtime_error("encode_state returned unexpected size");
      }
      std::memcpy(
          input_data.data() + static_cast<size_t>(b) * static_cast<size_t>(encoded_size),
          encoded_state.data(),
          static_cast<size_t>(encoded_size) * sizeof(float));
    }
    const auto input_build_end = std::chrono::steady_clock::now();
    worker_input_build_ns_.fetch_add(
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                input_build_end - input_build_start)
                .count()),
        std::memory_order_relaxed);

    policy_output_data.resize(policy_elem_count);
    value_output_data.resize(value_elem_count);

    std::array<int64_t, 4> input_shape = {
        batch_size, game::kInputChannels, game::kBoardSize, game::kBoardSize};
    std::array<int64_t, 2> policy_shape = {batch_size, game::kActionSize};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        cpu_memory_info_, input_data.data(), input_data.size(), input_shape.data(),
        input_shape.size());
    Ort::Value policy_output_tensor = Ort::Value::CreateTensor<float>(
        cpu_memory_info_, policy_output_data.data(), policy_output_data.size(),
        policy_shape.data(), policy_shape.size());
    Ort::Value value_output_tensor = Ort::Value::CreateTensor<float>(
        cpu_memory_info_, value_output_data.data(), value_output_data.size(),
        value_output_shape.data(), value_output_shape.size());

    const auto bind_start = std::chrono::steady_clock::now();
    Ort::IoBinding binding(session_);
    binding.BindInput(input_names_[0], input_tensor);
    binding.BindOutput(output_names_[0], policy_output_tensor);
    binding.BindOutput(output_names_[1], value_output_tensor);
    const auto bind_end = std::chrono::steady_clock::now();
    worker_bind_io_ns_.fetch_add(
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  bind_end - bind_start)
                                  .count()),
        std::memory_order_relaxed);

    const auto ort_run_start = std::chrono::steady_clock::now();
    session_.Run(Ort::RunOptions{nullptr}, binding);
    const auto ort_run_end = std::chrono::steady_clock::now();
    worker_ort_run_ns_.fetch_add(
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  ort_run_end - ort_run_start)
                                  .count()),
        std::memory_order_relaxed);

    const auto output_fetch_start = std::chrono::steady_clock::now();
    const float* policy_logits = policy_output_data.data();
    const float* value_ptr = value_output_data.data();
    const auto output_fetch_end = std::chrono::steady_clock::now();
    worker_ort_output_fetch_ns_.fetch_add(
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                output_fetch_end - output_fetch_start)
                .count()),
        std::memory_order_relaxed);

    const auto post_start = std::chrono::steady_clock::now();
    const size_t value_stride =
        std::max<size_t>(1, value_elem_count / static_cast<size_t>(batch_size));
    for (int b = 0; b < batch_size; b++) {
      const float* row_logits =
          policy_logits +
          static_cast<size_t>(b) * static_cast<size_t>(game::kActionSize);
      const auto softmax_start = std::chrono::steady_clock::now();
      auto& target_pair = outputs[begin + static_cast<size_t>(b)];
      target_pair.first.resize(static_cast<size_t>(game::kActionSize));
      softmax_logits(row_logits, game::kActionSize, target_pair.first.data());
      const auto softmax_end = std::chrono::steady_clock::now();
      worker_softmax_ns_.fetch_add(
          static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  softmax_end - softmax_start)
                  .count()),
          std::memory_order_relaxed);

      const auto write_start = std::chrono::steady_clock::now();
      target_pair.second = value_ptr[static_cast<size_t>(b) * value_stride];
      const auto write_end = std::chrono::steady_clock::now();
      worker_writeback_ns_.fetch_add(
          static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  write_end - write_start)
                  .count()),
          std::memory_order_relaxed);
    }
    const auto post_end = std::chrono::steady_clock::now();
    worker_postprocess_ns_.fetch_add(
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(post_end - post_start)
                .count()),
        std::memory_order_relaxed);
  } catch (const std::exception& e) {
    std::cerr << "OnnxInfer direct inference failure: " << e.what() << "\n";
    const std::vector<float> uniform(
        static_cast<size_t>(game::kActionSize),
        1.0f / static_cast<float>(game::kActionSize));
    for (size_t i = 0; i < count; i++) {
      outputs[begin + i] = {uniform, 0.0f};
    }
  }
}

std::pair<std::vector<float>, float> OnnxInfer::infer(
    const game::Board& canonical_state) {
  const auto batch_outputs = infer_batch(std::vector<game::Board>{canonical_state});
  return batch_outputs[0];
}

std::vector<std::pair<std::vector<float>, float>> OnnxInfer::infer_batch(
    const std::vector<game::Board>& canonical_states) {
  std::vector<std::pair<std::vector<float>, float>> outputs;
  if (canonical_states.empty()) {
    return outputs;
  }
  outputs.resize(canonical_states.size());
  size_t begin = 0;
  while (begin < canonical_states.size()) {
    const size_t count =
        std::min(static_cast<size_t>(max_batch_size_),
                 canonical_states.size() - begin);
    run_batch_direct(canonical_states, begin, count, outputs);
    begin += count;
  }
  query_count_.fetch_add(static_cast<uint64_t>(canonical_states.size()),
                         std::memory_order_relaxed);
  return outputs;
}

OnnxInfer::InferProfile OnnxInfer::profile_snapshot() const {
  InferProfile p;
  p.query_count = query_count_.load(std::memory_order_relaxed);
  p.worker_batch_count = worker_batch_count_.load(std::memory_order_relaxed);
  p.worker_item_count = worker_item_count_.load(std::memory_order_relaxed);
  p.worker_bind_io_ns = worker_bind_io_ns_.load(std::memory_order_relaxed);
  p.worker_input_build_ns = worker_input_build_ns_.load(std::memory_order_relaxed);
  p.worker_ort_run_ns = worker_ort_run_ns_.load(std::memory_order_relaxed);
  p.worker_ort_output_fetch_ns =
      worker_ort_output_fetch_ns_.load(std::memory_order_relaxed);
  p.worker_softmax_ns = worker_softmax_ns_.load(std::memory_order_relaxed);
  p.worker_writeback_ns = worker_writeback_ns_.load(std::memory_order_relaxed);
  p.worker_postprocess_ns = worker_postprocess_ns_.load(std::memory_order_relaxed);
  return p;
}

