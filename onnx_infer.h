#pragma once

#include <onnxruntime_cxx_api.h>

#include <atomic>
#include <string>
#include <utility>
#include <vector>

#include "games/game.h"

class OnnxInfer {
 public:
  struct InferProfile {
    uint64_t query_count = 0;
    uint64_t worker_batch_count = 0;
    uint64_t worker_item_count = 0;
    uint64_t worker_bind_io_ns = 0;
    uint64_t worker_input_build_ns = 0;
    uint64_t worker_ort_run_ns = 0;
    uint64_t worker_ort_output_fetch_ns = 0;
    uint64_t worker_softmax_ns = 0;
    uint64_t worker_writeback_ns = 0;
    uint64_t worker_postprocess_ns = 0;
  };

  OnnxInfer(
      const std::string& model_path,
      bool use_cuda,
      int cuda_device_id,
      int max_batch_size);
  ~OnnxInfer();

  std::pair<std::vector<float>, float> infer(const game::Board& canonical_state);
  std::vector<std::pair<std::vector<float>, float>> infer_batch(
      const std::vector<game::Board>& canonical_states);
  InferProfile profile_snapshot() const;

 private:
  void run_batch_direct(
      const std::vector<game::Board>& canonical_states,
      size_t begin,
      size_t count,
      std::vector<std::pair<std::vector<float>, float>>& outputs);

  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::Session session_;
  std::vector<std::string> input_names_holder_;
  std::vector<std::string> output_names_holder_;
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
  Ort::MemoryInfo cpu_memory_info_;
  int max_batch_size_;
  int64_t fixed_batch_size_ = -1;
  std::vector<int64_t> value_output_shape_template_;

  std::atomic<uint64_t> query_count_{0};
  std::atomic<uint64_t> worker_batch_count_{0};
  std::atomic<uint64_t> worker_item_count_{0};
  std::atomic<uint64_t> worker_bind_io_ns_{0};
  std::atomic<uint64_t> worker_input_build_ns_{0};
  std::atomic<uint64_t> worker_ort_run_ns_{0};
  std::atomic<uint64_t> worker_ort_output_fetch_ns_{0};
  std::atomic<uint64_t> worker_softmax_ns_{0};
  std::atomic<uint64_t> worker_writeback_ns_{0};
  std::atomic<uint64_t> worker_postprocess_ns_{0};
};

