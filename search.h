#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "onnx_infer.h"

struct SearchParams {
  int num_searches = 64;
  float cpuct = 2.0f;
  float temperature = 1.0f;
  float temperature_early = 1.0f;
  float temperature_halflife = 19.0f;
  float dirichlet_epsilon = 0.25f;
  float dirichlet_alpha = 0.3f;
  float target_pruning_threshold = 0.05f;
  float fpu_reduction = 0.2f;
  float c_base = 19652.0f;
  int parallel_games = 1;
  int pcr_full_search_prob = 25;

  bool use_target_pruning = true;
  bool use_fpu = true;
  bool use_dynamic_cpuct = true;
};

struct TrainingRow {
  std::vector<float> encoded_state;
  std::vector<float> policy;
  float value = 0.0f;
};

struct SelfplayStats {
  uint32_t win = 0;
  uint32_t draw = 0;
  uint32_t lose = 0;
  std::vector<std::vector<float>> average_depth_lists;
  std::vector<std::vector<float>> max_depth_lists;
  std::vector<std::vector<int8_t>> final_states;
};

struct SearchProfile {
  uint64_t games = 0;
  uint64_t game_total_ns = 0;
  uint64_t mcts_calls = 0;
  uint64_t mcts_total_ns = 0;
  uint64_t infer_calls = 0;
  uint64_t infer_total_ns = 0;
};

struct SelfplayResult {
  std::vector<TrainingRow> rows;
  SelfplayStats stats;
  SearchProfile profile;
};

SelfplayResult run_selfplay_games(
    OnnxInfer& infer,
    const SearchParams& params,
    int num_games,
    int num_threads,
    uint64_t seed);

void write_memory_file(
    const std::string& path,
    const std::vector<TrainingRow>& rows);

void write_stats_file(
    const std::string& path,
    const SelfplayStats& stats);
