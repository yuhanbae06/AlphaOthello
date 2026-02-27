#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include "onnx_infer.h"
#include "games/game.h"
#include "search.h"

namespace {

struct CliArgs {
  std::string onnx_path;
  std::string out_path;
  std::string stats_out_path;
  int games = 0;
  int searches = 64;
  int max_game_moves = 0;
  float cpuct = 2.0f;
  float temp = 1.0f;
  float temp_early = -1.0f;
  float temp_halflife = 19.0f;
  int threads = -1;
  int nn_max_batch_size = 64;
  int parallel_games = 1;
  int pcr_full_search_prob = 25;
  bool use_cuda = false;
  int cuda_device_id = 0;
  uint64_t seed = 0;
  float dirichlet_epsilon = 0.25f;
  float dirichlet_alpha = 0.3f;
  bool use_target_pruning = true;
  bool use_fpu = true;
  bool use_dynamic_cpuct = true;
  float fpu_reduction = 0.2f;
  float c_base = 19652.0f;
  float target_pruning_threshold = 0.05f;
};

void print_usage() {
  std::cout
      << "Usage: ./cpp_selfplay --onnx <model.onnx> --out <memory.bin> --games <N>"
      << " --searches <M> --cpuct <C> --temp <T> --threads <K> --seed <S>"
      << " [--temp-early <T0>] [--temp-halflife <H>]"
      << " --dirichlet-epsilon <E> --dirichlet-alpha <A>"
      << " [--parallel-games <N>]"
      << " [--pcr-full-search-prob <0..100>]"
      << " [--max-game-moves <N>]"
      << " [--nn-max-batch-size <N>]"
      << " [--use-cuda] [--cuda-device-id <ID>]"
      << " [--stats-out <stats.bin>]\n";
}

int default_threads() {
  const unsigned hw = std::thread::hardware_concurrency();
  if (hw == 0) {
    return 1;
  }
  return static_cast<int>(hw);
}

CliArgs parse_args(int argc, char** argv) {
  CliArgs args;
  for (int i = 1; i < argc; i++) {
    const std::string key = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      i++;
      return argv[i];
    };

    if (key == "--onnx") {
      args.onnx_path = next("--onnx");
    } else if (key == "--out") {
      args.out_path = next("--out");
    } else if (key == "--stats-out") {
      args.stats_out_path = next("--stats-out");
    } else if (key == "--games") {
      args.games = std::stoi(next("--games"));
    } else if (key == "--searches") {
      args.searches = std::stoi(next("--searches"));
    } else if (key == "--max-game-moves") {
      args.max_game_moves = std::stoi(next("--max-game-moves"));
    } else if (key == "--cpuct") {
      args.cpuct = std::stof(next("--cpuct"));
    } else if (key == "--temp") {
      args.temp = std::stof(next("--temp"));
    } else if (key == "--temp-early") {
      args.temp_early = std::stof(next("--temp-early"));
    } else if (key == "--temp-halflife") {
      args.temp_halflife = std::stof(next("--temp-halflife"));
    } else if (key == "--threads") {
      args.threads = std::stoi(next("--threads"));
    } else if (key == "--nn-max-batch-size") {
      args.nn_max_batch_size = std::stoi(next("--nn-max-batch-size"));
    } else if (key == "--parallel-games") {
      args.parallel_games = std::stoi(next("--parallel-games"));
    } else if (key == "--pcr-full-search-prob") {
      args.pcr_full_search_prob = std::stoi(next("--pcr-full-search-prob"));
    } else if (key == "--use-cuda") {
      args.use_cuda = true;
    } else if (key == "--cuda-device-id") {
      args.cuda_device_id = std::stoi(next("--cuda-device-id"));
    } else if (key == "--seed") {
      args.seed = static_cast<uint64_t>(std::stoull(next("--seed")));
    } else if (key == "--dirichlet-epsilon") {
      args.dirichlet_epsilon = std::stof(next("--dirichlet-epsilon"));
    } else if (key == "--dirichlet-alpha") {
      args.dirichlet_alpha = std::stof(next("--dirichlet-alpha"));
    } else if (key == "--no-target-pruning") {
      args.use_target_pruning = false;
    } else if (key == "--no-fpu") {
      args.use_fpu = false;
    } else if (key == "--no-dynamic-cpuct") {
      args.use_dynamic_cpuct = false;
    } else if (key == "--fpu-reduction") {
      args.fpu_reduction = std::stof(next("--fpu-reduction"));
    } else if (key == "--c-base") {
      args.c_base = std::stof(next("--c-base"));
    } else if (key == "--target-pruning-threshold") {
      args.target_pruning_threshold = std::stof(next("--target-pruning-threshold"));
    } else if (key == "--help" || key == "-h") {
      print_usage();
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + key);
    }
  }

  if (args.onnx_path.empty() || args.out_path.empty() || args.games <= 0) {
    throw std::runtime_error("required args missing: --onnx --out --games");
  }
  if (args.threads <= 0) {
    args.threads = default_threads();
  }
  if (args.nn_max_batch_size <= 0) {
    args.nn_max_batch_size = 1;
  }
  if (args.parallel_games <= 0) {
    args.parallel_games = 1;
  }
  if (args.max_game_moves < 0) {
    args.max_game_moves = 0;
  }
  args.pcr_full_search_prob = std::clamp(args.pcr_full_search_prob, 0, 100);
  if (args.temp_early < 0.0f) {
    args.temp_early = args.temp;
  }
  if (args.temp_halflife <= 0.0f) {
    args.temp_halflife = 1.0f;
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    auto to_sec = [](uint64_t ns) -> double {
      return static_cast<double>(ns) / 1e9;
    };

    const CliArgs cli = parse_args(argc, argv);

    SearchParams params;
    params.num_searches = cli.searches;
    params.max_game_moves = cli.max_game_moves;
    params.cpuct = cli.cpuct;
    params.temperature = cli.temp;
    params.temperature_early = cli.temp_early;
    params.temperature_halflife = cli.temp_halflife;
    params.dirichlet_epsilon = cli.dirichlet_epsilon;
    params.dirichlet_alpha = cli.dirichlet_alpha;
    params.parallel_games = cli.parallel_games;
    params.pcr_full_search_prob = cli.pcr_full_search_prob;
    params.use_dynamic_cpuct = cli.use_dynamic_cpuct;
    params.use_fpu = cli.use_fpu;
    params.use_target_pruning = cli.use_target_pruning;
    params.fpu_reduction = cli.fpu_reduction;
    params.c_base = cli.c_base;
    params.target_pruning_threshold = cli.target_pruning_threshold;

    const auto all_start = std::chrono::steady_clock::now();
    const auto infer_init_start = std::chrono::steady_clock::now();
    OnnxInfer infer(
        cli.onnx_path,
        cli.use_cuda,
        cli.cuda_device_id,
        cli.nn_max_batch_size);
    const auto infer_init_end = std::chrono::steady_clock::now();

    const auto selfplay_start = std::chrono::steady_clock::now();
    SelfplayResult result = run_selfplay_games(infer, params, cli.games, cli.threads, cli.seed);
    const auto selfplay_end = std::chrono::steady_clock::now();

    const auto write_memory_start = std::chrono::steady_clock::now();
    write_memory_file(cli.out_path, result.rows);
    const auto write_memory_end = std::chrono::steady_clock::now();

    const auto write_stats_start = std::chrono::steady_clock::now();
    if (!cli.stats_out_path.empty()) {
      write_stats_file(cli.stats_out_path, result.stats);
    }
    const auto write_stats_end = std::chrono::steady_clock::now();
    const auto all_end = std::chrono::steady_clock::now();

    std::cout << "game: " << kGameName << "\n";
    std::cout << "generated rows: " << result.rows.size() << "\n";
    std::cout << "win/draw/lose: " << result.stats.win << "/" << result.stats.draw << "/"
              << result.stats.lose << "\n";
    std::cout << "output: " << cli.out_path << "\n";
    if (!cli.stats_out_path.empty()) {
      std::cout << "stats: " << cli.stats_out_path << "\n";
    }

    const auto infer_prof = infer.profile_snapshot();
    const double infer_init_sec = std::chrono::duration_cast<std::chrono::duration<double>>(
                                      infer_init_end - infer_init_start)
                                      .count();
    const double selfplay_sec = std::chrono::duration_cast<std::chrono::duration<double>>(
                                    selfplay_end - selfplay_start)
                                    .count();
    const double write_memory_sec = std::chrono::duration_cast<std::chrono::duration<double>>(
                                        write_memory_end - write_memory_start)
                                        .count();
    const double write_stats_sec = std::chrono::duration_cast<std::chrono::duration<double>>(
                                       write_stats_end - write_stats_start)
                                       .count();
    const double total_sec =
        std::chrono::duration_cast<std::chrono::duration<double>>(all_end - all_start).count();

    const double mcts_total_sec = to_sec(result.profile.mcts_total_ns);
    const double infer_total_sec = to_sec(result.profile.infer_total_ns);
    const double mcts_other_sec = std::max(0.0, mcts_total_sec - infer_total_sec);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "[profile] cpp total=" << total_sec << "s"
              << " infer_init=" << infer_init_sec << "s"
              << " selfplay=" << selfplay_sec << "s"
              << " write_memory=" << write_memory_sec << "s"
              << " write_stats=" << write_stats_sec << "s\n";
    std::cout << "[profile] search games=" << result.profile.games
              << " mcts_calls=" << result.profile.mcts_calls
              << " infer_calls=" << result.profile.infer_calls
              << " mcts_total=" << mcts_total_sec << "s"
              << " infer_total=" << infer_total_sec << "s"
              << " mcts_other=" << mcts_other_sec << "s"
              << " game_total=" << to_sec(result.profile.game_total_ns) << "s\n";
    std::cout << "[profile] nn query_count=" << infer_prof.query_count
              << " batch_count=" << infer_prof.worker_batch_count
              << " item_count=" << infer_prof.worker_item_count
              << " bind_io=" << to_sec(infer_prof.worker_bind_io_ns) << "s"
              << " input_build=" << to_sec(infer_prof.worker_input_build_ns) << "s"
              << " ort_run=" << to_sec(infer_prof.worker_ort_run_ns) << "s"
              << " ort_output_fetch=" << to_sec(infer_prof.worker_ort_output_fetch_ns) << "s"
              << " softmax=" << to_sec(infer_prof.worker_softmax_ns) << "s"
              << " writeback=" << to_sec(infer_prof.worker_writeback_ns) << "s"
              << " postprocess=" << to_sec(infer_prof.worker_postprocess_ns) << "s\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "cpp_selfplay error: " << e.what() << "\n";
    print_usage();
    return 1;
  }
}
