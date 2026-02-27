#include "search.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

struct AtomicSearchProfile {
  std::atomic<uint64_t> games{0};
  std::atomic<uint64_t> game_total_ns{0};
  std::atomic<uint64_t> mcts_calls{0};
  std::atomic<uint64_t> mcts_total_ns{0};
  std::atomic<uint64_t> infer_calls{0};
  std::atomic<uint64_t> infer_total_ns{0};
};

class ScopedAddNs {
 public:
  explicit ScopedAddNs(std::atomic<uint64_t>* target)
      : target_(target), start_(std::chrono::steady_clock::now()) {}

  ~ScopedAddNs() {
    if (target_ == nullptr) {
      return;
    }
    const auto end = std::chrono::steady_clock::now();
    const auto ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count());
    target_->fetch_add(ns, std::memory_order_relaxed);
  }

 private:
  std::atomic<uint64_t>* target_;
  std::chrono::steady_clock::time_point start_;
};

struct Node {
  game::Board state{};
  float value_sum = 0.0f;
  float terminal_value = 0.0f;
  float prior = 0.0f;
  int parent = -1;
  int first_child = -1;
  int num_children = 0;
  int action_taken = -1;
  int visit_count = 0;
  bool terminal_known = false;
  bool terminal = false;
};

float ucb_score(const Node& child, const SearchParams& params, float sqrt_parent_visits, float parent_q, float dynamic_cpuct) {
  float q;
  if (params.use_fpu) {
    q = (child.visit_count == 0) ? (parent_q - params.fpu_reduction)
                                             : -(child.value_sum / static_cast<float>(child.visit_count));
  } else {
    q = (child.visit_count == 0) ? 0.0f : -(child.value_sum / static_cast<float>(child.visit_count));
  }

  const float u =
      dynamic_cpuct * child.prior *
      (sqrt_parent_visits / static_cast<float>(child.visit_count + 1));
  return q + u;
}

int select_child(int node_idx, const std::vector<Node>& tree, const SearchParams& params) {
  int best_idx = -1;
  float best_score = -1e30f;
  const Node& parent = tree[static_cast<size_t>(node_idx)];
  const int start = parent.first_child;
  const int end = start + parent.num_children;
  const float sqrt_parent_visits =
      std::sqrt(static_cast<float>(std::max(parent.visit_count, 1)));

  const float dynamic_cpuct = params.use_dynamic_cpuct ? params.cpuct + std::log((parent.visit_count + params.c_base + 1.0f) / params.c_base) : params.cpuct;
  const float parent_q = (parent.visit_count > 0) ? (parent.value_sum / static_cast<float>(parent.visit_count)) : 0.0f;

  for (int i = start; i < end; i++) {
    const float score = ucb_score(tree[static_cast<size_t>(i)], params, sqrt_parent_visits, parent_q, dynamic_cpuct);
    if (score > best_score) {
      best_score = score;
      best_idx = i;
    }
  }
  return best_idx;
}

bool evaluate_terminal(Node& node) {
  if (node.terminal_known) {
    return node.terminal;
  }
  node.terminal_known = true;
  if (node.action_taken >= 0 &&
      game::check_win(node.state, node.action_taken, -1)) {
    node.terminal = true;
    node.terminal_value = -1.0f;
    return true;
  }
  if (game::is_full(node.state)) {
    node.terminal = true;
    node.terminal_value = 0.0f;
    return true;
  }
  node.terminal = false;
  node.terminal_value = 0.0f;
  return false;
}

void backpropagate(int node_idx, std::vector<Node>& tree, float value) {
  while (node_idx != -1) {
    Node& node = tree[static_cast<size_t>(node_idx)];
    node.visit_count += 1;
    node.value_sum += value;
    value = -value;
    node_idx = node.parent;
  }
}

void masked_normalized_policy(
    const std::vector<float>& policy,
    int action_size,
    const int* valid_moves,
    int valid_count,
    std::vector<float>& out) {
  out.assign(static_cast<size_t>(action_size), 0.0f);

  if (valid_count <= 0) {
    return;
  }

  double sum = 0.0;
  for (int i = 0; i < valid_count; i++) {
    const int a = valid_moves[i];
    const float p = std::max(0.0f, policy[static_cast<size_t>(a)]);
    out[static_cast<size_t>(a)] = p;
    sum += p;
  }

  if (sum <= 1e-12) {
    const float uniform = 1.0f / static_cast<float>(valid_count);
    for (int i = 0; i < valid_count; i++) {
      const int a = valid_moves[i];
      out[static_cast<size_t>(a)] = uniform;
    }
  }

  const float inv_sum = 1.0f / static_cast<float>(sum);
  for (int i = 0; i < valid_count; i++) {
    const int a = valid_moves[i];
    out[static_cast<size_t>(a)] = static_cast<float>(out[static_cast<size_t>(a)] * inv_sum);
  }
}

void apply_dirichlet_noise(
    std::vector<float>& policy,
    const int* valid_moves,
    int valid_count,
    float epsilon,
    float alpha,
    std::mt19937& rng) {
  if (epsilon <= 0.0f || valid_count <= 0) {
    return;
  }  
  std::gamma_distribution<float> gamma(alpha, 1.0f);
  std::vector<float> noise(static_cast<size_t>(valid_count), 0.0f);
  float sum = 0.0f;
  for (int i = 0; i < valid_count; i++) {
    noise[i] = gamma(rng);
    sum += noise[i];
  }
  if (sum <= 1e-12f) {
    return;
  }

  const float inv_sum = 1.0f / sum;

  for (int i = 0; i < valid_count; i++) {
    noise[i] *= inv_sum;
  }

  for (int i = 0; i < valid_count; i++) {
    const int a = valid_moves[i];
    policy[static_cast<size_t>(a)] =
        (1.0f - epsilon) * policy[static_cast<size_t>(a)] + epsilon * noise[i];
  }
}

void apply_shaped_dirichlet_noise(
    std::vector<float>& policy,
    const int* valid_moves,
    int valid_count,
    float epsilon,
    float alpha,
    std::mt19937& rng) {
  float const total_alpha = alpha * static_cast<float>(valid_count);

  std::vector<float> noise(static_cast<size_t>(valid_count), 0.0f);
  float sum = 0.0f;

  for (int i = 0; i < valid_count; ++i) {
    const int a = valid_moves[i];
    float p = std::max(0.0f, policy[static_cast<size_t>(a)]);

    float dynamic_alpha = total_alpha * p + 1e-6f;

    std::gamma_distribution<float> gamma(dynamic_alpha, 1.0f);
    noise[i] = gamma(rng);
    sum += noise[i];
  }
  if (sum <= 1e-12f) {
    return;
  }

  const float inv_sum = 1.0f / sum;

  for (int i = 0; i < valid_count; i++) {
    noise[i] *= inv_sum;
  }

  for (int i = 0; i < valid_count; ++i) {
    const int a = valid_moves[i];
    policy[static_cast<size_t>(a)] = (1.0f - epsilon) * policy[static_cast<size_t>(a)] + epsilon * noise[i];
  }
}

int collect_valid_moves(const game::Board& state, std::vector<int>& scratch) {
  if (scratch.size() < static_cast<size_t>(game::kActionSize)) {
    scratch.resize(static_cast<size_t>(game::kActionSize));
  }
  return game::get_valid_moves(state, scratch.data());
}

void target_policy_pruning(std::vector<float>& policy, const int* valid_moves, int valid_count, float threshold_ratio) {
  if (threshold_ratio <= 0.0f || valid_count <= 0) return;

  float max_p = 0.0f;
  for (int i = 0; i < valid_count; i++) {
    max_p = std::max(max_p, policy[static_cast<size_t>(valid_moves[i])]);
  }

  if (max_p <= 1e-12f) return;

  const float relative_limit = max_p * threshold_ratio;

  float sum = 0.0f;
  int best_a = valid_moves[0];
  float actual_max = -1.0f;

  for (int i = 0; i < valid_count; i++) {
    const int a = valid_moves[i];
    float& p = policy[static_cast<size_t>(a)];

    if (p >= relative_limit) {
      sum += p;
    } else {
      p = 0.0f;
    }

    if (p > actual_max) {
      actual_max = p;
      best_a = a;
    }
  }

  if (sum > 1e-12f) {
    const float inv_sum = 1.0f / sum;
    for (int i = 0; i < valid_count; i++) {
      policy[static_cast<size_t>(valid_moves[i])] *= inv_sum;
    }
  } else {
    policy[static_cast<size_t>(best_a)] = 1.0f;
  }
}

void expand_batch(
    int node_idx,
    std::vector<Node>& tree,
    const std::vector<float>& policy) {
  thread_local std::vector<int> valid;
  const int valid_count = collect_valid_moves(tree[static_cast<size_t>(node_idx)].state, valid);

  tree[static_cast<size_t>(node_idx)].first_child = static_cast<int>(tree.size());
  int child_count = 0;
  for (int i = 0; i < valid_count; i++) {
    const int action = valid[i];
    if (policy[static_cast<size_t>(action)] <= 0.0f) {
      continue;
    }

    game::Board child_state = tree[static_cast<size_t>(node_idx)].state;
    game::apply_move(child_state, action, 1);
    child_state = game::flipped_perspective(child_state);

    Node child;
    child.state = std::move(child_state);
    child.parent = node_idx;
    child.action_taken = action;
    child.prior = policy[static_cast<size_t>(action)];
    tree.push_back(std::move(child));
    child_count++;
  }
  tree[static_cast<size_t>(node_idx)].num_children = child_count;
}

std::vector<std::pair<std::vector<float>, float>> infer_batch_with_profile(
    OnnxInfer& infer,
    const std::vector<game::Board>& states,
    AtomicSearchProfile* profile) {
  if (states.empty()) {
    return {};
  }
  const auto infer_start = std::chrono::steady_clock::now();
  auto results = infer.infer_batch(states);
  if (profile != nullptr) {
    const auto infer_end = std::chrono::steady_clock::now();
    const auto ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(infer_end - infer_start).count());
    profile->infer_total_ns.fetch_add(ns, std::memory_order_relaxed);
    profile->infer_calls.fetch_add(static_cast<uint64_t>(states.size()), std::memory_order_relaxed);
  }
  return results;
}

void run_mcts_batch(
    OnnxInfer& infer,
    const std::vector<game::Board>& canonical_roots,
    const SearchParams& params,
    std::mt19937& rng,
    AtomicSearchProfile* profile,
    std::vector<float>* average_leaf_depths,
    std::vector<float>* max_leaf_depths,
    std::vector<Node>& tree,
    std::vector<std::vector<float>>& all_action_probs,
    const std::vector<uint8_t>& is_full_search,
    const std::vector<int>& target_searches) {
  
  const size_t num_roots = canonical_roots.size();

  if (all_action_probs.size() < num_roots) {
    all_action_probs.resize(num_roots);
  }

  for (size_t i = 0; i < num_roots; ++i) {
    all_action_probs[i].assign(game::kActionSize, 0.0f);
  }

  if (num_roots == 0) {
    return;
  }

  ScopedAddNs mcts_timer(profile ? &profile->mcts_total_ns : nullptr);
  if (profile != nullptr) {
    profile->mcts_calls.fetch_add(static_cast<uint64_t>(num_roots), std::memory_order_relaxed);
  }

  tree.clear();
  thread_local std::vector<int> root_indices;
  thread_local std::vector<std::vector<int>> root_valid_moves;
  thread_local std::vector<int> valid_scratch;
  thread_local std::vector<int> depth_sum;
  thread_local std::vector<int> depth_max;
  thread_local std::vector<int> depth_count;
  thread_local std::vector<int> curr_leaves;
  thread_local std::vector<game::Board> states_to_infer;
  thread_local std::vector<int> game_indices_to_infer;
  
  root_indices.resize(num_roots); 
  depth_sum.assign(num_roots, 0);
  depth_max.assign(num_roots, 0);
  depth_count.assign(num_roots, 0);
  curr_leaves.assign(num_roots, -1);
  root_valid_moves.resize(num_roots);

  if (valid_scratch.size() < static_cast<size_t>(game::kActionSize)) {
    valid_scratch.resize(static_cast<size_t>(game::kActionSize), 0);
  }

  states_to_infer.clear();
  game_indices_to_infer.clear();

  for (size_t i = 0; i < num_roots; i++) {
    Node root;
    root.state = canonical_roots[i];
    root.visit_count = 1;
    root_indices[i] = static_cast<int>(tree.size());
    tree.push_back(std::move(root));
  }

  const auto root_evals = infer_batch_with_profile(infer, canonical_roots, profile);

  thread_local std::vector<float> root_policy;
  thread_local std::vector<float> root_policy_buffer;

  for (size_t i = 0; i < num_roots; i++) {
    const int valid_count =
        collect_valid_moves(tree[static_cast<size_t>(root_indices[i])].state, valid_scratch);
    root_valid_moves[i].assign(valid_scratch.begin(), valid_scratch.begin() + valid_count);
    masked_normalized_policy(
        root_evals[i].first, game::kActionSize, root_valid_moves[i].data(), valid_count, root_policy_buffer);
    if (is_full_search[i]) {
      if (params.use_shaped_dirichlet) {
        apply_shaped_dirichlet_noise(
          root_policy_buffer,
          root_valid_moves[i].data(),
          valid_count,
          params.dirichlet_epsilon,
          params.dirichlet_alpha,
          rng);
      } else {
        apply_dirichlet_noise(
          root_policy_buffer,
          root_valid_moves[i].data(),
          valid_count,
          params.dirichlet_epsilon,
          params.dirichlet_alpha,
          rng);
      }
    }
    masked_normalized_policy(
        root_policy_buffer, game::kActionSize, root_valid_moves[i].data(), valid_count, root_policy);
    expand_batch(root_indices[i], tree, root_policy);
  }

  int max_searches = 0;
  for (int ts : target_searches) {
    max_searches = std::max(max_searches, ts);
  }
  
  thread_local std::vector<float> leaf_policy;

  for (int s = 0; s < max_searches; s++) {
    states_to_infer.clear();
    game_indices_to_infer.clear();
    for (size_t i = 0; i < num_roots; i++) {
      if (s >= target_searches[i]) {
        continue;
      }
      int curr = root_indices[i];
      int depth = 0;
      while (tree[static_cast<size_t>(curr)].num_children > 0) {
        curr = select_child(curr, tree, params);
        if (curr < 0) {
          break;
        }
        depth++;
      }
      if (curr < 0) {
        continue;
      }

      curr_leaves[i] = curr;
      depth_sum[i] += depth;
      depth_max[i] = std::max(depth_max[i], depth);
      depth_count[i] += 1;

      if (!evaluate_terminal(tree[static_cast<size_t>(curr)])) {
        states_to_infer.push_back(tree[static_cast<size_t>(curr)].state);
        game_indices_to_infer.push_back(static_cast<int>(i));
      } else {
        backpropagate(curr, tree, tree[static_cast<size_t>(curr)].terminal_value);
      }
    }

    if (!states_to_infer.empty()) {
      auto evals = infer_batch_with_profile(infer, states_to_infer, profile);
      for (size_t j = 0; j < evals.size(); j++) {
        const int game_idx = game_indices_to_infer[j];
        const int leaf_idx = curr_leaves[static_cast<size_t>(game_idx)];
        const int valid_count =
            collect_valid_moves(tree[static_cast<size_t>(leaf_idx)].state, valid_scratch);
            masked_normalized_policy(
                evals[j].first, game::kActionSize, valid_scratch.data(), valid_count, leaf_policy);
        expand_batch(leaf_idx, tree, leaf_policy);
        backpropagate(leaf_idx, tree, evals[j].second);
      }
    }
  }

  for (size_t i = 0; i < num_roots; i++) {
    int root_idx = root_indices[i];
    float sum_visits = 0.0f;
    const int start = tree[static_cast<size_t>(root_idx)].first_child;
    const int end = start + tree[static_cast<size_t>(root_idx)].num_children;
    for (int j = start; j < end; j++) {
      const int a = tree[static_cast<size_t>(j)].action_taken;
      all_action_probs[i][static_cast<size_t>(a)] =
          static_cast<float>(tree[static_cast<size_t>(j)].visit_count);
      sum_visits += all_action_probs[i][static_cast<size_t>(a)];
    }

    if (sum_visits <= 1e-12f) {
      if (!root_valid_moves[i].empty()) {
        const float uniform = 1.0f / static_cast<float>(root_valid_moves[i].size());
        for (int a : root_valid_moves[i]) {
          all_action_probs[i][static_cast<size_t>(a)] = uniform;
        }
      }
      continue;
    }

    float inv_sum_visits = 1.0f / sum_visits;

    for (int a = 0; a < game::kActionSize; a++) {
      all_action_probs[i][static_cast<size_t>(a)] *= inv_sum_visits;
    }
  }

  if (average_leaf_depths != nullptr) {
    average_leaf_depths->assign(num_roots, 0.0f);
    for (size_t i = 0; i < num_roots; i++) {
      if (depth_count[i] > 0) {
        (*average_leaf_depths)[i] =
            static_cast<float>(depth_sum[i]) / static_cast<float>(depth_count[i]);
      }
    }
  }
  if (max_leaf_depths != nullptr) {
    max_leaf_depths->assign(num_roots, 0.0f);
    for (size_t i = 0; i < num_roots; i++) {
      (*max_leaf_depths)[i] = static_cast<float>(depth_max[i]);
    }
  }
  if (params.use_target_pruning) {
    for (int i = 0; i < num_roots; ++i) {
      if (is_full_search[i]) {
        target_policy_pruning(all_action_probs[i], root_valid_moves[i].data(),
                              static_cast<int>(root_valid_moves[i].size()), params.target_pruning_threshold);
      }
    }
  }
}


int sample_action(
    const std::vector<float>& probs,
    const int* valid_moves,
    int valid_count,
    float temperature,
    std::mt19937& rng) {
  if (valid_count <= 0) {
    return 0;
  }
  if (temperature <= 1e-6f) {
    int best_action = valid_moves[0];
    float best_prob = probs[static_cast<size_t>(best_action)];
    for (int i = 0; i < valid_count; i++) {
      const int a = valid_moves[i];
      if (probs[static_cast<size_t>(a)] > best_prob) {
        best_prob = probs[static_cast<size_t>(a)];
        best_action = a;
      }
    }
    return best_action;
  }

  std::vector<double> weights(static_cast<size_t>(valid_count), 0.0);
  double sum = 0.0;
  const double inv_temp = 1.0 / static_cast<double>(temperature);
  for (int i = 0; i < valid_count; i++) {
    const double w = std::pow(
        std::max(0.0f, probs[static_cast<size_t>(valid_moves[i])]), inv_temp);
    weights[static_cast<size_t>(i)] = w;
    sum += w;
  }
  if (sum <= 1e-12) {
    std::uniform_int_distribution<int> dist(0, valid_count - 1);
    return valid_moves[static_cast<size_t>(dist(rng))];
  }

  const float inv_sum = 1.0f / sum;

  for (double& w : weights) {
    w *= inv_sum;
  }
  std::discrete_distribution<int> dist(weights.begin(), weights.end());
  return valid_moves[static_cast<size_t>(dist(rng))];
}

float scheduled_temperature(
    const SearchParams& params,
    int move_number) {
  const double base = static_cast<double>(params.temperature);
  const double early = static_cast<double>(params.temperature_early);
  const double halflife = static_cast<double>(params.temperature_halflife);
  if (halflife <= 1e-9) {
    return static_cast<float>(std::max(0.0, base));
  }
  const int clamped_move = std::max(0, move_number);
  const double halflives = static_cast<double>(clamped_move) / halflife;
  const double decayed = early * std::pow(0.5, halflives);
  const double scheduled = std::max(base, decayed);
  return static_cast<float>(std::max(0.0, scheduled));
}

struct HistStep {
  game::Board canonical{};
  std::vector<float> policy;
  int8_t player = 1;
  bool is_full_search = true;
};

struct GameResult {
  std::vector<TrainingRow> rows;
  int winner = 0;
  std::vector<int8_t> final_state;
  std::vector<float> average_depth;
  std::vector<float> max_depth;
};

struct ActiveGame {
  game::Board board{};
  int8_t player = 1;
  std::vector<HistStep> hist;
  std::vector<float> average_depth;
  std::vector<float> max_depth;
  std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
};

GameResult finalize_game(ActiveGame&& active_game, int winner) {
  std::vector<TrainingRow> rows;
  rows.reserve(active_game.hist.size());
  for (const HistStep& h : active_game.hist) {
    if (!h.is_full_search) {
      continue;
    }
    TrainingRow row;
    game::encode_state(h.canonical, row.encoded_state);
    row.policy = h.policy;
    if (winner == 0) {
      row.value = 0.0f;
    } else {
      row.value = (winner == h.player) ? 1.0f : -1.0f;
    }
    rows.push_back(std::move(row));
  }

  GameResult result;
  result.rows = std::move(rows);
  result.winner = winner;
  game::serialize_final_state(active_game.board, result.final_state);
  result.average_depth = std::move(active_game.average_depth);
  result.max_depth = std::move(active_game.max_depth);
  return result;
}

}  // namespace

SelfplayResult run_selfplay_games(
    OnnxInfer& infer,
    const SearchParams& params,
    int num_games,
    int num_threads,
    uint64_t seed) {
  SelfplayResult result;
  if (num_games <= 0) {
    return result;
  }
  if (num_threads <= 0) {
    num_threads = 1;
  }

  const int parallel_games_per_worker = std::max(1, params.parallel_games);
  const int num_workers = std::max(1, std::min(num_threads, num_games));

  std::atomic<int> next_game(0);
  AtomicSearchProfile atomic_profile;
  std::mutex result_mutex;

  result.rows.reserve(static_cast<size_t>(num_games * 40));
  result.stats.average_depth_lists.reserve(static_cast<size_t>(num_games));
  result.stats.max_depth_lists.reserve(static_cast<size_t>(num_games));
  result.stats.final_states.reserve(static_cast<size_t>(num_games));

  std::vector<std::thread> workers;
  workers.reserve(num_workers);
  for (int t = 0; t < num_workers; t++) {
    workers.emplace_back([&, t]() {
      std::mt19937 rng(static_cast<uint32_t>(seed + 10007ULL * static_cast<uint64_t>(t)));
      std::vector<Node> thread_tree_pool;
      thread_tree_pool.reserve(
          static_cast<size_t>(params.num_searches * parallel_games_per_worker * 20));
      std::vector<std::vector<float>> all_action_probs;
      std::vector<ActiveGame> active_games;
      active_games.reserve(static_cast<size_t>(parallel_games_per_worker));
      std::vector<game::Board> canonical_states;
      std::vector<int> target_searches;
      std::vector<uint8_t> is_full_search;
      std::vector<float> average_depths;
      std::vector<float> max_depths;

      const size_t max_p_games = static_cast<size_t>(parallel_games_per_worker);
      canonical_states.reserve(max_p_games);
      target_searches.reserve(max_p_games);
      is_full_search.reserve(max_p_games);

      auto refill_active_games = [&]() {
        while (static_cast<int>(active_games.size()) < parallel_games_per_worker) {
          const int game_idx = next_game.fetch_add(1);
          if (game_idx >= num_games) {
            break;
          }
          ActiveGame g;
          g.board = game::initial_board();
          g.player = 1;
          g.hist.reserve(static_cast<size_t>(game::kBoardSize * game::kBoardSize));
          g.average_depth.reserve(static_cast<size_t>(game::kBoardSize * game::kBoardSize));
          g.max_depth.reserve(static_cast<size_t>(game::kBoardSize * game::kBoardSize));
          g.start_time = std::chrono::steady_clock::now();
          active_games.push_back(std::move(g));
        }
      };

      refill_active_games();
      while (true) {
        if (active_games.empty()) {
          break;
        }

        canonical_states.resize(active_games.size());
        target_searches.resize(active_games.size());
        is_full_search.resize(active_games.size());
        const int pcr_full_prob = std::clamp(params.pcr_full_search_prob, 0, 100);
        std::uniform_int_distribution<int> pcr_dist(0, 99);
        for (size_t i = 0; i < active_games.size(); i++) {
          canonical_states[i] =
              game::canonical_board(active_games[i].board, active_games[i].player);
          const bool full_search =
              (pcr_full_prob >= 100) ||
              ((pcr_full_prob > 0) && (pcr_dist(rng) < pcr_full_prob));
          is_full_search[i] = full_search;
          const int reduced = std::max(1, params.num_searches / 4);
          target_searches[i] = full_search ? params.num_searches : reduced;
        }

        run_mcts_batch(
          infer,
          canonical_states,
          params,
          rng,
          &atomic_profile,
          &average_depths,
          &max_depths,
          thread_tree_pool,
          all_action_probs,
          is_full_search,
          target_searches);

        for (int i = static_cast<int>(active_games.size()) - 1; i >= 0; i--) {
          ActiveGame& game_inst = active_games[static_cast<size_t>(i)];
          game_inst.average_depth.push_back(average_depths[static_cast<size_t>(i)]);
          game_inst.max_depth.push_back(max_depths[static_cast<size_t>(i)]);

          HistStep step;
          step.canonical = canonical_states[static_cast<size_t>(i)];
          step.policy = all_action_probs[static_cast<size_t>(i)];
          step.player = game_inst.player;
          step.is_full_search = is_full_search[static_cast<size_t>(i)];
          game_inst.hist.push_back(std::move(step));

          thread_local std::vector<int> valid;
          const int valid_count = collect_valid_moves(game_inst.board, valid);
          const float move_temp =
              scheduled_temperature(params, static_cast<int>(game_inst.hist.size()) - 1);
          const int action =
              sample_action(
                  all_action_probs[static_cast<size_t>(i)],
                  valid.data(),
                  valid_count,
                  move_temp,
                  rng);

          game::apply_move(game_inst.board, action, game_inst.player);
          const bool win = game::check_win(game_inst.board, action, game_inst.player);
          const bool full = game::is_full(game_inst.board);
          const bool max_moves_reached =
              (params.max_game_moves > 0) &&
              (static_cast<int>(game_inst.hist.size()) >= params.max_game_moves);
          if (win || full || max_moves_reached) {
            const auto game_end_time = std::chrono::steady_clock::now();
            const auto game_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    game_end_time - game_inst.start_time)
                    .count());
            atomic_profile.game_total_ns.fetch_add(game_ns, std::memory_order_relaxed);
            atomic_profile.games.fetch_add(1, std::memory_order_relaxed);

            const int winner = win ? game_inst.player : 0;
            GameResult game_result = finalize_game(std::move(game_inst), winner);
            {
              std::lock_guard<std::mutex> lock(result_mutex);
              result.rows.insert(
                  result.rows.end(),
                  std::make_move_iterator(game_result.rows.begin()),
                  std::make_move_iterator(game_result.rows.end()));
              result.stats.average_depth_lists.push_back(std::move(game_result.average_depth));
              result.stats.max_depth_lists.push_back(std::move(game_result.max_depth));
              result.stats.final_states.push_back(std::move(game_result.final_state));
              if (game_result.winner == 1) {
                result.stats.win += 1;
              } else if (game_result.winner == -1) {
                result.stats.lose += 1;
              } else {
                result.stats.draw += 1;
              }
            }

            if (static_cast<size_t>(i) + 1 != active_games.size()) {
              active_games[static_cast<size_t>(i)] = std::move(active_games.back());
            }
            active_games.pop_back();
          } else {
            game_inst.player = static_cast<int8_t>(-game_inst.player);
          }
        }
        refill_active_games();
      }
    });
  }

  for (auto& w : workers) {
    w.join();
  }

  result.profile.games = atomic_profile.games.load(std::memory_order_relaxed);
  result.profile.game_total_ns = atomic_profile.game_total_ns.load(std::memory_order_relaxed);
  result.profile.mcts_calls = atomic_profile.mcts_calls.load(std::memory_order_relaxed);
  result.profile.mcts_total_ns = atomic_profile.mcts_total_ns.load(std::memory_order_relaxed);
  result.profile.infer_calls = atomic_profile.infer_calls.load(std::memory_order_relaxed);
  result.profile.infer_total_ns = atomic_profile.infer_total_ns.load(std::memory_order_relaxed);
  return result;
}

void write_memory_file(
    const std::string& path,
    const std::vector<TrainingRow>& rows) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open output file: " + path);
  }

  const uint32_t count = static_cast<uint32_t>(rows.size());
  out.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));
  if (rows.empty()) {
    return;
  }

  const size_t state_size = static_cast<size_t>(game::encoded_state_size());
  const size_t policy_size = static_cast<size_t>(game::kActionSize);
  for (const TrainingRow& r : rows) {
    if (r.encoded_state.size() != state_size || r.policy.size() != policy_size) {
      throw std::runtime_error("training row has unexpected state/policy size");
    }
    out.write(
        reinterpret_cast<const char*>(r.encoded_state.data()),
        static_cast<std::streamsize>(state_size * sizeof(float)));
    out.write(
        reinterpret_cast<const char*>(r.policy.data()),
        static_cast<std::streamsize>(policy_size * sizeof(float)));
    out.write(reinterpret_cast<const char*>(&r.value), sizeof(float));
  }
}

void write_stats_file(
    const std::string& path,
    const SelfplayStats& stats) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open stats file: " + path);
  }

  out.write(reinterpret_cast<const char*>(&stats.win), sizeof(uint32_t));
  out.write(reinterpret_cast<const char*>(&stats.draw), sizeof(uint32_t));
  out.write(reinterpret_cast<const char*>(&stats.lose), sizeof(uint32_t));

  auto write_depth_lists = [&](const std::vector<std::vector<float>>& lists) {
    const uint32_t list_count = static_cast<uint32_t>(lists.size());
    out.write(reinterpret_cast<const char*>(&list_count), sizeof(uint32_t));
    for (const auto& depth_list : lists) {
      const uint32_t len = static_cast<uint32_t>(depth_list.size());
      out.write(reinterpret_cast<const char*>(&len), sizeof(uint32_t));
      if (!depth_list.empty()) {
        out.write(
            reinterpret_cast<const char*>(depth_list.data()),
            static_cast<std::streamsize>(depth_list.size() * sizeof(float)));
      }
    }
  };

  write_depth_lists(stats.average_depth_lists);
  write_depth_lists(stats.max_depth_lists);

  const uint32_t final_states_count = static_cast<uint32_t>(stats.final_states.size());
  out.write(reinterpret_cast<const char*>(&final_states_count), sizeof(uint32_t));
  const size_t final_state_bytes = static_cast<size_t>(game::final_state_size());
  for (const auto& board : stats.final_states) {
    if (board.size() == final_state_bytes) {
      out.write(
          reinterpret_cast<const char*>(board.data()),
          static_cast<std::streamsize>(final_state_bytes * sizeof(int8_t)));
      continue;
    }
    std::vector<int8_t> padded(final_state_bytes, 0);
    const size_t copy_n = std::min(final_state_bytes, board.size());
    std::copy(board.begin(), board.begin() + static_cast<std::ptrdiff_t>(copy_n), padded.begin());
    out.write(
        reinterpret_cast<const char*>(padded.data()),
        static_cast<std::streamsize>(final_state_bytes * sizeof(int8_t)));
  }
}
