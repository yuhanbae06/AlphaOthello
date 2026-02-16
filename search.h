#pragma once
#include <torch/script.h>

#include <vector>

#include "config.h"
#include "game.h"
#include "node.h"

template <typename Game>

class MCTS {
 private:
  Config config = Config::load();

  torch::jit::script::Module model;
  torch::Device device;

  std::vector<Node<Game>> tree;

  typedef typename Game::State State;

  inline bool is_fully_expanded(const Node<Game>& node) const {
    /* Return 1 if all children of the node have been expanded */
    return node.children_idxs.size() > 0;
  }

  inline float get_ucb(const Node<Game>& node, int child_idx) const {
    /* Return the UCB value for a child node */
    float q_value;
    if (tree[child_idx].visit_count == 0) {
      q_value = 0.0f;
    } else {
      q_value = -(tree[child_idx].value_sum / tree[child_idx].visit_count);
    }
    return q_value + config.C * tree[child_idx].prior *
                         static_cast<float>(std::sqrt(node.visit_count)) /
                         (1 + tree[child_idx].visit_count);
  }

  inline int select(const Node<Game>& node) const {
    /* Return the index of the best child node to explore */
    float best_ucb = -std::numeric_limits<float>::infinity();
    int best_child_idx = -1;

    for (int child_idx : node.children_idxs) {
      float ucb = get_ucb(node, child_idx);
      if (ucb > best_ucb) {
        best_ucb = ucb;
        best_child_idx = child_idx;
      }
    }

    return best_child_idx;
  }

  inline int expand(const int idx,
                    const std::vector<ActionProb>& action_probs) {
    /* Expand the node at index idx by adding child nodes for each action in
     * action_probs */
    int first_child_idx = static_cast<int>(tree.size());

    for (const auto [action_idx, prob] : action_probs) {
      State child_state = Game::get_next_state(tree[idx].state, action_idx);
      child_state = Game::change_perspective(child_state, child_state.turn);
      tree.emplace_back(child_state, config, idx, prob, tree[idx].depth + 1,
                        action_idx);
      int new_child_idx = static_cast<int>(tree.size()) - 1;
      tree[idx].children_idxs.push_back(new_child_idx);
    }

    return first_child_idx;
  }

  inline void backpropagate(int idx, float value) {
    /* Backpropagate the value up the tree */
    while (idx != -1) {
      tree[idx].visit_count++;
      tree[idx].value_sum += value;
      value = -value;
      idx = tree[idx].parent_idx;
    }
  }

 public:
  MCTS(const Config& config, const torch::jit::script::Module& model,
       const torch::Device& device, int memory_pool_size = 10000)
      : config(config), model(model), device(device) {
    /**/
    tree.reserve(memory_pool_size);

    if (torch::cuda::is_available()) {
      this->device = torch::Device(torch::kCUDA);
    }

    this->model.to(device);
    this->model.eval();
  }

  const std::vector<Node<Game>>& get_tree() const { return tree; }

  int get_best_action() {
    /* Return the action with the highest visit count from the root node */
    int best_action = -1;
    int max_visit_count = -1;

    for (int child_idx : tree[0].children_idxs) {
      if (tree[child_idx].visit_count > max_visit_count) {
        max_visit_count = tree[child_idx].visit_count;
        best_action = tree[child_idx].action_taken;
      }
    }

    return best_action;
  }

  void search(const State& state) {
    torch::NoGradGuard no_grad;

    tree.clear();
    tree.emplace_back(state, config, -1, -1.0f, 0, -1);

    for (int i = 0; i < config.num_searches; i++) {
      int idx = 0;
      float value;

      while (is_fully_expanded(tree[idx]) && !tree[idx].is_terminal) {
        idx = select(tree[idx]);
      }

      if (!tree[idx].is_terminal) {
        auto input_tensor =
            torch::from_blob(Game::get_encoded_state(tree[idx].state).data(),
                             {1, 6, Game::SIZE, Game::SIZE})
                .to(device);

        auto output = model.forward({input_tensor}).toTuple();
        auto policy_tensor =
            torch::softmax(output->elements()[0].toTensor(), 1).cpu();
        auto value_tensor = output->elements()[1].toTensor().cpu();

        float value = value_tensor[0].item<float>();
        std::vector<ActionProb> action_probs;

        auto p_accessor = policy_tensor.accessor<float, 2>();

        auto valid_moves = Game::get_valid_moves(tree[idx].state);

        float sum_prob = 0.0f;

        for (int action : valid_moves) {
          float prob = p_accessor[0][action];
          action_probs.push_back({action, prob});
          sum_prob += prob;
        }

        if (sum_prob > 0.0f) {
          for (auto& ap : action_probs) {
            ap.prob /= sum_prob;
          }
        }

        expand(idx, action_probs);

      } else {
        if (Game::check_win(tree[idx].state, 0)) {
          value = 1.0f;
        } else if (Game::check_win(tree[idx].state, 1)) {
          value = -1.0f;
        } else {
          value = 0.0f;
        }
      }
      backpropagate(idx, value);
    }
  }
};