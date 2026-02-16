#pragma once
#include <cmath>
#include <vector>

#include "config.h"
#include "game.h"

template <typename Game>

struct Node {
  typename Game::State state;
  Config config = Config::load();

  std::vector<int> children_idxs;
  int parent_idx;

  float prior;
  int visit_count;
  float value_sum;
  int depth;
  int action_taken;
  bool is_end = false;
  bool is_terminal = false;

  Node(const typename Game::State& state, const Config& config,
       int parent_idx = -1, float prior = 0.0f, int depth = 0,
       int action_taken = -1)
      : state(state),
        config(config),
        parent_idx(parent_idx),
        prior(prior),
        visit_count(0),
        value_sum(0.0f),
        depth(depth),
        action_taken(action_taken),
        is_end(false),
        is_terminal(false) {}
};