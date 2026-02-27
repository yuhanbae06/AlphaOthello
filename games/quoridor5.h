#pragma once

#include <cstdint>
#include <vector>

namespace quoridor5 {

using bitboard = unsigned __int128;

struct State {
  bitboard p_bits[2];
  bitboard walls_h;
  bitboard walls_v;
  bitboard h_block;
  bitboard v_block;
  int8_t walls_left[2];
  int8_t turn;
  bool is_jumping;
  int8_t jumper_idx;
  int8_t jump_dir;
};

using Board = State;

struct BridgeMask {
  bitboard h_mask;
  bitboard v_mask;
  uint8_t base;
};

constexpr int SIZE = 5;
constexpr int NUM_SQUARES = SIZE * SIZE;
constexpr int WALL_SIZE = SIZE + 1;
constexpr int INNER_WALL = SIZE - 1;
constexpr int WALL_CNT = WALL_SIZE * WALL_SIZE;
constexpr int INNER_WALL_CNT = INNER_WALL * INNER_WALL;
constexpr int ACTION_SIZE = 4 + 2 * INNER_WALL_CNT + 1;
constexpr int ACTION_PASS = ACTION_SIZE - 1;
constexpr int WALLS_LEFT = 3;

constexpr int kBoardSize = SIZE;
constexpr int kActionSize = ACTION_SIZE;
constexpr int kInputChannels = 6;

State get_initial_state();
State apply_action(const State& state, int action_idx);
int get_valid_moves(const State& state, int* moves_out);
bool check_win_by_index(const State& state, int p_idx);
State change_perspective(const State& state, int player);

inline State initial_board() { return get_initial_state(); }

inline State canonical_board(const State& state, int8_t player) {
  if (player == 1) {
    return state;
  }
  return change_perspective(state, 1);
}

inline State flipped_perspective(const State& state) {
  return change_perspective(state, 1);
}

inline bool is_full(const State& state) {
  static thread_local int scratch[ACTION_SIZE];
  return get_valid_moves(state, scratch) == 0;
}

inline bool apply_move(State& state, int action, int8_t player) {
  state.turn = (player == 1) ? 0 : 1;
  state = apply_action(state, action);
  return true;
}

inline bool check_win(const State& state, int action, int8_t player) {
  (void)action;
  const int p_idx = (player == 1) ? 0 : 1;
  return check_win_by_index(state, p_idx);
}

inline bool bit_test(bitboard bits, int idx) {
  return ((bits >> idx) & static_cast<bitboard>(1)) != 0;
}

inline int encoded_state_size() {
  return kInputChannels * kBoardSize * kBoardSize;
}

inline void encode_state(const State& state, std::vector<float>& out_encoded) {
  const int area = kBoardSize * kBoardSize;
  out_encoded.assign(static_cast<size_t>(encoded_state_size()), 0.0f);
  for (int i = 0; i < area; i++) {
    out_encoded[static_cast<size_t>(i)] = bit_test(state.p_bits[0], i) ? 1.0f : 0.0f;
    out_encoded[static_cast<size_t>(area + i)] = bit_test(state.p_bits[1], i) ? 1.0f : 0.0f;
    out_encoded[static_cast<size_t>(2 * area + i)] = bit_test(state.h_block, i) ? 1.0f : 0.0f;
    out_encoded[static_cast<size_t>(3 * area + i)] = bit_test(state.v_block, i) ? 1.0f : 0.0f;
  }
  const float my_walls = static_cast<float>(state.walls_left[0]) / static_cast<float>(WALLS_LEFT);
  const float opp_walls = static_cast<float>(state.walls_left[1]) / static_cast<float>(WALLS_LEFT);
  for (int i = 0; i < area; i++) {
    out_encoded[static_cast<size_t>(4 * area + i)] = my_walls;
    out_encoded[static_cast<size_t>(5 * area + i)] = opp_walls;
  }
}

inline void to_board_plane(const State& state, std::vector<int8_t>& out_plane) {
  const int area = kBoardSize * kBoardSize;
  out_plane.assign(static_cast<size_t>(area), 0);
  for (int i = 0; i < area; i++) {
    if (bit_test(state.p_bits[0], i)) {
      out_plane[static_cast<size_t>(i)] = 1;
    } else if (bit_test(state.p_bits[1], i)) {
      out_plane[static_cast<size_t>(i)] = -1;
    }
  }
}

inline int final_state_size() {
  const int board_area = kBoardSize * kBoardSize;
  const int wall_area = WALL_SIZE * WALL_SIZE;
  return board_area + 2 * wall_area;
}

inline void serialize_final_state(const State& state, std::vector<int8_t>& out_final) {
  const int board_area = kBoardSize * kBoardSize;
  const int wall_area = WALL_SIZE * WALL_SIZE;
  out_final.assign(static_cast<size_t>(final_state_size()), 0);

  for (int i = 0; i < board_area; i++) {
    if (bit_test(state.p_bits[0], i)) {
      out_final[static_cast<size_t>(i)] = 1;
    } else if (bit_test(state.p_bits[1], i)) {
      out_final[static_cast<size_t>(i)] = -1;
    }
  }
  for (int i = 0; i < wall_area; i++) {
    out_final[static_cast<size_t>(board_area + i)] = bit_test(state.walls_h, i) ? 1 : 0;
    out_final[static_cast<size_t>(board_area + wall_area + i)] = bit_test(state.walls_v, i) ? 1 : 0;
  }
}

}  // namespace quoridor5

