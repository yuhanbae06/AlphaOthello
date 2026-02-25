#include "gomoku.h"

namespace gomoku {

namespace {

inline int row_of(int action) { return action / kBoardSize; }
inline int col_of(int action) { return action % kBoardSize; }

int count_dir(const Board& board, int r, int c, int dr, int dc, int8_t player) {
  int count = 1;

  int nr = r + dr;
  int nc = c + dc;
  while (nr >= 0 && nr < kBoardSize && nc >= 0 && nc < kBoardSize &&
         board[static_cast<size_t>(nr * kBoardSize + nc)] == player) {
    count++;
    nr += dr;
    nc += dc;
  }

  nr = r - dr;
  nc = c - dc;
  while (nr >= 0 && nr < kBoardSize && nc >= 0 && nc < kBoardSize &&
         board[static_cast<size_t>(nr * kBoardSize + nc)] == player) {
    count++;
    nr -= dr;
    nc -= dc;
  }

  return count;
}

}  // namespace

Board initial_board() {
  Board b{};
  b.fill(0);
  return b;
}

Board canonical_board(const Board& board, int8_t player) {
  Board out{};
  for (int i = 0; i < kActionSize; i++) {
    out[static_cast<size_t>(i)] = static_cast<int8_t>(board[static_cast<size_t>(i)] * player);
  }
  return out;
}

Board flipped_perspective(const Board& board) {
  Board out{};
  for (int i = 0; i < kActionSize; i++) {
    out[static_cast<size_t>(i)] = static_cast<int8_t>(-board[static_cast<size_t>(i)]);
  }
  return out;
}

int get_valid_moves(const Board& board, int* moves_out) {
  int count = 0;
  for (int a = 0; a < kActionSize; a++) {
    if (board[static_cast<size_t>(a)] == 0) {
      moves_out[count++] = a;
    }
  }
  return count;
}

bool is_full(const Board& board) {
  static thread_local int scratch[kActionSize];
  return get_valid_moves(board, scratch) == 0;
}

bool apply_move(Board& board, int action, int8_t player) {
  if (action < 0 || action >= kActionSize) {
    return false;
  }
  if (board[static_cast<size_t>(action)] != 0) {
    return false;
  }
  board[static_cast<size_t>(action)] = player;
  return true;
}

bool check_win(const Board& board, int action, int8_t player) {
  if (action < 0 || action >= kActionSize) {
    return false;
  }
  if (board[static_cast<size_t>(action)] != player) {
    return false;
  }
  const int r = row_of(action);
  const int c = col_of(action);
  return count_dir(board, r, c, 1, 0, player) >= 5 ||
         count_dir(board, r, c, 0, 1, player) >= 5 ||
         count_dir(board, r, c, 1, 1, player) >= 5 ||
         count_dir(board, r, c, 1, -1, player) >= 5;
}

void encode_state(const Board& board, std::vector<float>& out_encoded) {
  out_encoded.assign(static_cast<size_t>(encoded_state_size()), 0.0f);
  for (int i = 0; i < kActionSize; i++) {
    const int8_t v = board[static_cast<size_t>(i)];
    out_encoded[static_cast<size_t>(i)] = (v == -1) ? 1.0f : 0.0f;
    out_encoded[static_cast<size_t>(kActionSize + i)] = (v == 0) ? 1.0f : 0.0f;
    out_encoded[static_cast<size_t>(2 * kActionSize + i)] = (v == 1) ? 1.0f : 0.0f;
  }
}

void to_board_plane(const Board& board, std::vector<int8_t>& out_plane) {
  out_plane.assign(board.begin(), board.end());
}

int encoded_state_size() { return kInputChannels * kActionSize; }

}  // namespace gomoku
