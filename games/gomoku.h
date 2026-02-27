#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace gomoku {

constexpr int kBoardSize = 10;
constexpr int kActionSize = kBoardSize * kBoardSize;
constexpr int kInputChannels = 3;

struct Bitboard100 {
  uint64_t lo = 0ULL;  // bits [0, 63]
  uint64_t hi = 0ULL;  // bits [64, 99]
};

struct Board {
  Bitboard100 p1{};  // +1 stones
  Bitboard100 p2{};  // -1 stones
};

Board initial_board();
Board canonical_board(const Board& board, int8_t player);
Board flipped_perspective(const Board& board);
int get_valid_moves(const Board& board, int* moves_out);
bool is_full(const Board& board);
bool apply_move(Board& board, int action, int8_t player);
bool check_win(const Board& board, int action, int8_t player);
void encode_state(const Board& board, std::vector<float>& out_encoded);
void to_board_plane(const Board& board, std::vector<int8_t>& out_plane);
int encoded_state_size();
inline int final_state_size() { return kBoardSize * kBoardSize; }
inline void serialize_final_state(const Board& board, std::vector<int8_t>& out_final) {
  to_board_plane(board, out_final);
}

}  // namespace gomoku
