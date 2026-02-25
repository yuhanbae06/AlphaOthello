#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace gomoku {

constexpr int kBoardSize = 10;
constexpr int kActionSize = kBoardSize * kBoardSize;
constexpr int kInputChannels = 3;
using Board = std::array<int8_t, kActionSize>;

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

}  // namespace gomoku
