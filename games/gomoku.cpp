#include "gomoku.h"

#include <array>
#include <cstdint>

namespace gomoku {

namespace {

constexpr int kHighBits = kActionSize - 64;
constexpr uint64_t kHighMask = (1ULL << kHighBits) - 1ULL;

inline bool is_valid_action(int action) {
  return action >= 0 && action < kActionSize;
}

inline bool bit_test(const Bitboard100& bb, int idx) {
  if (idx < 64) {
    return ((bb.lo >> idx) & 1ULL) != 0ULL;
  }
  return ((bb.hi >> (idx - 64)) & 1ULL) != 0ULL;
}

inline void bit_set(Bitboard100& bb, int idx) {
  if (idx < 64) {
    bb.lo |= (1ULL << idx);
    return;
  }
  bb.hi |= (1ULL << (idx - 64));
}

inline bool bit_contains(const Bitboard100& bits, const Bitboard100& mask) {
  return ((bits.lo & mask.lo) == mask.lo) && ((bits.hi & mask.hi) == mask.hi);
}

inline Bitboard100 empty_bits(const Board& board) {
  Bitboard100 out{};
  out.lo = ~(board.p1.lo | board.p2.lo);
  out.hi = (~(board.p1.hi | board.p2.hi)) & kHighMask;
  return out;
}

struct WinMaskTable {
  static constexpr int kMaxMasksPerAction = 20;
  std::array<std::array<Bitboard100, kMaxMasksPerAction>, kActionSize> masks{};
  std::array<uint8_t, kActionSize> count{};

  WinMaskTable() {
    auto add_segment = [&](int r0, int c0, int dr, int dc) {
      Bitboard100 segment{};
      int actions[5];
      for (int i = 0; i < 5; i++) {
        const int r = r0 + dr * i;
        const int c = c0 + dc * i;
        const int action = r * kBoardSize + c;
        actions[i] = action;
        bit_set(segment, action);
      }
      for (int i = 0; i < 5; i++) {
        const int action = actions[i];
        const uint8_t idx = count[static_cast<size_t>(action)]++;
        masks[static_cast<size_t>(action)][idx] = segment;
      }
    };

    for (int r = 0; r < kBoardSize; r++) {
      for (int c = 0; c <= kBoardSize - 5; c++) {
        add_segment(r, c, 0, 1);
      }
    }
    for (int c = 0; c < kBoardSize; c++) {
      for (int r = 0; r <= kBoardSize - 5; r++) {
        add_segment(r, c, 1, 0);
      }
    }
    for (int r = 0; r <= kBoardSize - 5; r++) {
      for (int c = 0; c <= kBoardSize - 5; c++) {
        add_segment(r, c, 1, 1);
      }
    }
    for (int r = 0; r <= kBoardSize - 5; r++) {
      for (int c = 4; c < kBoardSize; c++) {
        add_segment(r, c, 1, -1);
      }
    }
  }
};

const WinMaskTable& win_mask_table() {
  static const WinMaskTable table;
  return table;
}

inline const Bitboard100& player_bits(const Board& board, int8_t player) {
  return (player == 1) ? board.p1 : board.p2;
}

template <typename T>
void set_indices_from_bits(const Bitboard100& bb, std::vector<T>& out, size_t base, T value) {
  uint64_t lo = bb.lo;
  while (lo != 0ULL) {
    const unsigned tz = static_cast<unsigned>(__builtin_ctzll(lo));
    out[base + static_cast<size_t>(tz)] = value;
    lo &= (lo - 1ULL);
  }

  uint64_t hi = bb.hi & kHighMask;
  while (hi != 0ULL) {
    const unsigned tz = static_cast<unsigned>(__builtin_ctzll(hi));
    out[base + static_cast<size_t>(64 + tz)] = value;
    hi &= (hi - 1ULL);
  }
}

}  // namespace

Board initial_board() {
  return Board{};
}

Board canonical_board(const Board& board, int8_t player) {
  if (player == 1) {
    return board;
  }
  Board out{};
  out.p1 = board.p2;
  out.p2 = board.p1;
  return out;
}

Board flipped_perspective(const Board& board) {
  return canonical_board(board, -1);
}

int get_valid_moves(const Board& board, int* moves_out) {
  Bitboard100 empty = empty_bits(board);
  int count = 0;

  while (empty.lo != 0ULL) {
    const unsigned tz = static_cast<unsigned>(__builtin_ctzll(empty.lo));
    moves_out[count++] = static_cast<int>(tz);
    empty.lo &= (empty.lo - 1ULL);
  }
  while (empty.hi != 0ULL) {
    const unsigned tz = static_cast<unsigned>(__builtin_ctzll(empty.hi));
    moves_out[count++] = 64 + static_cast<int>(tz);
    empty.hi &= (empty.hi - 1ULL);
  }
  return count;
}

bool is_full(const Board& board) {
  const uint64_t occ_lo = board.p1.lo | board.p2.lo;
  const uint64_t occ_hi = (board.p1.hi | board.p2.hi) & kHighMask;
  return occ_lo == ~0ULL && occ_hi == kHighMask;
}

bool apply_move(Board& board, int action, int8_t player) {
  if (!is_valid_action(action) || (player != 1 && player != -1)) {
    return false;
  }
  if (bit_test(board.p1, action) || bit_test(board.p2, action)) {
    return false;
  }

  if (player == 1) {
    bit_set(board.p1, action);
  } else {
    bit_set(board.p2, action);
  }
  return true;
}

bool check_win(const Board& board, int action, int8_t player) {
  if (!is_valid_action(action) || (player != 1 && player != -1)) {
    return false;
  }
  const Bitboard100& bits = player_bits(board, player);
  if (!bit_test(bits, action)) {
    return false;
  }

  const WinMaskTable& table = win_mask_table();
  const uint8_t count = table.count[static_cast<size_t>(action)];
  for (uint8_t i = 0; i < count; i++) {
    const Bitboard100& mask = table.masks[static_cast<size_t>(action)][i];
    if (bit_contains(bits, mask)) {
      return true;
    }
  }
  return false;
}

void encode_state(const Board& board, std::vector<float>& out_encoded) {
  out_encoded.assign(static_cast<size_t>(encoded_state_size()), 0.0f);
  set_indices_from_bits<float>(board.p2, out_encoded, 0, 1.0f);
  set_indices_from_bits<float>(
      empty_bits(board), out_encoded, static_cast<size_t>(kActionSize), 1.0f);
  set_indices_from_bits<float>(
      board.p1, out_encoded, static_cast<size_t>(2 * kActionSize), 1.0f);
}

void to_board_plane(const Board& board, std::vector<int8_t>& out_plane) {
  out_plane.assign(static_cast<size_t>(kActionSize), 0);
  set_indices_from_bits<int8_t>(board.p2, out_plane, 0, static_cast<int8_t>(-1));
  set_indices_from_bits<int8_t>(board.p1, out_plane, 0, static_cast<int8_t>(1));
}

int encoded_state_size() { return kInputChannels * kActionSize; }

}  // namespace gomoku
