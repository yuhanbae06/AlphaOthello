#pragma once
#include <cstdint>

typedef uint64_t bitboard;

struct State {
  bitboard p_bits[2];
  bitboard walls_h;
  bitboard walls_v;
  int8_t walls_left[2];
  int8_t turn;
};

struct ActionProb {
  int action_idx;
  float prob;
};