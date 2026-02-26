#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include "quoridor.h"

using bitboard = uint64_t;

namespace Quoridor {

// 0,7,14,21,28,35,42 bits
bitboard L_MASK = 0, R_MASK = 0;
bitboard goal_masks[2];

BridgeMask bridge_h[WALL_CNT];
BridgeMask bridge_v[WALL_CNT];

// Incremental block LUTs: placing a single wall adds these bits to block masks.
bitboard H_EXPAND_LUT[WALL_CNT];  // 36 -> 49 for horizontal walls
bitboard V_EXPAND_LUT[WALL_CNT];  // 36 -> 49 for vertical walls

constexpr bitboard WALL_MASK_36 = (WALL_CNT == 64) ? ~0ULL : ((1ULL << WALL_CNT) - 1ULL);

void init_expand_luts() {
  for (int r = 0; r < WALL_SIZE; ++r) {
    for (int c = 0; c < WALL_SIZE; ++c) {
      int idx = r * WALL_SIZE + c;

      // Horizontal wall at (r,c) affects two squares in row r: (r,c) and (r,c+1)
      // Matches your expand_h: (row | row<<1) << (r*SIZE)
      bitboard hb = 0;
      hb |= 1ULL << (r * SIZE + c);
      hb |= 1ULL << (r * SIZE + (c + 1));
      H_EXPAND_LUT[idx] = hb;

      // Vertical wall at (r,c) affects two squares in col c: rows r and r+1
      // Matches your expand_v: row << (r*SIZE) and row << ((r+1)*SIZE)
      bitboard vb = 0;
      vb |= 1ULL << (r * SIZE + c);
      vb |= 1ULL << ((r + 1) * SIZE + c);
      V_EXPAND_LUT[idx] = vb;
    }
  }
}

inline int get_lsb_index(uint64_t v) {
  if (v == 0) return 64;
#ifdef _MSC_VER
  unsigned long index;
  _BitScanForward64(&index, v);
  return (int)index;
#else
  return __builtin_ctzll(v);
#endif
}

inline uint64_t flip_bits(uint64_t val, int max_bits) {
  uint64_t res = 0;
  while (val) {
    int idx = get_lsb_index(val);
    int new_idx = (max_bits - 1) - idx;
    res |= (1ULL << new_idx);
    val &= (val - 1);
  }
  return res;
}

void recompute_blocks(State& s) {
  s.h_block = 0;
  s.v_block = 0;

  bitboard wh = s.walls_h & WALL_MASK_36;
  while (wh) {
    int i = get_lsb_index(wh);
    wh &= (wh - 1);
    s.h_block |= H_EXPAND_LUT[i];
  }

  bitboard wv = s.walls_v & WALL_MASK_36;
  while (wv) {
    int i = get_lsb_index(wv);
    wv &= (wv - 1);
    s.v_block |= V_EXPAND_LUT[i];
  }
}

// Bridge masks (same logic you built: conservative, base is edge anchor count).
void init_bridge_masks() {
  for (int r = 0; r < WALL_SIZE; ++r) {
    for (int c = 0; c < WALL_SIZE; ++c) {
      int idx = r * WALL_SIZE + c;

      // Horizontal wall masks
      {
        uint64_t hmask = 0, vmask = 0;
        uint8_t base = 0;

        if (c == 0) base++;
        if (c == WALL_SIZE - 1) base++;

        if (c >= 2) hmask |= 1ULL << (r * WALL_SIZE + (c - 2));
        if (c + 2 < WALL_SIZE) hmask |= 1ULL << (r * WALL_SIZE + (c + 2));

        for (int dr = -1; dr <= 1; ++dr) {
          int rr = r + dr;
          if (0 <= rr && rr < WALL_SIZE) {
            if (c - 1 >= 0) vmask |= 1ULL << (rr * WALL_SIZE + (c - 1));
            if (c + 1 < WALL_SIZE) vmask |= 1ULL << (rr * WALL_SIZE + (c + 1));
          }
        }

        if (r - 1 >= 0) vmask |= 1ULL << ((r - 1) * WALL_SIZE + c);
        if (r + 1 < WALL_SIZE) vmask |= 1ULL << ((r + 1) * WALL_SIZE + c);

        bridge_h[idx] = {hmask, vmask, base};  // base is 0/1/2, but in 6x6 edges it can be 2 at corners of wall-grid
      }

      // Vertical wall masks
      {
        uint64_t hmask = 0, vmask = 0;
        uint8_t base = 0;

        if (r == 0) base++;
        if (r == WALL_SIZE - 1) base++;

        if (r >= 2) vmask |= 1ULL << ((r - 2) * WALL_SIZE + c);
        if (r + 2 < WALL_SIZE) vmask |= 1ULL << ((r + 2) * WALL_SIZE + c);

        for (int dc = -1; dc <= 1; ++dc) {
          int cc = c + dc;
          if (0 <= cc && cc < WALL_SIZE) {
            if (r - 1 >= 0) hmask |= 1ULL << ((r - 1) * WALL_SIZE + cc);
            if (r + 1 < WALL_SIZE) hmask |= 1ULL << ((r + 1) * WALL_SIZE + cc);
          }
        }

        if (c - 1 >= 0) vmask |= 1ULL << (r * WALL_SIZE + (c - 1));
        if (c + 1 < WALL_SIZE) vmask |= 1ULL << (r * WALL_SIZE + (c + 1));

        bridge_v[idx] = {hmask, vmask, base};
      }
    }
  }
}

void init() {
  static bool initialized = false;
  if (initialized) return;

  for (int i = 0; i < NUM_SQUARES; i += SIZE) {
    L_MASK |= 1ULL << i;
    R_MASK |= 1ULL << (i + SIZE - 1);
  }

  bitboard first_row = (1ULL << SIZE) - 1;
  goal_masks[1] = first_row;
  goal_masks[0] = first_row << (SIZE * (SIZE - 1));

  init_expand_luts();
  init_bridge_masks();
  initialized = true;
}

State get_initial_state() {
  init();
  State s{};
  s.p_bits[0] = 1ULL << (SIZE / 2);
  s.p_bits[1] = 1ULL << (NUM_SQUARES - 1 - SIZE / 2);
  s.walls_h = 0;
  s.walls_v = 0;
  s.h_block = 0;
  s.v_block = 0;
  s.walls_left[0] = WALLS_LEFT;
  s.walls_left[1] = WALLS_LEFT;
  s.turn = 0;
  return s;
}

void render(const State& state) {
  std::cout << "  ";
  for (int c = 0; c < SIZE; ++c) std::cout << c << " ";
  std::cout << "\n";

  for (int r = 0; r < SIZE; ++r) {
    std::cout << r << " ";
    for (int c = 0; c < SIZE; ++c) {
      int idx = r * SIZE + c;
      if ((state.p_bits[0] >> idx) & 1)
        std::cout << "1";
      else if ((state.p_bits[1] >> idx) & 1)
        std::cout << "2";
      else
        std::cout << ".";

      if (c < SIZE - 1) {
        bool has_v_wall = false;
        if (r > 0 && (state.walls_v & (1ULL << ((r - 1) * WALL_SIZE + c)))) has_v_wall = true;
        if (r < WALL_SIZE && (state.walls_v & (1ULL << (r * WALL_SIZE + c)))) has_v_wall = true;
        std::cout << (has_v_wall ? "|" : " ");
      } else {
        std::cout << " ";
      }
    }
    std::cout << "\n";

    if (r < WALL_SIZE) {
      std::cout << "  ";
      for (int c = 0; c < WALL_SIZE; ++c) {
        if ((state.walls_h >> (r * WALL_SIZE + c)) & 1)
          std::cout << "- ";
        else
          std::cout << "  ";
      }
      std::cout << "\n";
    }
  }
  std::cout << "Walls Left: P1=" << (int)state.walls_left[0] << " P2=" << (int)state.walls_left[1] << "\n\n";
}

State apply_action(const State& state, int action_idx) {
  State next_state = state;
  int p_idx = state.turn;

  if (action_idx < NUM_SQUARES) {
    next_state.p_bits[p_idx] = 1ULL << action_idx;
  } else if (action_idx < NUM_SQUARES + WALL_CNT) {
    int wall_idx = action_idx - NUM_SQUARES;
    bitboard bit = 1ULL << wall_idx;
    next_state.walls_h |= bit;
    next_state.h_block |= H_EXPAND_LUT[wall_idx];
    next_state.walls_left[p_idx]--;
  } else {
    int wall_idx = action_idx - (NUM_SQUARES + WALL_CNT);
    bitboard bit = 1ULL << wall_idx;
    next_state.walls_v |= bit;
    next_state.v_block |= V_EXPAND_LUT[wall_idx];
    next_state.walls_left[p_idx]--;
  }

  next_state.turn = 1 - state.turn;
  return next_state;
}

// Flood fill uses precomputed blocks in state (NO expand_h/expand_v inside).
bool has_path(const State& state, int p_idx) {
  bitboard reachable = state.p_bits[p_idx];
  bitboard goal = goal_masks[p_idx];
  const bitboard h_block = state.h_block;
  const bitboard v_block = state.v_block;

  while (true) {
    bitboard prev = reachable;

    bitboard up = (reachable >> SIZE) & ~h_block;
    bitboard down = (reachable << SIZE) & ~(h_block << SIZE);
    bitboard left = ((reachable & ~L_MASK) >> 1) & ~v_block;
    bitboard right = ((reachable & ~R_MASK) << 1) & ~(v_block << 1);

    reachable |= (up | down | left | right);

    if (reachable & goal) return true;
    if (reachable == prev) return false;
  }
}

// base can be 0/1/2; we handle 2 safely too.
inline bool is_bridge_fast(const State& state, int r, int c, int type) {
  int idx = r * WALL_SIZE + c;
  const auto& m = (type == 0) ? bridge_h[idx] : bridge_v[idx];

  uint64_t combined = (state.walls_h & m.h_mask) | (state.walls_v & m.v_mask);

  // anchors = popcount(combined) + base >= 2  (without popcount)
  // base==2 => always true
  // base==1 => need >=1 bit in combined
  // base==0 => need >=2 bits in combined (combined & (combined-1)) != 0
  return m.base ? (combined != 0) : (combined & (combined - 1)) != 0;
}

// Move blocked check (unchanged; still uses walls_h / walls_v)
bool is_move_blocked(const State& state, int from, int to) {
  int r1 = from / SIZE, c1 = from % SIZE;
  int r2 = to / SIZE, c2 = to % SIZE;

  if (r1 == r2) {  // left/right
    int min_c = std::min(c1, c2);
    bitboard wall_bit = 1ULL << (r1 * WALL_SIZE + min_c);
    if (r1 > 0 && (state.walls_v & (1ULL << ((r1 - 1) * WALL_SIZE + min_c)))) return true;
    if (r1 < WALL_SIZE && (state.walls_v & wall_bit)) return true;
  } else {  // up/down
    int min_r = std::min(r1, r2);
    bitboard wall_bit = 1ULL << (min_r * WALL_SIZE + c1);
    if (c1 > 0 && (state.walls_h & (1ULL << (min_r * WALL_SIZE + c1 - 1)))) return true;
    if (c1 < WALL_SIZE && (state.walls_h & wall_bit)) return true;
  }
  return false;
}

// Valid move generation
int get_valid_moves(const State& in_state, int* moves_out) {
  State state = in_state;
  int count = 0;
  const int p_idx = state.turn;
  const int opp_idx = 1 - p_idx;

  // 1) Pawn moves (same as your code)
  bitboard my_pos = state.p_bits[p_idx];
  bitboard opp_pos = state.p_bits[opp_idx];
  const int curr_idx = get_lsb_index(my_pos);

  static const int dr[] = {-1, 1, 0, 0};
  static const int dc[] = {0, 0, -1, 1};

  for (int i = 0; i < 4; ++i) {
    int r = curr_idx / SIZE, c = curr_idx % SIZE;
    int nr = r + dr[i], nc = c + dc[i];
    if (nr < 0 || nr >= SIZE || nc < 0 || nc >= SIZE) continue;

    if (!is_move_blocked(state, curr_idx, nr * SIZE + nc)) {
      bitboard target_bit = 1ULL << (nr * SIZE + nc);
      if (target_bit != opp_pos) {
        moves_out[count++] = (nr * SIZE + nc);
      } else {
        int jnr = nr + dr[i], jnc = nc + dc[i];
        bool straight_jump = false;

        if (jnr >= 0 && jnr < SIZE && jnc >= 0 && jnc < SIZE &&
            !is_move_blocked(state, nr * SIZE + nc, jnr * SIZE + jnc)) {
          moves_out[count++] = (jnr * SIZE + jnc);
          straight_jump = true;
        }

        if (!straight_jump) {
          for (int j = 0; j < 4; ++j) {
            if ((i < 2 && j >= 2) || (i >= 2 && j < 2)) {
              int dnr = nr + dr[j], dnc = nc + dc[j];
              if (dnr >= 0 && dnr < SIZE && dnc >= 0 && dnc < SIZE &&
                  !is_move_blocked(state, nr * SIZE + nc, dnr * SIZE + dnc)) {
                moves_out[count++] = (dnr * SIZE + dnc);
              }
            }
          }
        }
      }
    }
  }

  // 2) Wall moves: two applied improvements
  //   (A) Iterate only EMPTY wall cells via bit iteration, not full 36 scan.
  //   (B) has_path uses state.h_block/v_block (incremental), no expand_* per call.
  if (state.walls_left[p_idx] > 0) {
    bitboard occupied = (state.walls_h | state.walls_v) & WALL_MASK_36;
    bitboard candidates = (~occupied) & WALL_MASK_36;

    while (candidates) {
      int idx = get_lsb_index(candidates);
      candidates &= (candidates - 1);

      int r = idx / WALL_SIZE;
      int c = idx % WALL_SIZE;
      bitboard wall_bit = 1ULL << idx;

      // Horizontal placement at idx
      {
        // overlap with adjacent horizontal (same row, c-1 or c+1)
        bool overlap =
            (c > 0 && (state.walls_h & (wall_bit >> 1))) || (c < (WALL_SIZE - 1) && (state.walls_h & (wall_bit << 1)));

        if (!overlap) {
          // Apply incrementally
          bitboard old_wh = state.walls_h;
          bitboard old_hb = state.h_block;

          state.walls_h = old_wh | wall_bit;
          state.h_block = old_hb | H_EXPAND_LUT[idx];

          if (!is_bridge_fast(state, r, c, 0) || (has_path(state, 0) && has_path(state, 1))) {
            moves_out[count++] = (NUM_SQUARES + idx);
          }

          // Revert
          state.walls_h = old_wh;
          state.h_block = old_hb;
        }
      }

      // Vertical placement at idx
      {
        // overlap with adjacent vertical (same col, r-1 or r+1 in wall-grid)
        bool overlap = (r > 0 && (state.walls_v & (wall_bit >> WALL_SIZE))) ||
                       (r < (WALL_SIZE - 1) && (state.walls_v & (wall_bit << WALL_SIZE)));

        if (!overlap) {
          bitboard old_wv = state.walls_v;
          bitboard old_vb = state.v_block;

          state.walls_v = old_wv | wall_bit;
          state.v_block = old_vb | V_EXPAND_LUT[idx];

          if (!is_bridge_fast(state, r, c, 1) || (has_path(state, 0) && has_path(state, 1))) {
            moves_out[count++] = (NUM_SQUARES + WALL_CNT + idx);
          }

          state.walls_v = old_wv;
          state.v_block = old_vb;
        }
      }
    }
  }

  return count;
}

bool check_win(const State& state, int p_idx) { return (state.p_bits[p_idx] & goal_masks[p_idx]) != 0; }

// If you use change_perspective elsewhere: recompute blocks from wall bits to keep correctness.
State change_perspective(const State& state, int player) {
  if (player == 0) return state;

  State new_state{};
  new_state.turn = 0;

  new_state.walls_left[0] = state.walls_left[1];
  new_state.walls_left[1] = state.walls_left[0];

  new_state.p_bits[0] = flip_bits(state.p_bits[1], NUM_SQUARES);
  new_state.p_bits[1] = flip_bits(state.p_bits[0], NUM_SQUARES);

  new_state.walls_h = flip_bits(state.walls_h, WALL_CNT);
  new_state.walls_v = flip_bits(state.walls_v, WALL_CNT);

  recompute_blocks(new_state);
  return new_state;
}

}  // namespace Quoridor
