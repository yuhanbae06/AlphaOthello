#pragma once
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

#include "types.h"

#ifdef _MSC_VER
#include <intrin.h>
#endif

class Quoridor {
 public:
  using State = ::State;

  static const int SIZE = 7;
  static const int NUM_SQUARES = SIZE * SIZE;
  static const int WALL_SIZE = SIZE - 1;
  static const int ACTION_SIZE = NUM_SQUARES + 2 * WALL_SIZE * WALL_SIZE;
  static const int WALLS_LEFT = 5;

  void render(const State& state) const {
    std::cout << "  ";
    for (int c = 0; c < SIZE; ++c) std::cout << c << " ";
    std::cout << "\n";

    for (int r = 0; r < SIZE; ++r) {
      std::cout << r << " ";
      for (int c = 0; c < SIZE; ++c) {
        int idx = r * SIZE + c;
        if ((state.p_bits[0] >> idx) & 1)
          std::cout << "1 ";  // P1
        else if ((state.p_bits[1] >> idx) & 1)
          std::cout << "2 ";  // P2
        else
          std::cout << ". ";
      }
      std::cout << "\n";

      // 가로 벽 출력 시각화 (간단 버전)
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
    std::cout << "Walls Left: P1=" << (int)state.walls_left[0]
              << " P2=" << (int)state.walls_left[1] << "\n\n";
  }

  // ------------------------------------------------------------
  // 1. L_MASK (왼쪽 벽 마스크) - 람다 함수로 즉시 초기화
  // ------------------------------------------------------------
  inline static const bitboard L_MASK = []() {
    bitboard m = 0;
    for (int i = 0; i < NUM_SQUARES; i += SIZE) {
      m |= 1ULL << i;
    }
    return m;
  }();

  // ------------------------------------------------------------
  // 2. R_MASK (오른쪽 벽 마스크)
  // ------------------------------------------------------------
  inline static const bitboard R_MASK = []() {
    bitboard m = 0;
    for (int i = 0; i < NUM_SQUARES; i += SIZE) {
      m |= 1ULL << (i + SIZE - 1);
    }
    return m;
  }();

  // ------------------------------------------------------------
  // 3. goal_masks (승리 조건 마스크)
  // ------------------------------------------------------------
  // 배열은 초기화 리스트 { } 안에 계산식을 바로 넣는 게 제일 깔끔합니다.
  inline static const bitboard goal_masks[2] = {
      // P0 목표: 마지막 행 (비트 시프트로 계산)
      ((1ULL << SIZE) - 1) << (SIZE * (SIZE - 1)),

      // P1 목표: 첫 번째 행
      (1ULL << SIZE) - 1};

  Quoridor() {}

  static State get_initial_state() {
    State s;
    s.p_bits[0] = 1ULL << (SIZE / 2);                    // 상단 중앙
    s.p_bits[1] = 1ULL << (NUM_SQUARES - 1 - SIZE / 2);  // 하단 중앙
    s.walls_h = 0;
    s.walls_v = 0;
    s.walls_left[0] = WALLS_LEFT;
    s.walls_left[1] = WALLS_LEFT;
    s.turn = 0;
    return s;
  }

  static State apply_action(const State& state, int action_idx) {
    State next_state = state;  // 현재 상태 복사
    int p_idx = state.turn;

    if (action_idx < NUM_SQUARES) {
      // 1. 말 이동 (0~48)
      next_state.p_bits[p_idx] = 1ULL << action_idx;
    } else if (action_idx < NUM_SQUARES + WALL_SIZE * WALL_SIZE) {
      // 2. 가로 벽 설치 (49~84)
      int wall_idx = action_idx - NUM_SQUARES;
      next_state.walls_h |= (1ULL << wall_idx);
      next_state.walls_left[p_idx]--;
    } else {
      // 3. 세로 벽 설치 (85~120)
      int wall_idx = action_idx - (NUM_SQUARES + WALL_SIZE * WALL_SIZE);
      next_state.walls_v |= (1ULL << wall_idx);
      next_state.walls_left[p_idx]--;
    }

    // 턴 교체
    next_state.turn = 1 - state.turn;
    return next_state;
  }

  // [최적화 1] 비트 병렬 Flood Fill: 전체 보드를 한 번에 확장
  static bool has_path(const State& state, int p_idx) {
    bitboard reachable = state.p_bits[p_idx];
    bitboard goal = goal_masks[p_idx];

    // 벽에 의한 이동 차단 마스크 생성 (한 번의 연산으로 전체 보드 적용)
    bitboard h_block = expand_h(state.walls_h);
    bitboard v_block = expand_v(state.walls_v);

    while (true) {
      bitboard prev = reachable;
      // 위로 이동 (Up): 현재 위치 >> 7, 단 아래칸 가로벽(h_block << 7)에
      // 막히지 않아야 함
      bitboard up = (reachable >> SIZE) & ~(h_block);
      // 아래로 이동 (Down): 현재 위치 << 7, 단 현재칸 가로벽(h_block)에
      // 막히지 않아야 함
      bitboard down = (reachable << SIZE) & ~(h_block << SIZE);
      // 왼쪽 이동 (Left): 현재 위치 >> 1, 가장자리 및 세로벽 차단
      bitboard left = ((reachable & ~L_MASK) >> 1) & ~(v_block);
      // 오른쪽 이동 (Right): 현재 위치 << 1, 가장자리 및 세로벽 차단
      bitboard right = ((reachable & ~R_MASK) << 1) & ~(v_block << 1);

      reachable |= (up | down | left | right);

      if (reachable & goal) return true;
      if (reachable == prev) return false;
    }
  }

  // [최적화 2] 벽 비트 확장 로직 (36 -> 49 Mapping)
  // h_walls의 1비트는 두 칸의 상하 이동을 동시에 막음

  static inline bitboard expand_h(bitboard w) {
    bitboard res = 0;
    bitboard wall_mask = (1ULL << WALL_SIZE) - 1;
    for (int r = 0; r < WALL_SIZE; r++) {
      bitboard row = (w >> (r * WALL_SIZE)) & wall_mask;
      res |= (row | (row << 1)) << (r * SIZE);
    }
    return res;
  }

  static inline bitboard expand_v(bitboard w) {
    bitboard res = 0;
    bitboard wall_mask = (1ULL << WALL_SIZE) - 1;
    for (int r = 0; r < WALL_SIZE; r++) {
      bitboard row = (w >> (r * WALL_SIZE)) & wall_mask;
      res |= row << (r * SIZE);
      res |= row << ((r + 1) * SIZE);
    }
    return res;
  }

  // 유효 수 계산 (Python의 jump/diagonal 로직 포함)
  static std::vector<int> get_valid_moves(State& state) {
    std::vector<int> moves;
    int p_idx = state.turn;
    int opp_idx = 1 - p_idx;
    bitboard my_pos = state.p_bits[p_idx];
    bitboard opp_pos = state.p_bits[opp_idx];
    int curr_idx = get_lsb_index(my_pos);

    // 1. 말 이동 로직
    static const int dr[] = {-1, 1, 0, 0};  // U, D, L, R
    static const int dc[] = {0, 0, -1, 1};

    for (int i = 0; i < 4; ++i) {
      int r = curr_idx / SIZE, c = curr_idx % SIZE;
      int nr = r + dr[i], nc = c + dc[i];
      if (nr < 0 || nr >= SIZE || nc < 0 || nc >= SIZE) continue;

      if (!is_move_blocked(state, curr_idx, nr * SIZE + nc)) {
        bitboard target_bit = 1ULL << (nr * SIZE + nc);
        if (target_bit != opp_pos) {
          moves.push_back(nr * SIZE + nc);
        } else {
          // 점프 로직
          int jnr = nr + dr[i], jnc = nc + dc[i];
          bool straight_jump = false;
          if (jnr >= 0 && jnr < SIZE && jnc >= 0 && jnc < SIZE &&
              !is_move_blocked(state, nr * SIZE + nc, jnr * SIZE + jnc)) {
            moves.push_back(jnr * SIZE + jnc);
            straight_jump = true;
          }
          if (!straight_jump) {
            // 대각선 점프
            for (int j = 0; j < 4; ++j) {
              if ((i < 2 && j >= 2) ||
                  (i >= 2 && j < 2)) {  // 수직->수평 or 수평->수직
                int dnr = nr + dr[j], dnc = nc + dc[j];
                if (dnr >= 0 && dnr < SIZE && dnc >= 0 && dnc < SIZE &&
                    !is_move_blocked(state, nr * SIZE + nc, dnr * SIZE + dnc)) {
                  moves.push_back(dnr * SIZE + dnc);
                }
              }
            }
          }
        }
      }
    }

    // 2. 벽 설치 로직
    if (state.walls_left[p_idx] > 0) {
      for (int r = 0; r < WALL_SIZE; ++r) {
        for (int c = 0; c < WALL_SIZE; ++c) {
          bitboard wall_bit = 1ULL << (r * WALL_SIZE + c);
          // 가로벽
          if (!(state.walls_h & wall_bit) && !(state.walls_v & wall_bit)) {
            bool overlap =
                (c > 0 && (state.walls_h & (wall_bit >> 1))) ||
                (c < WALL_SIZE - 1 && (state.walls_h & (wall_bit << 1)));
            if (!overlap) {
              state.walls_h |= wall_bit;
              if (has_path(state, 0) && has_path(state, 1))
                moves.push_back(NUM_SQUARES + r * WALL_SIZE + c);
              state.walls_h &= ~wall_bit;
            }
          }
          // 세로벽
          if (!(state.walls_v & wall_bit) && !(state.walls_h & wall_bit)) {
            bool overlap =
                (r > 0 && (state.walls_v & (wall_bit >> WALL_SIZE))) ||
                (r < WALL_SIZE - 1 &&
                 (state.walls_v & (wall_bit << WALL_SIZE)));
            if (!overlap) {
              state.walls_v |= wall_bit;
              if (has_path(state, 0) && has_path(state, 1))
                moves.push_back(NUM_SQUARES + WALL_SIZE * WALL_SIZE +
                                r * WALL_SIZE + c);
              state.walls_v &= ~wall_bit;
            }
          }
        }
      }
    }
    return moves;
  }

  // 특정 이동이 벽에 막혔는지 확인
  static bool is_move_blocked(const State& state, int from, int to) {
    int r1 = from / SIZE, c1 = from % SIZE;
    int r2 = to / SIZE, c2 = to % SIZE;
    if (r1 == r2) {  // 좌우 이동
      int min_c = std::min(c1, c2);
      bitboard wall_bit = 1ULL << (r1 * WALL_SIZE + min_c);
      if (r1 > 0 && (state.walls_v & (1ULL << ((r1 - 1) * WALL_SIZE + min_c))))
        return true;
      if (r1 < WALL_SIZE && (state.walls_v & wall_bit)) return true;
    } else {  // 상하 이동
      int min_r = std::min(r1, r2);
      bitboard wall_bit = 1ULL << (min_r * WALL_SIZE + c1);
      if (c1 > 0 && (state.walls_h & (1ULL << (min_r * WALL_SIZE + c1 - 1))))
        return true;
      if (c1 < WALL_SIZE && (state.walls_h & wall_bit)) return true;
    }
    return false;
  }

  static bool check_win(const State& state, int p_idx) {
    // 플레이어의 위치 비트와 해당 플레이어의 목표 마스크를 AND 연산
    return (state.p_bits[p_idx] & goal_masks[p_idx]) != 0;
  }

  static State get_next_state(const State& state, int action_idx) {
    return apply_action(state, action_idx);
  }

  static State change_perspective(const State& state, int player) {
    if (player == 0) return state;

    State new_state;
    new_state.turn = 0;

    new_state.walls_left[0] = state.walls_left[1];
    new_state.walls_left[1] = state.walls_left[0];

    new_state.p_bits[0] = flip_bits(state.p_bits[1], NUM_SQUARES);
    new_state.p_bits[1] = flip_bits(state.p_bits[0], NUM_SQUARES);

    new_state.walls_h = flip_bits(state.walls_h, WALL_SIZE * WALL_SIZE);
    new_state.walls_v = flip_bits(state.walls_v, WALL_SIZE * WALL_SIZE);

    return new_state;
  }

  static std::vector<float> get_encoded_state(const State& state) {
    // 채널 6개, 7x7 크기
    std::vector<float> encoded(6 * NUM_SQUARES, 0.0f);

    int current_turn = state.turn;         // 0 or 1
    bool need_flip = (current_turn == 1);  // P2 차례면 보드 뒤집어서 저장

    int my_idx = current_turn;
    int opp_idx = 1 - current_turn;

    // Plane 0: My Position
    if (state.p_bits[my_idx]) {
      int idx = get_lsb_index(state.p_bits[my_idx]);
      int final_idx = need_flip ? (NUM_SQUARES - 1 - idx) : idx;
      encoded[0 * NUM_SQUARES + final_idx] = 1.0f;
    }

    // Plane 1: Opponent Position
    if (state.p_bits[opp_idx]) {
      int idx = get_lsb_index(state.p_bits[opp_idx]);
      int final_idx = need_flip ? (NUM_SQUARES - 1 - idx) : idx;
      encoded[1 * NUM_SQUARES + final_idx] = 1.0f;
    }

    // Helper Lambda for Walls
    // 벽 인덱스(0~35)를 7x7 그리드의 좌표로 매핑 후 1.0f 할당
    auto fill_wall_plane = [&](int plane_offset, bitboard walls) {
      bitboard temp = walls;
      while (temp) {
        int w_idx = get_lsb_index(temp);
        int final_w_idx =
            need_flip ? (WALL_SIZE * WALL_SIZE - 1 - w_idx) : w_idx;

        // Wall Index(0~35) -> 7x7 Grid Index(0~48) 매핑
        // 6x6 격자는 7x7 격자에서 (0,0)~(5,5)에 해당하므로 행/열 계산 필요
        int r = final_w_idx / WALL_SIZE;
        int c = final_w_idx % WALL_SIZE;

        // 7x7 평면상의 인덱스 (stride = 7)
        int grid_idx = r * SIZE + c;
        encoded[plane_offset * NUM_SQUARES + grid_idx] = 1.0f;

        temp &= (temp - 1);
      }
    };

    // Plane 2: Horizontal Walls
    fill_wall_plane(2, state.walls_h);

    // Plane 3: Vertical Walls
    fill_wall_plane(3, state.walls_v);

    // Plane 4: My Walls Left (Scalar plane)
    float my_val = (float)state.walls_left[my_idx] /
                   (float)WALLS_LEFT;  // 6.0f: Initial walls
    std::fill(encoded.begin() + 4 * NUM_SQUARES,
              encoded.begin() + 5 * NUM_SQUARES, my_val);

    // Plane 5: Opponent Walls Left
    float opp_val = (float)state.walls_left[opp_idx] / (float)WALLS_LEFT;
    std::fill(encoded.begin() + 5 * NUM_SQUARES,
              encoded.begin() + 6 * NUM_SQUARES, opp_val);

    return encoded;
  }

  // 4. Utils
  static int get_opponent(int player) { return 1 - player; }
  static float get_opponent_value(float value) { return -value; }

  static inline int get_lsb_index(uint64_t v) {
    if (v == 0) return 64;  // 안전 장치
#ifdef _MSC_VER
    // Visual Studio
    unsigned long index;
    _BitScanForward64(&index, v);
    return (int)index;
#else
    // GCC, Clang
    return __builtin_ctzll(v);
#endif
  }

  static inline uint64_t flip_bits(uint64_t val, int max_bits) {
    uint64_t res = 0;
    while (val) {
      int idx = get_lsb_index(val);
      int new_idx = (max_bits - 1) - idx;
      res |= (1ULL << new_idx);
      val &= (val - 1);  // 최하위 비트 삭제
    }
    return res;
  }
};
