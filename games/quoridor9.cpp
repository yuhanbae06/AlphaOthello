#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// [1] 128비트 자료형 정의 (GCC / Clang 전용)
namespace quoridor9 {

using bitboard = unsigned __int128;

struct State {
    bitboard p_bits[2];   // 81 bits (0..80)
    bitboard walls_h;     // 100 bits (10x10)
    bitboard walls_v;     // 100 bits (10x10)
    bitboard h_block;     
    bitboard v_block;     
    int8_t walls_left[2];
    int8_t turn;          // 0 or 1
    
    // 점프 상태 제어를 위한 변수
    bool is_jumping;      
    int8_t jumper_idx;    
    int8_t jump_dir;      // 진입 방향 (0: 상, 1: 하, 2: 좌, 3: 우)
};

struct BridgeMask {
    bitboard h_mask;      // 64비트를 넘어갈 수 있으므로 bitboard로 변경
    bitboard v_mask;
    uint8_t  base;        
};

class Quoridor {
public:
    // [2] 9x9 보드에 맞춘 상수 변경
    static const int SIZE = 9;                             
    static const int NUM_SQUARES = SIZE * SIZE;            // 81
    static const int WALL_SIZE = SIZE + 1;                 // 10
    static const int INNER_WALL = SIZE - 1;                // 8
    static const int WALL_CNT = WALL_SIZE * WALL_SIZE;     // 100
    static const int INNER_WALL_CNT = INNER_WALL * INNER_WALL; // 64
    static const int ACTION_SIZE = 4 + 2 * INNER_WALL_CNT + 1; 
    static const int ACTION_PASS = ACTION_SIZE - 1;
    static const int WALLS_LEFT = 10;                      // 10개로 변경

    bitboard L_MASK = 0, R_MASK = 0;
    bitboard goal_masks[2];
    bitboard VALID_WALL_MASK = 0; 

    BridgeMask bridge_h[WALL_CNT];
    BridgeMask bridge_v[WALL_CNT];
    bitboard H_EXPAND_LUT[WALL_CNT]; 
    bitboard V_EXPAND_LUT[WALL_CNT]; 

    // 128비트 전체 마스크
    static constexpr bitboard WALL_MASK_FULL = ~((bitboard)0);

    Quoridor() {
        for (int i = 0; i < NUM_SQUARES; i += SIZE) {
            L_MASK |= ((bitboard)1) << i;
            R_MASK |= ((bitboard)1) << (i + SIZE - 1);
        }

        bitboard first_row = (((bitboard)1) << SIZE) - 1;
        goal_masks[1] = first_row;
        goal_masks[0] = first_row << (SIZE * (SIZE - 1));

        // 플레이어가 벽을 둘 수 있는 유효 마스크 (1~8 범위)
        for (int r = 1; r <= SIZE - 1; ++r) {
            for (int c = 1; c <= SIZE - 1; ++c) {
                VALID_WALL_MASK |= (((bitboard)1) << (r * WALL_SIZE + c));
            }
        }

        init_expand_luts();
        init_bridge_masks();
    }

    State get_initial_state() const {
        State s{};
        // [3] 초기 위치 조정 (중앙)
        s.p_bits[0] = ((bitboard)1) << (SIZE / 2); // 맨 윗줄 중앙
        s.p_bits[1] = ((bitboard)1) << (NUM_SQUARES - 1 - SIZE / 2); // 맨 아랫줄 중앙
        s.walls_h = 0;
        s.walls_v = 0;
        
        // 보드의 상하좌우 끝부분을 영구적인 벽 비트로 설정
        for (int i = 0; i < WALL_SIZE; ++i) {
            s.walls_h |= (((bitboard)1) << (0 * WALL_SIZE + i));           // Top edge
            s.walls_h |= (((bitboard)1) << ((SIZE)*WALL_SIZE + i));        // Bottom edge
            s.walls_v |= (((bitboard)1) << (i * WALL_SIZE + 0));           // Left edge
            s.walls_v |= (((bitboard)1) << (i * WALL_SIZE + SIZE));        // Right edge
        }
        recompute_blocks(s);

        s.walls_left[0] = WALLS_LEFT;
        s.walls_left[1] = WALLS_LEFT;
        s.turn = 0;
        s.is_jumping = false;
        s.jumper_idx = -1;
        s.jump_dir = -1;
        return s;
    }

    void render(const State& state) const {
        std::cout << "  ";
        for (int c = 0; c < SIZE; ++c) std::cout << c << " ";
        std::cout << "\n";

        for (int r = 0; r < SIZE; ++r) {
            std::cout << r << " ";
            for (int c = 0; c < SIZE; ++c) {
                int idx = r * SIZE + c;
                if ((int)((state.p_bits[0] >> idx) & 1)) std::cout << "1";
                else if ((int)((state.p_bits[1] >> idx) & 1)) std::cout << "2";
                else std::cout << ".";

                if (c < SIZE - 1) {
                    bool has_v_wall = false;
                    if (state.walls_v & (((bitboard)1) << (r * WALL_SIZE + (c + 1)))) has_v_wall = true;
                    if (state.walls_v & (((bitboard)1) << ((r + 1) * WALL_SIZE + (c + 1)))) has_v_wall = true;
                    std::cout << (has_v_wall ? "|" : " ");
                } else {
                    std::cout << " ";
                }
            }
            std::cout << "\n";

            if (r < SIZE - 1) {
                std::cout << "  ";
                for (int c = 0; c < SIZE; ++c) {
                    bool has_h_wall = false;
                    if (state.walls_h & (((bitboard)1) << ((r + 1) * WALL_SIZE + c))) has_h_wall = true;
                    if (state.walls_h & (((bitboard)1) << ((r + 1) * WALL_SIZE + (c + 1)))) has_h_wall = true;
                    if (has_h_wall) std::cout << "- ";
                    else std::cout << "  ";
                }
                std::cout << "\n";
            }
        }
        std::cout << "Walls Left: P1=" << (int)state.walls_left[0]
                  << " P2=" << (int)state.walls_left[1] << "\n\n";
    }

    State apply_action(const State& state, int action_idx) const {
        State next_state = state;

        if (action_idx >= ACTION_PASS) {
            next_state.turn = 1 - state.turn;
            return next_state;
        }

        if (action_idx < 4) {
            static const int dr[] = {-1, 1, 0, 0};
            static const int dc[] = {0, 0, -1, 1};
            int from_idx = get_lsb_index(state.p_bits[state.turn]);
            int r = from_idx / SIZE, c = from_idx % SIZE;
            int to_idx = (r + dr[action_idx]) * SIZE + (c + dc[action_idx]);

            if (state.is_jumping) {
                next_state.p_bits[state.turn] = ((bitboard)1) << to_idx;
                next_state.is_jumping = false;
                next_state.jump_dir = -1;
            } else {
                int opp_idx = get_lsb_index(state.p_bits[1 - state.turn]);
                if (to_idx == opp_idx) {
                    next_state.p_bits[state.turn] = ((bitboard)1) << to_idx;
                    next_state.is_jumping = true;
                    next_state.jumper_idx = state.turn;
                    next_state.jump_dir = action_idx;
                } else {
                    next_state.p_bits[state.turn] = ((bitboard)1) << to_idx;
                }
            }
        } else if (action_idx < 4 + INNER_WALL_CNT) {
            const int compact_idx = action_idx - 4;
            const int wall_idx = compact_to_wall_idx(compact_idx);
            next_state.walls_h |= (((bitboard)1) << wall_idx);
            next_state.h_block |= H_EXPAND_LUT[wall_idx];
            next_state.walls_left[state.turn]--;
        } else {
            const int compact_idx = action_idx - (4 + INNER_WALL_CNT);
            const int wall_idx = compact_to_wall_idx(compact_idx);
            next_state.walls_v |= (((bitboard)1) << wall_idx);
            next_state.v_block |= V_EXPAND_LUT[wall_idx];
            next_state.walls_left[state.turn]--;
        }

        next_state.turn = 1 - state.turn;
        return next_state;
    }

    int get_valid_moves(State& state, int* moves_out) const {
        int count = 0;
        const int p_idx = state.turn;
        const int opp_idx = 1 - p_idx;

        static const int dr[] = {-1, 1, 0, 0};
        static const int dc[] = {0, 0, -1, 1};

        if (state.is_jumping) {
            if (state.turn != state.jumper_idx) {
                moves_out[count++] = ACTION_PASS;
                return count;
            } else {
                int curr_idx = get_lsb_index(state.p_bits[state.turn]);
                int r = curr_idx / SIZE, c = curr_idx % SIZE;
                
                int dir = state.jump_dir;
                for (int d : {dir, dir ^ 2, dir ^ 3}) {
                    int nr = r + dr[d], nc = c + dc[d];

                    if (nr >= 0 && nr < SIZE && nc >= 0 && nc < SIZE && !is_move_blocked(state, curr_idx, nr * SIZE + nc)) {
                        moves_out[count++] = d;
                        if (d == dir) return count;
                    }
                }
                return count;
            }
        }

        int curr_idx = get_lsb_index(state.p_bits[p_idx]);
        int r = curr_idx / SIZE, c = curr_idx % SIZE;

        for (int i = 0; i < 4; ++i) {
            int nr = r + dr[i], nc = c + dc[i];
            if (nr < 0 || nr >= SIZE || nc < 0 || nc >= SIZE) continue;

            if (!is_move_blocked(state, curr_idx, nr * SIZE + nc)) {
                int target_idx = nr * SIZE + nc;
                
                if (target_idx == get_lsb_index(state.p_bits[opp_idx])) {
                    int escapes = 0;
                    for (int j = 0; j < 4; ++j) {
                        if (j == (i ^ 1)) continue; 
                        
                        int b_nr = nr + dr[j], b_nc = nc + dc[j];
                        if (b_nr >= 0 && b_nr < SIZE && b_nc >= 0 && b_nc < SIZE) {
                            if (!is_move_blocked(state, target_idx, b_nr * SIZE + b_nc)) {
                                escapes++;
                            }
                        }
                    }
                    if (escapes == 0) continue; 
                }
                moves_out[count++] = i;
            }
        }

        if (state.walls_left[p_idx] > 0) {
            bitboard occupied = state.walls_h | state.walls_v;
            bitboard candidates = (~occupied) & VALID_WALL_MASK;

            while (candidates) {
                int idx = get_lsb_index(candidates);
                candidates &= (candidates - 1);

                int w_r = idx / WALL_SIZE;
                int w_c = idx % WALL_SIZE;
                bitboard wall_bit = ((bitboard)1) << idx;

                bool overlap_h = (state.walls_h & (wall_bit >> 1)) || (state.walls_h & (wall_bit << 1));
                if (!overlap_h) {
                    bitboard old_wh = state.walls_h, old_hb = state.h_block;
                    state.walls_h |= wall_bit;
                    state.h_block |= H_EXPAND_LUT[idx];

                    if (!is_bridge_fast(state, w_r, w_c, 0) || (has_path(state, 0) && has_path(state, 1))) {
                        moves_out[count++] = (4 + wall_idx_to_compact(idx));
                    }
                    state.walls_h = old_wh; state.h_block = old_hb;
                }

                bool overlap_v = (state.walls_v & (wall_bit >> WALL_SIZE)) || (state.walls_v & (wall_bit << WALL_SIZE));
                if (!overlap_v) {
                    bitboard old_wv = state.walls_v, old_vb = state.v_block;
                    state.walls_v |= wall_bit;
                    state.v_block |= V_EXPAND_LUT[idx];

                    if (!is_bridge_fast(state, w_r, w_c, 1) || (has_path(state, 0) && has_path(state, 1))) {
                        moves_out[count++] =
                            (4 + INNER_WALL_CNT + wall_idx_to_compact(idx));
                    }
                    state.walls_v = old_wv; state.v_block = old_vb;
                }
            }
        }
        return count;
    }

    bool is_move_blocked(const State& state, int from, int to) const {
        int r1 = from / SIZE, c1 = from % SIZE;
        int r2 = to / SIZE,   c2 = to % SIZE;

        if (r1 == r2) { 
            int w_c = std::max(c1, c2); 
            bitboard w1 = ((bitboard)1) << (r1 * WALL_SIZE + w_c);
            bitboard w2 = ((bitboard)1) << ((r1 + 1) * WALL_SIZE + w_c);
            return (state.walls_v & (w1 | w2)) != 0;
        } else {        
            int w_r = std::max(r1, r2); 
            bitboard w1 = ((bitboard)1) << (w_r * WALL_SIZE + c1);
            bitboard w2 = ((bitboard)1) << (w_r * WALL_SIZE + (c1 + 1));
            return (state.walls_h & (w1 | w2)) != 0;
        }
    }

    bool has_path(const State& state, int p_idx) const {
        bitboard reachable = state.p_bits[p_idx];
        bitboard goal = goal_masks[p_idx];
        const bitboard h_block = state.h_block;
        const bitboard v_block = state.v_block;

        while (true) {
            bitboard prev = reachable;

            bitboard up    = (reachable >> SIZE) & ~h_block;
            bitboard down  = (reachable << SIZE) & ~(h_block << SIZE);
            bitboard left  = ((reachable & ~L_MASK) >> 1) & ~v_block;
            bitboard right = ((reachable & ~R_MASK) << 1) & ~(v_block << 1);

            reachable |= (up | down | left | right);

            if (reachable & goal) return true;
            if (reachable == prev) return false;
        }
    }

    bool check_win(const State& state, int p_idx) const {
        return (state.p_bits[p_idx] & goal_masks[p_idx]) != 0;
    }

    void init_bridge_masks() {
        for (int r = 1; r < SIZE; ++r) {
            for (int c = 1; c < SIZE; ++c) {
                int idx = r * WALL_SIZE + c;

                {
                    bitboard hmask = 0, vmask = 0;
                    uint8_t base = 0;

                    if (c == 1) base++;
                    else {
                        for (int dr = -1; dr <= 1; ++dr) {
                            if (r + dr >= 1 && r + dr < SIZE)
                                vmask |= ((bitboard)1) << ((r + dr) * WALL_SIZE + (c - 1));
                        }
                    }

                    if (c == SIZE - 1) base++;
                    else {
                        for (int dr = -1; dr <= 1; ++dr) {
                            if (r + dr >= 1 && r + dr < SIZE)
                                vmask |= ((bitboard)1) << ((r + dr) * WALL_SIZE + (c + 1));
                        }
                    }

                    if (c >= 3) hmask |= ((bitboard)1) << (r * WALL_SIZE + (c - 2));
                    if (c + 2 < SIZE) hmask |= ((bitboard)1) << (r * WALL_SIZE + (c + 2));

                    if (r - 1 >= 1) vmask |= ((bitboard)1) << ((r - 1) * WALL_SIZE + c);
                    if (r + 1 < SIZE) vmask |= ((bitboard)1) << ((r + 1) * WALL_SIZE + c);

                    bridge_h[idx] = {hmask, vmask, base};
                }

                {
                    bitboard hmask = 0, vmask = 0;
                    uint8_t base = 0;

                    if (r == 1) base++;
                    else {
                        for (int dc = -1; dc <= 1; ++dc) {
                            if (c + dc >= 1 && c + dc < SIZE)
                                hmask |= ((bitboard)1) << ((r - 1) * WALL_SIZE + (c + dc));
                        }
                    }

                    if (r == SIZE - 1) base++;
                    else {
                        for (int dc = -1; dc <= 1; ++dc) {
                            if (c + dc >= 1 && c + dc < SIZE)
                                hmask |= ((bitboard)1) << ((r + 1) * WALL_SIZE + (c + dc));
                        }
                    }

                    if (r >= 3) vmask |= ((bitboard)1) << ((r - 2) * WALL_SIZE + c);
                    if (r + 2 < SIZE) vmask |= ((bitboard)1) << ((r + 2) * WALL_SIZE + c);

                    if (c - 1 >= 1) hmask |= ((bitboard)1) << (r * WALL_SIZE + (c - 1));
                    if (c + 1 < SIZE) hmask |= ((bitboard)1) << (r * WALL_SIZE + (c + 1));

                    bridge_v[idx] = {hmask, vmask, base};
                }
            }
        }
    }

    inline bool is_bridge_fast(const State& state, int r, int c, int type) const {
        int idx = r * WALL_SIZE + c;
        const auto& m = (type == 0) ? bridge_h[idx] : bridge_v[idx];
        bitboard combined = (state.walls_h & m.h_mask) | (state.walls_v & m.v_mask);
        return m.base ? (combined != 0) : (combined & (combined - 1)) != 0;
    }

    inline int compact_to_wall_idx(int compact_idx) const {
        const int r = 1 + (compact_idx / (SIZE - 1));
        const int c = 1 + (compact_idx % (SIZE - 1));
        return r * WALL_SIZE + c;
    }

    inline int wall_idx_to_compact(int wall_idx) const {
        const int r = wall_idx / WALL_SIZE;
        const int c = wall_idx % WALL_SIZE;
        return (r - 1) * (SIZE - 1) + (c - 1);
    }

    // [4] 128비트용 하위 비트 스캔 함수
    inline int get_lsb_index(bitboard v) const {
        if (v == 0) return 128; // 비어있는 경우
        uint64_t low = (uint64_t)v;
        if (low != 0) {
            return __builtin_ctzll(low);
        } else {
            uint64_t high = (uint64_t)(v >> 64);
            return 64 + __builtin_ctzll(high);
        }
    }

private:
    void init_expand_luts() {
        for (int w_r = 1; w_r <= SIZE - 1; ++w_r) {
            for (int w_c = 1; w_c <= SIZE - 1; ++w_c) {
                int idx = w_r * WALL_SIZE + w_c;

                bitboard hb = 0;
                hb |= ((bitboard)1) << ((w_r - 1) * SIZE + (w_c - 1));
                hb |= ((bitboard)1) << ((w_r - 1) * SIZE + w_c);
                H_EXPAND_LUT[idx] = hb;

                bitboard vb = 0;
                vb |= ((bitboard)1) << ((w_r - 1) * SIZE + (w_c - 1));
                vb |= ((bitboard)1) << (w_r * SIZE + (w_c - 1));
                V_EXPAND_LUT[idx] = vb;
            }
        }
    }

    void recompute_blocks(State& s) const {
        s.h_block = 0; s.v_block = 0;
        bitboard wh = s.walls_h & WALL_MASK_FULL;
        while (wh) {
            int i = get_lsb_index(wh);
            wh &= (wh - 1);
            if(i < WALL_CNT) s.h_block |= H_EXPAND_LUT[i];
        }
        bitboard wv = s.walls_v & WALL_MASK_FULL;
        while (wv) {
            int i = get_lsb_index(wv);
            wv &= (wv - 1);
            if(i < WALL_CNT) s.v_block |= V_EXPAND_LUT[i];
        }
    }
};

namespace {
const Quoridor& engine_instance() {
    static const Quoridor engine{};
    return engine;
}
int lsb_index(bitboard v) {
    if (v == 0) return 128;
    uint64_t low = static_cast<uint64_t>(v);
    if (low != 0) {
        return __builtin_ctzll(low);
    }
    uint64_t high = static_cast<uint64_t>(v >> 64);
    return 64 + __builtin_ctzll(high);
}
bitboard flip_bits(bitboard val, int max_bits) {
    bitboard res = 0;
    while (val) {
        const int idx = lsb_index(val);
        if (idx < max_bits) {
            const int new_idx = (max_bits - 1) - idx;
            res |= (static_cast<bitboard>(1) << new_idx);
        }
        val &= (val - 1);
    }
    return res;
}
void recompute_blocks_from_walls(State& s) {
    s.h_block = 0;
    s.v_block = 0;
    bitboard wh = s.walls_h;
    while (wh) {
        const int i = lsb_index(wh);
        wh &= (wh - 1);
        const int w_r = i / Quoridor::WALL_SIZE;
        const int w_c = i % Quoridor::WALL_SIZE;
        if (w_r >= 1 && w_r <= Quoridor::SIZE - 1 &&
            w_c >= 1 && w_c <= Quoridor::SIZE - 1) {
            s.h_block |= (static_cast<bitboard>(1) << ((w_r - 1) * Quoridor::SIZE + (w_c - 1)));
            s.h_block |= (static_cast<bitboard>(1) << ((w_r - 1) * Quoridor::SIZE + w_c));
        }
    }
    bitboard wv = s.walls_v;
    while (wv) {
        const int i = lsb_index(wv);
        wv &= (wv - 1);
        const int w_r = i / Quoridor::WALL_SIZE;
        const int w_c = i % Quoridor::WALL_SIZE;
        if (w_r >= 1 && w_r <= Quoridor::SIZE - 1 &&
            w_c >= 1 && w_c <= Quoridor::SIZE - 1) {
            s.v_block |= (static_cast<bitboard>(1) << ((w_r - 1) * Quoridor::SIZE + (w_c - 1)));
            s.v_block |= (static_cast<bitboard>(1) << (w_r * Quoridor::SIZE + (w_c - 1)));
        }
    }
}
int flip_jump_dir(int dir) {
    if (dir == 0) return 1;
    if (dir == 1) return 0;
    if (dir == 2) return 3;
    if (dir == 3) return 2;
    return -1;
}
}  // namespace
State get_initial_state() {
    return engine_instance().get_initial_state();
}
State apply_action(const State& state, int action_idx) {
    return engine_instance().apply_action(state, action_idx);
}
int get_valid_moves(const State& state, int* moves_out) {
    State mutable_state = state;
    return engine_instance().get_valid_moves(mutable_state, moves_out);
}
bool check_win_by_index(const State& state, int p_idx) {
    return engine_instance().check_win(state, p_idx);
}
State change_perspective(const State& state, int player) {
    if (player == 0) {
        return state;
    }
    State out{};
    out.p_bits[0] = flip_bits(state.p_bits[1], Quoridor::NUM_SQUARES);
    out.p_bits[1] = flip_bits(state.p_bits[0], Quoridor::NUM_SQUARES);
    out.walls_h = flip_bits(state.walls_h, Quoridor::WALL_CNT);
    out.walls_v = flip_bits(state.walls_v, Quoridor::WALL_CNT);
    out.walls_left[0] = state.walls_left[1];
    out.walls_left[1] = state.walls_left[0];
    out.turn = static_cast<int8_t>(1 - state.turn);
    out.is_jumping = state.is_jumping;
    if (state.is_jumping) {
        out.jumper_idx = static_cast<int8_t>(1 - state.jumper_idx);
        out.jump_dir = static_cast<int8_t>(flip_jump_dir(state.jump_dir));
    } else {
        out.jumper_idx = -1;
        out.jump_dir = -1;
    }
    recompute_blocks_from_walls(out);
    return out;
}
}  // namespace quoridor9

