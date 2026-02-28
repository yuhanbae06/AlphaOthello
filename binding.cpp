#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <random>
#include <string>
#include <vector>

// 루트 디렉토리에 있는 실제 헤더 파일명들
#include "onnx_infer.h"
#include "games/quoridor9.h"
#include "search.h"

namespace py = pybind11;
using namespace quoridor9;

// ── 128비트 <-> 문자열 안전 변환 헬퍼 ──
std::string int128_to_string(bitboard v) {
  if (v == 0) return "0";
  std::string s;
  while (v > 0) {
    s += std::to_string((int)(v % 10));
    v /= 10;
  }
  std::reverse(s.begin(), s.end());
  return s;
}

bitboard string_to_int128(const std::string& s) {
  bitboard v = 0;
  for (char c : s) {
    if (c >= '0' && c <= '9') {
      v = v * 10 + (c - '0');
    }
  }
  return v;
}

// ── 프론트/팀원 맞춤형 액션 복원 (Unflip) ──
int unflip_action(int a) {
  if (a == 0) return 1;  // Up <-> Down
  if (a == 1) return 0;
  if (a == 2) return 3;  // Left <-> Right
  if (a == 3) return 2;
  // 수평 벽 복원 (0~63 인덱스 뒤집기)
  if (a >= 4 && a < 4 + INNER_WALL_CNT) return 4 + (INNER_WALL_CNT - 1 - (a - 4));
  // 수직 벽 복원
  if (a >= 4 + INNER_WALL_CNT && a < 4 + 2 * INNER_WALL_CNT)
    return 4 + INNER_WALL_CNT + (INNER_WALL_CNT - 1 - (a - (4 + INNER_WALL_CNT)));
  return a;  // PASS
}

// ── C++ MCTS 엔진 상태 유지 클래스 ──
class QuoridorAIWrapper {
public:
  OnnxInfer infer;
  std::mt19937 rng;

  // 🔥 에러 1 해결: 인자 4개 (경로, GPU 사용, 배치 1, 스레드 1) 맞춰주기
  QuoridorAIWrapper(const std::string& model_path) : infer(model_path, true, 1, 1) {
    rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
  }

  py::tuple get_ai_move(const State& original_state, int num_searches) {
    bool flipped = (original_state.turn == 1);
    State canonical_state = flipped ? change_perspective(original_state, 1) : original_state;

    // 🔥 에러 2 해결: search.cpp에 뚫어놓은 심부름꾼 함수 호출
    std::vector<float> probs = get_ai_action_probs(infer, canonical_state, num_searches);

    std::vector<float> final_probs(ACTION_SIZE, 0.0f);
    int best_action = -1;
    float best_prob = -1.0f;

    for (int a = 0; a < ACTION_SIZE; ++a) {
      if (probs[a] > 0) {
        int target_a = flipped ? unflip_action(a) : a;
        final_probs[target_a] = probs[a];

        if (probs[a] > best_prob) {
          best_prob = probs[a];
          best_action = target_a;
        }
      }
    }

    return py::make_tuple(best_action, final_probs);
  }
};

// ── 파이썬 바인딩 모듈 ──
PYBIND11_MODULE(quoridor_engine, m) {
  m.doc() = "Quoridor Bitboard Engine with ONNX MCTS";

  // 1. State 구조체 바인딩 (__int128은 문자열 속성으로 매핑)
  py::class_<State>(m, "State")
      .def(py::init<>())
      .def_readwrite("turn", &State::turn)
      .def_readwrite("is_jumping", &State::is_jumping)
      .def_readwrite("jump_dir", &State::jump_dir)
      .def_property(
          "walls_left", [](const State& s) { return std::vector<int8_t>{s.walls_left[0], s.walls_left[1]}; },
          [](State& s, std::vector<int8_t> v) {
            s.walls_left[0] = v[0];
            s.walls_left[1] = v[1];
          })
      .def_property(
          "p0_bits_str", [](const State& s) { return int128_to_string(s.p_bits[0]); },
          [](State& s, const std::string& val) { s.p_bits[0] = string_to_int128(val); })
      .def_property(
          "p1_bits_str", [](const State& s) { return int128_to_string(s.p_bits[1]); },
          [](State& s, const std::string& val) { s.p_bits[1] = string_to_int128(val); })
      .def_property(
          "walls_h_str", [](const State& s) { return int128_to_string(s.walls_h); },
          [](State& s, const std::string& val) { s.walls_h = string_to_int128(val); })
      .def_property(
          "walls_v_str", [](const State& s) { return int128_to_string(s.walls_v); },
          [](State& s, const std::string& val) { s.walls_v = string_to_int128(val); });

  // 2. 게임 핵심 로직 바인딩
  m.def("get_initial_state", &get_initial_state);
  m.def("apply_action", &apply_action);
  m.def("check_win", &check_win_by_index);

  // get_valid_moves 포인터 배열을 Python 리스트로 변환
  m.def("get_valid_moves", [](const State& state) {
    int moves[ACTION_SIZE];
    int count = get_valid_moves(state, moves);
    std::vector<int> result(moves, moves + count);
    return result;
  });

  // 3. AI 래퍼 바인딩
  py::class_<QuoridorAIWrapper>(m, "QuoridorAI")
      .def(py::init<const std::string&>())
      .def("get_ai_move", &QuoridorAIWrapper::get_ai_move);

  // 4. 상수 노출
  m.attr("SIZE") = SIZE;
  m.attr("WALL_SIZE") = WALL_SIZE;
  m.attr("ACTION_SIZE") = ACTION_SIZE;
}