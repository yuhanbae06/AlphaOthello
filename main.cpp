#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "config.h"
#include "game.h"
#include "search.h"

// ---------------------------------------------------------
// [Main] 테스트 실행
// ---------------------------------------------------------
int main() {
  typedef Quoridor Game;
  typedef typename Game::State State;

  // 2. MCTS 설정
  Config conf = Config::load();
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    std::cout << "[Info] CUDA Detected. Using GPU.\n";
    device = torch::Device(torch::kCUDA);
  }

  // 3. MCTS 생성 (ScriptModule 로드)
  try {
    torch::jit::script::Module net = torch::jit::load("model.pt");
    net.to(device);

    MCTS<Game> mcts(conf, net, device);
    State initial_state = Game::get_initial_state();
    auto start = std::chrono::high_resolution_clock::now();  // 시작 시간
    mcts.search(initial_state);
    auto end = std::chrono::high_resolution_clock::now();  // 종료 시간

    std::chrono::duration<double> diff = end - start;
    double seconds = diff.count();

    std::cout << "[Success] MCTS Search Completed without Errors!\n";

    std::cout << "\n================ MCTS SEARCH RESULTS ================"
              << std::endl;

    std::cout << "--------------------------------------" << std::endl;
    std::cout << "Total Time: " << seconds << "s" << std::endl;
    // NPS (Nodes Per Second) 계산: 총 탐색 횟수 / 걸린 시간
    std::cout << "NPS: " << conf.num_searches / seconds << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    const auto& tree = mcts.get_tree();
    const auto& root = tree[0];

    // 루트 노드의 자식들(실제 AI가 고민한 후보 수들)을 하나씩 검사
    for (int child_idx : root.children_idxs) {
      const auto& child = tree[child_idx];

      float win_rate = 0.0f;
      if (child.visit_count > 0) {
        // MCTS 특성상 부모 관점의 승률이므로 부호를 반전해서 출력 (필요에 따라
        // 조정)
        win_rate = (child.value_sum / child.visit_count);
      }

      std::cout << "[Action " << std::setw(2) << child.action_taken << "] "
                << "Visits: " << std::setw(4) << child.visit_count << " | "
                << "Value: " << std::fixed << std::setprecision(4) << win_rate
                << " | "
                << "Prior: " << std::setprecision(4) << child.prior
                << std::endl;
    }

    int best = mcts.get_best_action();
    std::cout << "-----------------------------------------------------"
              << std::endl;
    std::cout << ">>> AI's Final Choice: Action " << best << std::endl;
    std::cout << "=====================================================\n"
              << std::endl;
    std::cout << "Tree Size: " << tree.size() << std::endl;
    std::cout << "Root Children Count: " << root.children_idxs.size()
              << std::endl;

    if (root.children_idxs.empty()) {
      std::cout << "[Warning] Root has no children! Check your expand() or "
                   "get_valid_moves()."
                << std::endl;

      auto initial_moves = Quoridor::get_valid_moves(initial_state);
      std::cout << "Debug - Initial Moves Count: " << initial_moves.size()
                << std::endl;
    }

    return 0;
  } catch (const c10::Error& e) {
    std::cerr << "[Error] LibTorch Error: " << e.msg() << "\n";
    return -1;
  } catch (const std::exception& e) {
    std::cerr << "[Error] Exception: " << e.what() << "\n";
    return -1;
  }
}