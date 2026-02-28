from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import torch

# 1. C++ 엔진 임포트 (심볼릭 링크나 경로 설정이 되어있다고 가정)

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from NeuralNet import ResNet
from Game import Quoridor5, Quoridor7, Quoridor9
import quoridor_engine

MODEL_PATH = "../saved_model/model_0_Quoridor9.pt" 
ONNX_PATH = "../saved_model/model_best.onnx"

ai_engine = None

app = FastAPI(title="Quoridor API Server (C++ Engine Powered)")

game = Quoridor9()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic 모델 (프론트엔드 팀원 규격 유지) ──
class GameState(BaseModel):
    p_bits: List[Union[int, str]]
    walls_h: Union[int, str]
    walls_v: Union[int, str]
    walls_left: List[int]
    turn: int

class MoveRequest(BaseModel):
    state: GameState
    current_player: int  # 프론트 기준: 1 (사람) or -1 (AI)
    action: int

class AIRequest(BaseModel):
    state: GameState
    num_searches: int = 100

def map_turn_to_cpp(frontend_player: int) -> int:
    """프론트의 1/-1을 C++ 엔진의 0/1로 변환"""
    return 0 if frontend_player == 1 else 1

def map_turn_to_frontend(cpp_turn: int) -> int:
    """C++ 엔진의 0/1을 프론트의 1/-1로 변환"""
    return 1 if cpp_turn == 0 else -1

def dict_to_cpp_state(req: GameState, current_player: int):
    # 1. 깡통 State 대신, 내부가 깨끗하게 0으로 초기화된 상태를 가져옵니다.
    cpp_state = quoridor_engine.get_initial_state()
            
    # 2. 현재 프론트엔드의 화면 상태로 덮어씌우기
    cpp_state.p0_bits_str = str(req.p_bits[0])
    cpp_state.p1_bits_str = str(req.p_bits[1])
    cpp_state.walls_h_str = str(req.walls_h)
    cpp_state.walls_v_str = str(req.walls_v)
    cpp_state.walls_left = [int(req.walls_left[0]), int(req.walls_left[1])]
    
    cpp_state.turn = 0 if current_player == 1 else 1

    return cpp_state

def cpp_state_to_dict(cpp_state) -> dict:
    return {
        "p_bits": [cpp_state.p0_bits_str, cpp_state.p1_bits_str],
        "walls_h": cpp_state.walls_h_str,
        "walls_v": cpp_state.walls_v_str,
        "walls_left": cpp_state.walls_left,
        "turn": map_turn_to_frontend(cpp_state.turn)
    }


if not os.path.exists(ONNX_PATH):
    export_to_onnx()

# ══════════════ API 엔드포인트 ═══════════════════════════════

@app.get("/api/init")
def init_game():
    global global_cpp_state
    # 엔진이 완벽하게 초기화(캐시 0점 조절)한 상태를 가져옵니다.
    global_cpp_state = quoridor_engine.get_initial_state()
    return {
        "state": cpp_state_to_dict(global_cpp_state),
        "current_player": 1,
        "game_over": False,
        "winner": None
    }

def get_bit_idx(bit_str: str) -> int:
    val = int(bit_str)
    idx = 0
    while val > 1:
        val >>= 1
        idx += 1
    return idx

@app.post("/api/valid_moves")
def get_valid_moves(req: GameState):
    global global_cpp_state
    # 🔥 프론트엔드의 가짜 상태(req)는 쳐다보지도 않습니다. 진짜 상태만 씁니다.
    valid_actions = quoridor_engine.get_valid_moves(global_cpp_state)
    
    frontend_mask = [0] * 209 
    is_p1_turn = (global_cpp_state.turn == 0)
    curr_idx = get_bit_idx(global_cpp_state.p0_bits_str if is_p1_turn else global_cpp_state.p1_bits_str)
    
    for action in valid_actions:
        if action < 4: 
            next_s = quoridor_engine.apply_action(global_cpp_state, action)
            new_idx = get_bit_idx(next_s.p0_bits_str if is_p1_turn else next_s.p1_bits_str)
            if 0 <= new_idx <= 80: frontend_mask[new_idx] = 1
        else:
            frontend_idx = action - 4 + 81
            if frontend_idx < 209: frontend_mask[frontend_idx] = 1

    return {"valid_moves": frontend_mask}

@app.post("/api/make_move")
def make_move(req: MoveRequest):
    global global_cpp_state
    frontend_action = req.action
    cpp_action = -1
    
    is_p1_turn = (global_cpp_state.turn == 0)

    valid_actions = quoridor_engine.get_valid_moves(global_cpp_state)

    # 1. 프론트 번호 -> C++ 액션 번호로 통역
    if 0 <= frontend_action <= 80:
        for a in valid_actions:
            if a < 4:
                next_s = quoridor_engine.apply_action(global_cpp_state, a)
                new_idx = get_bit_idx(next_s.p0_bits_str if is_p1_turn else next_s.p1_bits_str)
                if new_idx == frontend_action:
                    cpp_action = a
                    break
    elif 81 <= frontend_action <= 144:
        cpp_action = frontend_action - 81 + 4
    elif 145 <= frontend_action <= 208:
        cpp_action = frontend_action - 145 + 68

    if cpp_action not in valid_actions:
        print(f"🚨 불법 건축물 감지! 차단됨: {cpp_action}")
        return {"error": "Invalid move"}

    # 2. 🔥 진짜 상태를 다음 상태로 덮어씌웁니다! (캐시 완벽 보존)
    global_cpp_state = quoridor_engine.apply_action(global_cpp_state, cpp_action)

    if global_cpp_state.is_jumping:
        print("🦘 말 겹침(Jump) 발생! 서버가 상대방 턴을 자동 PASS 합니다.")
        ACTION_PASS_VALUE = quoridor_engine.ACTION_SIZE - 1 
        global_cpp_state = quoridor_engine.apply_action(global_cpp_state, ACTION_PASS_VALUE)

    # 3. 모든 전이가 끝난 '진짜 최종 턴'을 프론트엔드로 전달
    is_win = quoridor_engine.check_win(global_cpp_state, map_turn_to_cpp(req.current_player))
    
    next_frontend_player = 1 if global_cpp_state.turn == 0 else -1

    return {
        "state": cpp_state_to_dict(global_cpp_state),
        "current_player": next_frontend_player,
        "game_over": is_win,
        "winner": req.current_player if is_win else None
    }

@app.post("/api/ai_move")
def ai_move(req: AIRequest):
    global global_cpp_state
    
    # 🚨 핵심 수정: C++에서 반환하는 튜플을 변수 2개로 쪼개서 받습니다!
    best_cpp_action, probs = ai_engine.get_ai_move(global_cpp_state, req.num_searches)
    
    frontend_action = -1
    is_p1_turn = (global_cpp_state.turn == 0)

    # 2. C++ 번호를 프론트엔드 번호(0 ~ 208)로 통역합니다.
    if best_cpp_action < 4:
        # 💡 주의: 쿼리도르는 점프(Jump)가 있기 때문에 단순 +1, -9 계산보다
        # apply_action으로 다음 상태를 뽑아본 뒤 위치를 찾는 것이 가장 안전합니다.
        next_s = quoridor_engine.apply_action(global_cpp_state, best_cpp_action)
        frontend_action = get_bit_idx(next_s.p0_bits_str if is_p1_turn else next_s.p1_bits_str)
        
    elif 4 <= best_cpp_action <= 67:
        # 가로 벽(H)
        frontend_action = best_cpp_action - 4 + 81
        
    elif 68 <= best_cpp_action <= 131:
        # 세로 벽(V)
        frontend_action = best_cpp_action - 68 + 145
    
    print(frontend_action)

    return {
        "best_action": frontend_action 
    }

@app.on_event("startup")
def load_engine():
    global ai_engine
    print(f"🤖 AI 엔진 로드 중... ({ONNX_PATH})")
    # C++의 QuoridorAIWrapper 객체를 생성합니다.
    ai_engine = quoridor_engine.QuoridorAI(ONNX_PATH)
    print("✅ AI 엔진 로드 완료!")
