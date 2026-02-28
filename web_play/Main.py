from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union, Optional
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import sys
import torch
import importlib
from to_onnx import pt_to_onnx

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

# === Initialize ===

BOARD_SIZE = int(os.getenv("VITE_BOARD_SIZE", 9))
PT_PATH = os.path.join(os.getenv("MODEL_PATH"), "model_0.pt")
ONNX_PATH = os.path.join(os.getenv("MODEL_PATH"), f'model_{BOARD_SIZE}.onnx')

module_name = f"quoridor_engine_{BOARD_SIZE}"

try:
    # 4. 동적 임포트 실행
    quoridor_engine = importlib.import_module(module_name)
    print(f"✅ 성공적으로 {module_name} 모듈을 로드했습니다.")
except ImportError:
    # 에러 핸들링: 만약 빌드를 안 했다면 알림
    print(f"❌ 에러: {module_name} 모듈을 찾을 수 없습니다. 빌드를 확인하세요.")
    # 기본 엔진으로 폴백하거나 서버 실행을 중단
    raise ImportError(f"Please build the engine for size {BOARD_SIZE} first.")

from NeuralNet import ResNet
from Game import Quoridor5, Quoridor7, Quoridor9

if BOARD_SIZE == 5:
    game = Quoridor5()
elif BOARD_SIZE == 7:
    game = Quoridor7()
elif BOARD_SIZE == 9:
    game = Quoridor9()
else: print(f"Unsupported BOARD_SIZE: {BOARD_SIZE}.")

if not os.path.exists(ONNX_PATH):
    pt_to_onnx(game=game, model_path=PT_PATH, onnx_output_path=ONNX_PATH)

ai_engine = quoridor_engine.QuoridorAI(ONNX_PATH)

app = FastAPI(title="Quoridor API Server (C++ Engine Powered)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic Model ─
class GameState(BaseModel):
    p_bits: List[Union[int, str]]
    walls_h: Union[int, str]
    walls_v: Union[int, str]
    walls_left: List[int]
    turn: int
    is_jumping: bool = False
    jump_dir: int = -1
    jumper_idx: Optional[int] = 0
    h_block: int | str = "0"
    v_block: int | str = "0"

class MoveRequest(BaseModel):
    state: GameState
    current_player: int  # 1 - Player 1, -1 - Player 2
    action: int

class AIRequest(BaseModel):
    state: GameState
    num_searches: int = 100

# === utilties ===

def map_turn_to_cpp(frontend_player: int) -> int:
    """프론트의 1/-1을 C++ 엔진의 0/1로 변환"""
    return 0 if frontend_player == 1 else 1

def map_turn_to_frontend(cpp_turn: int) -> int:
    """C++ 엔진의 0/1을 프론트의 1/-1로 변환"""
    return 1 if cpp_turn == 0 else -1

def dict_to_cpp_state(req: GameState, current_player: int):
    cpp_state = quoridor_engine.get_initial_state()
            
    cpp_state.p0_bits_str = str(req.p_bits[0])
    cpp_state.p1_bits_str = str(req.p_bits[1])
    cpp_state.walls_h_str = str(req.walls_h)
    cpp_state.walls_v_str = str(req.walls_v)
    cpp_state.walls_left = [int(req.walls_left[0]), int(req.walls_left[1])]
    
    cpp_state.turn = 0 if current_player == 1 else 1

    cpp_state.is_jumping = req.is_jumping
    if hasattr(cpp_state, 'jump_dir'):
        cpp_state.jump_dir = req.jump_dir

    cpp_state.jumper_idx = -1 if req.jumper_idx == None else map_turn_to_cpp(req.jumper_idx)

    cpp_state.h_block_str = str(req.h_block)
    cpp_state.v_block_str = str(req.v_block)

    return cpp_state

def cpp_state_to_dict(cpp_state) -> dict:
    return {
        "p_bits": [cpp_state.p0_bits_str, cpp_state.p1_bits_str],
        "walls_h": cpp_state.walls_h_str,
        "walls_v": cpp_state.walls_v_str,
        "walls_left": cpp_state.walls_left,
        "turn": map_turn_to_frontend(cpp_state.turn),
        "is_jumping": cpp_state.is_jumping,
        "jump_dir": cpp_state.jump_dir,
        "jumper_idx": map_turn_to_frontend(cpp_state.jumper_idx),
        "h_block": cpp_state.h_block_str,
        "v_block": cpp_state.v_block_str
    }


# === Game Variables ===

SIZE = quoridor_engine.SIZE
NUM_SQUARES = BOARD_SIZE ** 2
WALL_COUNT = (BOARD_SIZE - 1) ** 2
ACTION_SIZE = quoridor_engine.ACTION_SIZE
WALL_ACTION_SIZE = (SIZE - 1) ** 2
FE_MAX_ACTION = NUM_SQUARES + (2 * WALL_ACTION_SIZE)
ACTION_PASS_VALUE = quoridor_engine.ACTION_SIZE - 1 

# ══════════════ API 엔드포인트 ═══════════════════════════════

@app.get("/api/init")
def init_game():
    cpp_state = quoridor_engine.get_initial_state()
    return {
        "state": cpp_state_to_dict(cpp_state),
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
    cpp_state = dict_to_cpp_state(req, req.turn)
    valid_actions = quoridor_engine.get_valid_moves(cpp_state)
    
    if getattr(cpp_state, 'is_jumping', False):
        opp_dir = {0: 1, 1: 0, 2: 3, 3: 2}.get(getattr(cpp_state, 'jump_dir', -1), -1)
        valid_actions = [a for a in valid_actions if a != opp_dir]

    frontend_mask = [0] * FE_MAX_ACTION
    is_p1_turn = (cpp_state.turn == 0)
    
    for action in valid_actions:
        if action < 4: 
            next_s = quoridor_engine.apply_action(cpp_state, action)
            new_idx = get_bit_idx(next_s.p0_bits_str if is_p1_turn else next_s.p1_bits_str)
            if 0 <= new_idx <= (NUM_SQUARES - 1) : frontend_mask[new_idx] = 1
        else:
            frontend_idx = action - 4 + NUM_SQUARES
            if frontend_idx < FE_MAX_ACTION: frontend_mask[frontend_idx] = 1
    
    print(cpp_state.turn, valid_actions)

    return {"valid_moves": frontend_mask}

@app.post("/api/make_move")
def make_move(req: MoveRequest):
    cpp_state = dict_to_cpp_state(req.state, req.state.turn)
    frontend_action = req.action
    cpp_action = -1
    
    is_p1_turn = (cpp_state.turn == 0)

    valid_actions = quoridor_engine.get_valid_moves(cpp_state)

    if getattr(cpp_state, 'is_jumping', False):
        opp_dir = {0: 1, 1: 0, 2: 3, 3: 2}.get(getattr(cpp_state, 'jump_dir', -1), -1)
        valid_actions = [a for a in valid_actions if a != opp_dir]

    if 0 <= frontend_action < NUM_SQUARES:
        for a in valid_actions:
            if a < 4:
                next_s = quoridor_engine.apply_action(cpp_state, a)
                new_idx = get_bit_idx(next_s.p0_bits_str if is_p1_turn else next_s.p1_bits_str)
                if new_idx == frontend_action:
                    cpp_action = a
                    break
    elif NUM_SQUARES <= frontend_action < FE_MAX_ACTION:
        cpp_action = frontend_action - NUM_SQUARES + 4

    if cpp_action not in valid_actions:
        print(f"Illegal Move: {cpp_action}")
        return {"error": "Invalid move"}

    cpp_state = quoridor_engine.apply_action(cpp_state, cpp_action)
    next_valid = quoridor_engine.get_valid_moves(cpp_state)

    if len(next_valid) == 1 and next_valid[0] == ACTION_PASS_VALUE:
        cpp_state = quoridor_engine.apply_action(cpp_state, ACTION_PASS_VALUE)

    is_win = quoridor_engine.check_win(cpp_state, map_turn_to_cpp(req.current_player))
    
    next_frontend_player = 1 if cpp_state.turn == 0 else -1

    return {
        "state": cpp_state_to_dict(cpp_state),
        "current_player": next_frontend_player,
        "game_over": is_win,
        "winner": req.current_player if is_win else None
    }

@app.post("/api/ai_move")
def ai_move(req: AIRequest):
    cpp_state = dict_to_cpp_state(req.state, req.state.turn)
    
    best_cpp_action, _ = ai_engine.get_ai_move(cpp_state, req.num_searches)
    
    frontend_action = -1
    is_p1_turn = (cpp_state.turn == 0)

    if best_cpp_action < 4:
        next_s = quoridor_engine.apply_action(cpp_state, best_cpp_action)
        frontend_action = get_bit_idx(next_s.p0_bits_str if is_p1_turn else next_s.p1_bits_str)
        
    elif 4 <= best_cpp_action < WALL_ACTION_SIZE + 4:
        frontend_action = best_cpp_action - 4 + NUM_SQUARES
        
    elif WALL_ACTION_SIZE + 4 <= best_cpp_action < 2 * WALL_ACTION_SIZE + 4:
        frontend_action = best_cpp_action - 4 - WALL_ACTION_SIZE + (WALL_ACTION_SIZE + NUM_SQUARES)
    
    return {
        "best_action": frontend_action 
    }
