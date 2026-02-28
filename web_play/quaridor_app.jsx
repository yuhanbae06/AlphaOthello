import React, { useState, useEffect } from 'react';

const API_URL = 'http://localhost:8000/api';
const SIZE = 9;
const CELL = 54;
const WALL = 10;
const PAD = 8;
const W = PAD * 2 + SIZE * CELL + (SIZE - 1) * WALL;

// --- 유틸 함수 ---
// 파이썬의 bit_length() 역할을 자바스크립트의 BigInt로 구현
const bitToPos = (bit) => {
  if (!bit || bit === 0) return null;
  let pos = 0n;
  let b = BigInt(bit);
  while (b > 1n) {
    b >>= 1n;
    pos++;
  }
  const p = Number(pos);
  return [Math.floor(p / SIZE), p % SIZE];
};

// 특정 위치의 비트가 1인지 확인 (벽 체크용)
const checkBit = (mask, i) => (BigInt(mask) >> BigInt(i)) & 1n === 1n;

export default function App() {
  const [gameState, setGameState] = useState(null);
  const [currentPlayer, setCurrentPlayer] = useState(1);
  const [gameOver, setGameOver] = useState(false);
  const [winner, setWinner] = useState(null);
  const [mode, setMode] = useState('move'); // 'move', 'wall_h', 'wall_v'
  const [validMoves, setValidMoves] = useState([]);
  const [numSearches, setNumSearches] = useState(100);
  const [loadingAI, setLoadingAI] = useState(false);

  // 1. 초기 게임 세팅
  const initSession = async () => {
    try {
      const res = await fetch(`${API_URL}/init`);
      const data = await res.json();
      setGameState(data.state);
      setCurrentPlayer(data.current_player);
      setGameOver(data.game_over);
      setWinner(data.winner);
      setMode('move');
      fetchValidMoves(data.state);
    } catch (error) {
      console.error("초기화 실패:", error);
    }
  };

  useEffect(() => {
    initSession();
  }, []);

  // 2. 가능한 수(Valid Moves) 가져오기
  const fetchValidMoves = async (state) => {
    try {
      const res = await fetch(`${API_URL}/valid_moves`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(state),
      });
      const data = await res.json();
      setValidMoves(data.valid_moves);
    } catch (error) {
      console.error("Valid moves 로드 실패:", error);
    }
  };

  // 3. 수 두기 (사람 액션)
  const makeMove = async (action, isAI = false) => {
    // 사람의 턴일 때만 유효성 검사 수행 (AI가 보낸 수는 이미 백엔드에서 검증됨)
    if (!isAI && validMoves[action] === 0) return;
    try {
      const res = await fetch(`${API_URL}/make_move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          state: gameState,
          current_player: currentPlayer,
          action: action
        }),
      });
      const data = await res.json();
      setGameState(data.state);
      setCurrentPlayer(data.current_player);
      setGameOver(data.game_over);
      setWinner(data.winner);
      setMode('move');
      if (!data.game_over) fetchValidMoves(data.state);
    } catch (error) {
      console.error("Move 실패:", error);
    }
  };

  // 4. AI 수 실행
  const runAI = async () => {
    setLoadingAI(true);
    try {
      const res = await fetch(`${API_URL}/ai_move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ state: gameState, num_searches: numSearches }),
      });
      const data = await res.json();

      await makeMove(data.best_action, true); // AI가 결정한 수로 보드 업데이트
    } catch (error) {
      console.error("AI 실행 실패:", error);
    } finally {
      setLoadingAI(false);
    }
  };

  // --- 보드 렌더링 로직 ---
  const renderBoard = () => {
    if (!gameState) return null;

    const p1 = bitToPos(gameState.p_bits[0]);
    const p2 = bitToPos(gameState.p_bits[1]);

    const cx = (c) => PAD + c * (CELL + WALL);
    const cy = (dr) => PAD + dr * (CELL + WALL);
    const getDispRow = (r) => SIZE - 1 - r; // 화면 출력용 Row

    const elements = [];

    // 셀 렌더링
    for (let r = 0; r < SIZE; r++) {
      for (let c = 0; c < SIZE; c++) {
        const idx = r * SIZE + c;
        const x = cx(c);
        const y = cy(getDispRow(r));
        const isValid = validMoves[idx] === 1 && mode === 'move' && currentPlayer === 1;

        // 배경색 설정
        const bg = isValid ? '#4ade80' : '#c8934a';
        const shadow = isValid ? '0 0 10px rgba(74,222,128,.7)' : '0 2px 4px rgba(0,0,0,.2)';

        elements.push(
          <div
            key={`cell_${r}_${c}`}
            onClick={() => isValid && makeMove(idx)}
            style={{
              position: 'absolute', left: x, top: y, width: CELL, height: CELL,
              background: bg, borderRadius: 6, boxShadow: shadow,
              cursor: isValid ? 'pointer' : 'default'
            }}
          >
            {/* 플레이어 1 (나) */}
            {p1 && p1[0] === r && p1[1] === c && (
              <div style={{
                position: 'absolute', left: 6, top: 6, width: CELL - 12, height: CELL - 12,
                background: 'radial-gradient(circle at 35% 35%,#93c5fd,#1d4ed8)',
                borderRadius: '50%', border: '2.5px solid #1e3a8a',
                boxShadow: '0 4px 10px rgba(0,0,0,.4)'
              }} />
            )}
            {/* 플레이어 -1 (AI) */}
            {p2 && p2[0] === r && p2[1] === c && (
              <div style={{
                position: 'absolute', left: 6, top: 6, width: CELL - 12, height: CELL - 12,
                background: 'radial-gradient(circle at 35% 35%,#fca5a5,#b91c1c)',
                borderRadius: '50%', border: '2.5px solid #7f1d1d',
                boxShadow: '0 4px 10px rgba(0,0,0,.4)'
              }} />
            )}
          </div>
        );
      }
    }

    // 가로 벽 (H) 렌더링
    const wallsHSize = (SIZE - 1) * (SIZE - 1);
    for (let i = 0; i < wallsHSize; i++) {
      if (checkBit(gameState.walls_h, i)) {
        const wr = Math.floor(i / (SIZE - 1));
        const wc = i % (SIZE - 1);
        const x = cx(wc);
        const y = cy(getDispRow(wr + 1)) + CELL;
        elements.push(
          <div key={`hw_${i}`} style={{
            position: 'absolute', left: x, top: y, width: CELL * 2 + WALL, height: WALL,
            background: '#7c3f1e', borderRadius: 3, zIndex: 10,
            boxShadow: '0 2px 6px rgba(0,0,0,.5)'
          }} />
        );
      }
    }

    // 세로 벽 (V) 렌더링
    for (let i = 0; i < wallsHSize; i++) {
      if (checkBit(gameState.walls_v, i)) {
        const wr = Math.floor(i / (SIZE - 1));
        const wc = i % (SIZE - 1);
        const x = cx(wc) + CELL;
        const y = cy(getDispRow(wr));
        elements.push(
          <div key={`vw_${i}`} style={{
            position: 'absolute', left: x, top: y, width: WALL, height: CELL * 2 + WALL,
            background: '#7c3f1e', borderRadius: 3, zIndex: 10,
            boxShadow: '2px 0 6px rgba(0,0,0,.5)'
          }} />
        );
      }
    }

    return (
      <div style={{
        position: 'relative', width: W, height: W, background: '#1a6b1a',
        borderRadius: 12, boxShadow: '0 8px 25px rgba(0,0,0,.4)', margin: '0 auto'
      }}>
        {elements}
      </div>
    );
  };

  if (!gameState) return <div style={{ textAlign: 'center', padding: 50 }}>게임을 불러오는 중...</div>;

  const pieceActionSize = SIZE * SIZE;
  const wallsActionSize = (SIZE - 1) * (SIZE - 1);

  // --- 화면 레이아웃 ---
  return (
    <div style={{ display: 'flex', justifyContent: 'center', gap: '40px', fontFamily: 'sans-serif', padding: '20px' }}>
      
      {/* ── 왼쪽 패널 ── */}
      <div style={{ width: '250px' }}>
        <h3>🔵 나 (플레이어)</h3>
        <p>남은 벽: <strong>{gameState.walls_left[0]}개</strong></p>
        <hr />
        <h4>⚙️ 설정</h4>
        <label>AI 탐색 횟수: {numSearches}</label>
        <input 
          type="range" min="10" max="300" step="10" 
          value={numSearches} onChange={(e) => setNumSearches(Number(e.target.value))} 
          style={{ width: '100%' }}
        />
        <button onClick={initSession} style={{ width: '100%', padding: '10px', marginTop: '10px' }}>
          🔄 새 게임
        </button>
      </div>

      {/* ── 중앙 패널 (보드) ── */}
      <div style={{ textAlign: 'center' }}>
        <h2>🎮 쿼리도 vs AI</h2>
        
        <div style={{ marginBottom: '15px' }}>
          {gameOver ? (
            <h3 style={{ color: winner === 1 ? 'green' : 'red' }}>
              {winner === 1 ? '🎉 승리! 축하해요!' : '😢 AI가 이겼어요!'}
            </h3>
          ) : currentPlayer === 1 ? (
            <h3 style={{ color: 'blue' }}>🔵 내 차례 — {mode === 'move' ? '이동 모드' : '벽 설치 모드'}</h3>
          ) : (
            <h3 style={{ color: 'red' }}>🔴 AI 차례</h3>
          )}
        </div>

        {/* 게임 보드 */}
        {renderBoard()}

        {/* 조작 버튼 영역 */}
        {!gameOver && currentPlayer === 1 && (
          <div style={{ marginTop: '20px' }}>
            <button onClick={() => setMode('move')} style={{ marginRight: '10px', padding: '10px' }}>
              {mode === 'move' ? '✅ ' : ''}🚶 이동 모드
            </button>
            <button onClick={() => setMode('wall_h')} style={{ marginRight: '10px', padding: '10px' }}>
              {mode === 'wall_h' ? '✅ ' : ''}🧱 가로 벽
            </button>
            <button onClick={() => setMode('wall_v')} style={{ padding: '10px' }}>
              {mode === 'wall_v' ? '✅ ' : ''}🧱 세로 벽
            </button>

            {/* 벽 설치 인터페이스 (간단한 드롭다운이나 버튼 리스트로 구현 가능) */}
            {mode !== 'move' && (
              <div style={{ marginTop: '15px', display: 'flex', flexWrap: 'wrap', width: W, gap: '5px' }}>
                <p style={{ width: '100%', fontSize: '12px' }}>클릭하여 {mode === 'wall_h' ? '가로' : '세로'} 벽을 설치하세요 (버튼이 활성화된 곳만 가능)</p>
                {Array.from({ length: wallsActionSize }).map((_, i) => {
                  const actionIdx = mode === 'wall_h' ? pieceActionSize + i : pieceActionSize + wallsActionSize + i;
                  const isValid = validMoves[actionIdx] === 1;
                  return (
                    <button 
                      key={i} 
                      disabled={!isValid}
                      onClick={() => makeMove(actionIdx)}
                      style={{ padding: '5px', fontSize: '10px' }}
                    >
                      ({Math.floor(i/(SIZE-1))},{i%(SIZE-1)})
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {!gameOver && currentPlayer === -1 && (
          <div style={{ marginTop: '20px' }}>
            <button onClick={runAI} disabled={loadingAI} style={{ padding: '10px 20px', background: '#dc2626', color: 'white', border: 'none', borderRadius: '5px' }}>
              {loadingAI ? '🤖 AI 계산 중...' : '🤖 AI 수 실행'}
            </button>
          </div>
        )}
      </div>

      {/* ── 오른쪽 패널 ── */}
      <div style={{ width: '250px' }}>
        <h3>🔴 AI</h3>
        <p>남은 벽: <strong>{gameState.walls_left[1]}개</strong></p>
        <hr />
        <h4>🧠 AI 분석</h4>
        <p style={{ fontSize: '12px', color: 'gray' }}>FastAPI 연동을 완료하면 이 곳에 AI의 승률 및 예측 확률 등을 표시할 수 있습니다.</p>
      </div>

    </div>
  );
}