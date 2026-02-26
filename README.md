# AlphaOthello

AlphaZero algorithm to play board games.

## 1. Command Line Interface (CLI) 사용법

이 프로젝트는 하나의 진입 스크립트에서 **subcommand 방식**으로 실행됩니다.

```bash
python Main.py <mode> [options]
```

- `<mode>`: 실행 모드 (`test`, `learn`, `play` 중 하나)
- `[options]`: 각 모드별 추가 옵션

---

### 1.1 test 모드

모델의 기본 동작 또는 테스트용 로직을 실행합니다.

```bash
python Main.py test --game othello
```

옵션:

- `--game` (str, default: `othello`)
  - 사용할 게임 환경 이름

실행되는 함수:

```python
model_test(game)
```

---

### 1.2 learn 모드

모델 학습을 수행합니다. 실험 설정(config)을 함께 지정할 수 있습니다.

```bash
python Main.py learn --game othello --config exp0
```

옵션:

- `--game` (str, default: `othello`)
  - 학습에 사용할 게임

- `--config` (str, default: `exp0`)
  - 학습 설정 이름 (예: YAML config 파일 이름)

실행되는 함수:

```python
model_learn(game, config)
```

#### 기존 체크포인트에서 학습 재개

이미 학습된 모델 파일(예: `model_42_Quoridor.pt`)이 있으면 해당 지점 다음
iteration부터 이어서 학습할 수 있습니다.

```bash
python Main.py learn --config exp2 \
  --resume-model ./saved_model/model_42_Quoridor.pt
```

옵션:

- `--resume-model` (str):
  재개할 모델 체크포인트 경로
- `--resume-optimizer` (str, optional):
  옵티마이저 체크포인트 경로. 지정하지 않으면
  `--resume-model` 경로에서 `optimizer_<iter>_*.pt` 형식으로 자동 추정
- `--resume-iter` (int, optional):
  마지막 완료 iteration 인덱스를 수동 지정

명시적으로 지정하는 예시:

```bash
python Main.py learn --config exp2 \
  --resume-model ./saved_model/model_42_Quoridor.pt \
  --resume-optimizer ./saved_model/optimizer_42_Quoridor.pt \
  --resume-iter 42
```

주의사항:

- 재개 시작 지점은 `start_iteration = last_iteration + 1` 입니다.
- config의 `num_iterations`는 재개 시작 iteration보다 커야 합니다.

---

### 1.3 play 모드

학습된 모델을 사용하여 실제 플레이를 수행합니다.

```bash
python Main.py play --game othello --version 0
```

옵션:

- `--game` (str, default: `othello`)
  - 플레이할 게임

- `--version` (str, default: `0`)
  - 사용할 모델 버전

실행되는 함수:

```python
model_play(game, version)
```

---

### 1.4 전체 명령어 요약

```text
python Main.py test   --game <game_name>
python Main.py learn  --game <game_name> --config <config_name>
python Main.py play   --game <game_name> --version <model_version>
```

---
