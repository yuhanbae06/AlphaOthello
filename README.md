# AlphaOthello
Othello AI with AlphaZero algorithm

## 1. Command Line Interface (CLI) 사용법

이 프로젝트는 하나의 진입 스크립트에서 **subcommand 방식**으로 실행됩니다.

```bash
python Main.py <mode> [options]
```

* `<mode>`: 실행 모드 (`test`, `learn`, `play` 중 하나)
* `[options]`: 각 모드별 추가 옵션

---

### 1.1 test 모드

모델의 기본 동작 또는 테스트용 로직을 실행합니다.

```bash
python Main.py test --game othello
```

옵션:

* `--game` (str, default: `othello`)

  * 사용할 게임 환경 이름

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

* `--game` (str, default: `othello`)

  * 학습에 사용할 게임
* `--config` (str, default: `exp0`)

  * 학습 설정 이름 (예: YAML config 파일 이름)

실행되는 함수:

```python
model_learn(game, config)
```

---

### 1.3 play 모드

학습된 모델을 사용하여 실제 플레이를 수행합니다.

```bash
python Main.py play --game othello --version 0
```

옵션:

* `--game` (str, default: `othello`)

  * 플레이할 게임
* `--version` (str, default: `0`)

  * 사용할 모델 버전

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
