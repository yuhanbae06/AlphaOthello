import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch

from AlphaZero import MCTS
from AlphaZeroParallel import AlphaZeroParallel
from Game import make_game
from NeuralNet import ResNet
from utils import load_config


torch.manual_seed(0)


def _is_rank0_process():
    return int(os.environ.get("RANK", "0")) == 0


def _infer_last_iteration_from_model_path(model_path):
    filename = os.path.basename(model_path)
    match = re.search(r"^model_(\d+)_.*\.pt$", filename)
    if match is None:
        raise ValueError(
            f"Could not infer iteration from model filename '{filename}'. "
            "Expected format like model_43_Quoridor.pt"
        )
    return int(match.group(1))


def _default_optimizer_path_from_model_path(model_path):
    dirname = os.path.dirname(model_path)
    filename = os.path.basename(model_path)
    optimizer_name = re.sub(r"^model_", "optimizer_", filename)
    return os.path.join(dirname, optimizer_name)


def _action_to_text(game, action):
    action = int(action)
    game_name = str(getattr(game, "game_name", "")).lower()

    if game_name == "gomoku":
        r = action // game.column_count
        c = action % game.column_count
        return f"place stone at (row={r}, col={c})"

    if game_name.startswith("quoridor"):
        if action == getattr(game, "action_pass", -1):
            return "pass (jump phase)"
        if action < 4:
            return ("move up", "move down", "move left", "move right")[action]

        inner_wall_cnt = int(getattr(game, "inner_wall_cnt", 0))
        inner_wall = int(getattr(game, "inner_wall", 0))
        if inner_wall_cnt > 0 and inner_wall > 0:
            if action < 4 + inner_wall_cnt:
                idx = action - 4
                r = idx // inner_wall
                c = idx % inner_wall
                return f"place horizontal wall at inner(r={r + 1}, c={c + 1})"
            if action < 4 + 2 * inner_wall_cnt:
                idx = action - (4 + inner_wall_cnt)
                r = idx // inner_wall
                c = idx % inner_wall
                return f"place vertical wall at inner(r={r + 1}, c={c + 1})"

    return f"action {action}"


def _show_visualized_state(game, state, player, turn_idx):
    try:
        vis = game.get_visualized_state(state)
        if vis.ndim == 3 and vis.shape[0] in (1, 3):
            image = np.transpose(vis, (1, 2, 0))
        else:
            image = vis
        plt.figure("AlphaZero Play", figsize=(6, 6))
        plt.clf()
        plt.imshow(np.clip(image, 0.0, 1.0), interpolation="nearest")
        plt.axis("off")
        plt.title(f"{game} | turn={turn_idx} | player={player}")
        plt.pause(0.001)
    except Exception as e:
        print(f"[warn] visualization failed: {e}")


def _ask_human_action(game, valid_moves, player):
    valid_actions = [i for i in range(game.action_size) if valid_moves[i] == 1]
    if not valid_actions:
        raise RuntimeError("No valid actions available")

    print(f"\nplayer {player} valid actions:")
    for idx, action in enumerate(valid_actions):
        print(f"  [{idx}] {_action_to_text(game, action)} (id={action})")

    while True:
        raw = input("choose index or action id: ").strip()
        if raw == "":
            print("empty input")
            continue
        try:
            choice = int(raw)
        except ValueError:
            print("enter an integer")
            continue

        if 0 <= choice < len(valid_actions):
            return valid_actions[choice]
        if choice in valid_actions:
            return choice
        print("invalid choice")


def model_test():
    game = make_game("gomoku")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = game.get_initial_state()
    state = game.get_next_state(state, 44, 1)
    encoded_state = game.get_encoded_state(state)
    tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)

    model = ResNet(game, 4, 64, device=device, input_channels=game.input_channels)
    model.eval()

    policy, value = model(tensor_state)
    value = value.item()
    policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

    print("value:", value)
    print("state:\n", state)

    plt.bar(range(game.action_size), policy)
    plt.title("Policy Distribution (Gomoku10)")
    plt.show()


def model_learn(config_name, resume_model=None, resume_optimizer=None, resume_iter=None):
    args = load_config(f"./configs/learn/{config_name}.yaml")
    game_name = args.get("game", "gomoku")
    game = make_game(game_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_stem = os.path.splitext(os.path.basename(str(config_name)))[0]
    log_root = str(args.get("log_root", "logs"))
    args["log_dir"] = os.path.join(log_root, str(game_name), config_stem)
    args["config_name"] = config_stem
    if _is_rank0_process():
        print(f"[learn] tensorboard log_dir: {args['log_dir']}")

    model = ResNet(game, 4, 64, device, input_channels=game.input_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    args["device"] = str(device)
    args["game"] = game_name

    if resume_model:
        if not os.path.exists(resume_model):
            raise FileNotFoundError(f"resume model not found: {resume_model}")

        model.load_state_dict(torch.load(resume_model, map_location=device))

        if resume_iter is None:
            last_iteration = _infer_last_iteration_from_model_path(resume_model)
        else:
            last_iteration = int(resume_iter)
        args["start_iteration"] = last_iteration + 1

        optimizer_path = resume_optimizer
        if optimizer_path is None:
            optimizer_path = _default_optimizer_path_from_model_path(resume_model)
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
            print(f"[learn] resumed optimizer from: {optimizer_path}")
        else:
            print(f"[learn] optimizer checkpoint not found, using fresh optimizer: {optimizer_path}")

        print(
            f"[learn] resumed model from: {resume_model} "
            f"(last_iteration={last_iteration}, start_iteration={args['start_iteration']})"
        )

    trainer = AlphaZeroParallel(model, optimizer, game, args, monitor=True)
    trainer.learn()


def model_train_ddp(config_name, model_path, optimizer_path, iteration):
    args = load_config(f"./configs/learn/{config_name}.yaml")
    game_name = args.get("game", "gomoku")
    game = make_game(game_name)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    config_stem = os.path.splitext(os.path.basename(str(config_name)))[0]
    log_root = str(args.get("log_root", "logs"))
    args["log_dir"] = os.path.join(log_root, str(game_name), config_stem)
    args["config_name"] = config_stem
    args["ddp_train_only"] = True

    model = ResNet(game, 4, 64, device, input_channels=game.input_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    model.load_state_dict(torch.load(model_path, map_location=device))
    optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))

    trainer = AlphaZeroParallel(model, optimizer, game, args, monitor=False)
    trainer.train_iteration_only(int(iteration), model_path, optimizer_path)


def _load_model_for_play(game, device, version):
    model = ResNet(game, 4, 64, device, input_channels=game.input_channels)
    model_path = f"./saved_model/model_{version}_{game.__repr__()}.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, model_path


def model_play(version, config_name="play0", human_player=1, versus_version=None):
    args = load_config(f"./configs/play/{config_name}.yaml")
    game = make_game(args.get("game", "gomoku"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_a, model_a_path = _load_model_for_play(game, device, version)
    mcts_a = MCTS(game, args, model_a)

    ai_vs_ai = versus_version is not None
    mcts_b = None
    model_b_path = None
    if ai_vs_ai:
        if str(versus_version) == str(version):
            mcts_b = mcts_a
            model_b_path = model_a_path
        else:
            model_b, model_b_path = _load_model_for_play(game, device, versus_version)
            mcts_b = MCTS(game, args, model_b)
        print(f"[play] AI vs AI enabled: P1={model_a_path}, P2={model_b_path}")
    else:
        print(f"[play] Human vs AI: AI model={model_a_path}, human_player={human_player}")

    state = game.get_initial_state()
    player = 1
    turn_idx = 0
    plt.ion()

    while True:
        _show_visualized_state(game, state, player, turn_idx)
        if (not ai_vs_ai) and (player == human_player):
            valid_moves = game.get_valid_moves(state)
            action = _ask_human_action(game, valid_moves, player)
        else:
            neutral_state = game.change_perspective(state, player)
            mcts = mcts_a if player == 1 else (mcts_b if mcts_b is not None else mcts_a)
            mcts_probs = mcts.search(neutral_state)
            action = int(np.argmax(mcts_probs))
            action = game.action_from_canonical(action, player) if hasattr(game, "action_from_canonical") else action
            print(f"ai chooses: {_action_to_text(game, action)} (id={action})")

        state = game.get_next_state(state, action, player)
        value, is_terminal = game.get_value_and_terminated(game.change_perspective(state, player), action)
        turn_idx += 1
        if is_terminal:
            _show_visualized_state(game, state, player, turn_idx)
            if value == 1:
                print(player, "won")
            elif value == -1:
                print(game.get_opponent(player), "won")
            else:
                print("draw")
            break
        player = game.get_opponent(player)
    plt.ioff()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    test_parser = subparsers.add_parser("test")
    learn_parser = subparsers.add_parser("learn")
    learn_parser.add_argument("--config", type=str, default="exp0")
    learn_parser.add_argument(
        "--resume-model",
        type=str,
        default=None,
        help="Path to existing model checkpoint (e.g. ./saved_model/model_43_Quoridor.pt)",
    )
    learn_parser.add_argument(
        "--resume-optimizer",
        type=str,
        default=None,
        help="Optional optimizer checkpoint path. Defaults to matching optimizer_<iter>_*.pt",
    )
    learn_parser.add_argument(
        "--resume-iter",
        type=int,
        default=None,
        help="Override last completed iteration index (default: inferred from --resume-model filename)",
    )
    train_ddp_parser = subparsers.add_parser("train-ddp", help=argparse.SUPPRESS)
    train_ddp_parser.add_argument("--config", type=str, required=True)
    train_ddp_parser.add_argument("--model-path", type=str, required=True)
    train_ddp_parser.add_argument("--optimizer-path", type=str, required=True)
    train_ddp_parser.add_argument("--iteration", type=int, required=True)
    play_parser = subparsers.add_parser("play")
    play_parser.add_argument("--version", type=str, default="0")
    play_parser.add_argument("--config", type=str, default="play0")
    play_parser.add_argument("--human-player", type=int, default=1)
    play_parser.add_argument(
        "--versus-version",
        type=str,
        default=None,
        help="If set, run AI vs AI. --version is player 1 model, --versus-version is player 2 model.",
    )

    args = parser.parse_args()

    if args.mode == "test":
        model_test()
    elif args.mode == "learn":
        model_learn(args.config, args.resume_model, args.resume_optimizer, args.resume_iter)
    elif args.mode == "train-ddp":
        model_train_ddp(args.config, args.model_path, args.optimizer_path, args.iteration)
    elif args.mode == "play":
        model_play(args.version, args.config, args.human_player, args.versus_version)
