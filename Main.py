from Game import *
from NeuralNet import *
from Node import *
from AlphaZero import *
from AlphaZeroParallel import *
from Args import *
from utils import *

import numpy as np
print(np.__version__)


import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

from tqdm import trange

import matplotlib.pyplot as plt
import argparse
import random
import math
import ray
import os

game_dict = {
    "tictactoe": TicTacToe(),
    "connectfour": ConnectFour(),
    "othello": Othello()
}


def model_test(game_name):
    game = game_dict[game_name]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state = game.get_initial_state()
    state = game.get_next_state(state, 13, 1)
    
    
    encoded_state = game.get_encoded_state(state)
    
    tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)
    
    model = ResNet(game, 4, 64, device=device)
    # model.load_state_dict(torch.load('model_2_TicTacToe.pt', map_location=device))
    model.eval()
    
    policy, value = model(tensor_state)
    value = value.item()
    policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()
    
    print(value)
    
    print(state)
    print(tensor_state)
    
    plt.bar(range(game.action_size), policy)
    plt.show()


def model_learn(game_name, config_name):
    game = game_dict[game_name]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ResNet(game, 4, 64, device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    # args = get_args(config_name).dict_()
    args = load_config(f"./configs/{config_name}.yaml")

    context = ray.init(runtime_env={"py_modules": ["AlphaZeroParallel.py"]})
    print(context.dashboard_url)

    # alphaZero = AlphaZeroParallel(model, optimizer, game, args, True)
    alphaZero = AlphaZeroParallelRay(model, optimizer, game, args, True)
    alphaZero.learn()

    ray.shutdown()


def model_play(game_name, version):
    game = game_dict[game_name]
    player = 1
    
    args = {
        'C': 2,
        'num_searches': 1000,
        'dirichlet_epsilon': 0.,
        'dirichlet_alpha': 0.3
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ResNet(game, 4, 64, device)
    model.load_state_dict(torch.load("./saved_model/model_{0}_{1}.pt".format(version, game.__repr__()), map_location=device))
    model.eval()
    
    mcts = MCTS(game, args, model)
    
    state = game.get_initial_state()
    
    
    while True:
        print(state)
        
        if player == 1:
            valid_moves = game.get_valid_moves(state)
            print("valid_moves", [i for i in range(game.action_size) if valid_moves[i] == 1])
            action = int(input(f"{player}:"))
    
            if valid_moves[action] == 0:
                print("action not valid")
                continue
                
        else:
            neutral_state = game.change_perspective(state, player)
            print(game.get_valid_moves(neutral_state))
            mcts_probs = mcts.search(neutral_state)
            action = np.argmax(mcts_probs)
            
        state = game.get_next_state(state, action, player)
        
        value, is_terminal = game.get_value_and_terminated(game.change_perspective(state, player), action)
        
        if is_terminal:
            print(state)
            if value == 1:
                print(player, "won")
            elif value == -1:
                print(game.get_opponent(player), "won")
            else:
                print("draw")
            break
            
        player = game.get_opponent(player)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")
    # --- 'test' mode ---
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--game", type=str, default="othello")
    # --- 'learn' mode ---
    learn_parser = subparsers.add_parser("learn")
    learn_parser.add_argument("--game", type=str, default="othello")
    learn_parser.add_argument("--config", type=str, default="exp0")
    # --- 'play' mode ---
    play_parser = subparsers.add_parser("play")
    play_parser.add_argument("--game", type=str, default="othello")
    play_parser.add_argument("--version", type=str, default="0")

    args = parser.parse_args()

    if args.mode == "test":
        model_test(args.game)
    elif args.mode == "learn":
        model_learn(args.game, args.config)
    elif args.mode == "play":
        model_play(args.game, args.version)