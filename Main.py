# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: alphaothello
#     language: python
#     name: alphaothello
# ---

# +
import import_ipynb
from Game import TicTacToe, ConnectFour
from NeuralNet import ResNet
from Node import Node
from AlphaZero import MCTS, AlphaZero
from AlphaZeroParallel import MCTSParallel, AlphaZeroParallel, AlphaZeroParallelRay
from Args import *

import numpy as np
print(np.__version__)


import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

from tqdm.notebook import trange

import matplotlib.pyplot as plt
import random
import math
import ray


# -

def model_test():
    tictactoe = TicTacToe()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state = tictactoe.get_initial_state()
    state = tictactoe.get_next_state(state, 4, -1)
    
    
    encoded_state = tictactoe.get_encoded_state(state)
    
    tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)
    
    model = ResNet(tictactoe, 4, 64, device=device)
    # model.load_state_dict(torch.load('model_2_TicTacToe.pt', map_location=device))
    model.eval()
    
    policy, value = model(tensor_state)
    value = value.item()
    policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()
    
    print(value)
    
    print(state)
    print(tensor_state)
    
    plt.bar(range(tictactoe.action_size), policy)
    plt.show()


def model_learn():
    game = TicTacToe()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ResNet(game, 4, 64, device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    
    # args = {
    #     'C': 2,
    #     'num_searches': 60,
    #     'num_iterations': 3,
    #     'num_selfPlay_iterations': 500,
    #     'num_parallel_games': 100,
    #     'num_epochs': 4,
    #     'batch_size': 64,
    #     'temperature': 1.25,
    #     'dirichlet_epsilon': 0.25,
    #     'dirichlet_alpha': 0.3
    # }

    args = get_args().dict_()

    ray.init(runtime_env={"py_modules": ["AlphaZeroParallel.py"]})
    
    alphaZero = AlphaZeroParallelRay(model, optimizer, game, args)
    alphaZero.learn()

    ray.shutdown()


def model_play():
    game = TicTacToe()
    player = 1
    
    args = {
        'C': 2,
        'num_searches': 1000,
        'dirichlet_epsilon': 0.,
        'dirichlet_alpha': 0.3
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ResNet(game, 4, 64, device)
    model.load_state_dict(torch.load("./saved_model/model_2_TicTacToe.pt", map_location=device))
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
            mcts_probs = mcts.search(neutral_state)
            action = np.argmax(mcts_probs)
            
        state = game.get_next_state(state, action, player)
        
        value, is_terminal = game.get_value_and_terminated(state, action)
        
        if is_terminal:
            print(state)
            if value == 1:
                print(player, "won")
            else:
                print("draw")
            break
            
        player = game.get_opponent(player)


