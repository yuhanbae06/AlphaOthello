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
from Node import Node

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tqdm.notebook import trange
import random
import pickle
import ray


# -

class MCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
        
    @torch.no_grad()
    def search(self, states, spGames):
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])
        
        for i, spg in enumerate(spGames):
            spg_policy = policy[i]
            valid_moves = self.game.get_valid_moves(states[i])
            spg_policy *= valid_moves
            spg_policy /= np.sum(spg_policy)

            spg.root = Node(self.game, self.args, states[i], visit_count=1)
            spg.root.expand(spg_policy)
        
        for search in range(self.args['num_searches']):
            for spg in spGames:
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    node = node.select()

                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
                if self.game.__repr__() != "Othello":
                    value = self.game.get_opponent_value(value)
                
                if is_terminal:
                    node.backpropagate(value)
                    
                else:
                    spg.node = node
                    
            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if spGames[mappingIdx].node is not None]
                    
            if len(expandable_spGames) > 0:
                states = np.stack([spGames[mappingIdx].node.state for mappingIdx in expandable_spGames])
                
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                )
                policy = torch.softmax(policy, axis=1).detach().cpu().numpy()
                value = value.detach().cpu().numpy()
                
            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                spg_policy, spg_value = policy[i], value[i]
                
                valid_moves = self.game.get_valid_moves(node.state)
                spg_policy *= valid_moves
                spg_policy /= np.sum(spg_policy)

                node.expand(spg_policy)
                node.backpropagate(spg_value)


class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args, monitor=False):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)
        self.monitor = monitor
        self.history = dict(win=0, draw=0, lose=0, policy_losses=[], value_losses=[])
        
    def selfPlay(self):
        return_memory = []
        return_history = dict(win=0, draw=0, lose=0)
        player = 1
        spGames = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]
        
        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])
            neutral_states = self.game.change_perspective(states, player)
            
            self.mcts.search(neutral_states, spGames)
            
            for i in range(len(spGames))[::-1]:
                spg = spGames[i]
                
                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs)

                spg.memory.append((spg.root.state, action_probs, player))

                temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                temperature_action_probs /= np.sum(temperature_action_probs)
                action = np.random.choice(self.game.action_size, p=temperature_action_probs) # Divide temperature_action_probs with its sum in case of an error

                spg.state = self.game.get_next_state(spg.state, action, player)

                value, is_terminal = self.game.get_value_and_terminated(self.game.change_perspective(spg.state, player), action)

                if is_terminal:
                    if self.monitor:
                        if value and player == 1:
                            return_history['win'] += 1
                        elif value and player == -1:
                            return_history['lose'] += 1
                        else:
                            return_history['draw'] += 1
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                        return_memory.append((
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    del spGames[i]
                    
            player = self.game.get_opponent(player)
            
        return return_memory, return_history
                
    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])] # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
            state, policy_targets, value_targets = zip(*sample)
            
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
            
            out_policy, out_value = self.model(state)
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            if self.monitor:
                self.history['policy_losses'].append(policy_loss.detach().cpu().item())
                self.history['value_losses'].append(value_loss.detach().cpu().item())
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []
            
            self.model.eval()
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
                return_memory, return_history = self.selfPlay()
                memory += return_memory
                self.add_history(return_history)
                
            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)

            if self.monitor:
                file = open(f"./saved_history/history_{iteration}_{self.game}.pickle", "wb")
                pickle.dump(self.history, file)
                file.close()
                # print(self.history)
                self.history = dict(win=0, draw=0, lose=0, policy_losses=[], value_losses=[])
            
            torch.save(self.model.state_dict(), f"./saved_model/model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), f"./saved_model/optimizer_{iteration}_{self.game}.pt")

    def add_history(self, return_history):
        for key, value in return_history.items():
            self.history[key] += value


class AlphaZeroParallelRay:
    def __init__(self, model, optimizer, game, args, monitor=False, log_dir="logs"):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)
        self.monitor = monitor
        self.history = dict(win=0, draw=0, lose=0, policy_losses=[], value_losses=[])
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['writer'] = None  # Remove writer before pickling
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.writer = SummaryWriter(log_dir=self.log_dir)  # Recreate writer after unpickling
    
    @ray.remote(num_gpus=0.2)
    def selfPlay(self):
        return_memory = []
        return_history = dict(win=0, draw=0, lose=0)
        player = 1
        spGames = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]
        
        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])
            neutral_states = self.game.change_perspective(states, player)
            
            self.mcts.search(neutral_states, spGames)
            
            for i in range(len(spGames))[::-1]:
                spg = spGames[i]
                
                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs)

                spg.memory.append((spg.root.state, action_probs, player))

                temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                temperature_action_probs /= np.sum(temperature_action_probs)
                action = np.random.choice(self.game.action_size, p=temperature_action_probs) # Divide temperature_action_probs with its sum in case of an error

                spg.state = self.game.get_next_state(spg.state, action, player)

                value, is_terminal = self.game.get_value_and_terminated(self.game.change_perspective(spg.state, player), action)

                if is_terminal:
                    if self.monitor:
                        if value and player == 1:
                            return_history['win'] += 1
                        elif value and player == -1:
                            return_history['lose'] += 1
                        else:
                            return_history['draw'] += 1
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                        return_memory.append((
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    del spGames[i]
                    
            player = self.game.get_opponent(player)
            
        return return_memory, return_history
                
    def train(self, memory, num_iteration, num_epoch):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])] # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
            state, policy_targets, value_targets = zip(*sample)
            
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
            
            out_policy, out_value = self.model(state)
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            if self.monitor:
                self.history['policy_losses'].append(policy_loss.detach().cpu().item())
                self.history['value_losses'].append(value_loss.detach().cpu().item())
                self.log_scalar("loss_"+str(num_iteration), loss, num_epoch*(len(memory)//self.args['batch_size'])+batchIdx/self.args['batch_size'])
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []
            
            self.model.eval()

            futures = [self.selfPlay.remote(self) for i in range(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games'])]
            memory_list = ray.get(futures)
            for i in range(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
                return_memory, return_history = memory_list[i]
                memory += return_memory
                self.add_history(return_history)
                print(len(return_memory))
                
            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory, iteration, epoch)

            if self.monitor:
                file = open(f"./saved_history/history_{iteration}_{self.game}.pickle", "wb")
                pickle.dump(self.history, file)
                file.close()
                # print(self.history)
                self.history = dict(win=0, draw=0, lose=0, policy_losses=[], value_losses=[])
            
            torch.save(self.model.state_dict(), f"./saved_model/model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), f"./saved_model/optimizer_{iteration}_{self.game}.pt")

    def add_history(self, return_history):
        for key, value in return_history.items():
            self.history[key] += value

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def close_writer(self):
        self.writer.close()


class SPG:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None
