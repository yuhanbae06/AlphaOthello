from Node import Node

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tqdm import trange
import random
import pickle
import ray
import os

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
            spg.root.is_end = True
            spg.root.expand(spg_policy)

            spg.search_depth_average += spg.root.depth
            spg.search_depth_num += 1
        
        for search in range(self.args['num_searches']):
            for spg in spGames:
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    node = node.select()

                if not node.is_terminal:
                    node.is_end = True
                    spg.search_depth_max = max(spg.search_depth_max, node.depth)
                    if node.parent.is_end:
                        node.parent.is_end = False
                        spg.search_depth_average += 1 / spg.search_depth_num
                    else:
                        spg.search_depth_num += 1
                        spg.search_depth_average *= (spg.search_depth_num - 1) / spg.search_depth_num
                        spg.search_depth_average += node.depth / spg.search_depth_num

                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
                if self.game.__repr__() != "Othello":
                    value = self.game.get_opponent_value(value)
                
                if is_terminal:
                    node.is_terminal = True
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


# +
class SelfPlay:
    def __init__(self, model, game, args, monitor):
        self.model = model
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)
        self.monitor = monitor

    def play(self):
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
                        if value * player == 1:
                            return_history['win'] += 1
                        elif value * player == -1:
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

class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args, monitor=False):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)
        self.monitor = monitor
        self.history = dict(win=0, draw=0, lose=0, policy_losses=[], value_losses=[])
        
    # def selfPlay(self):
    #     return_memory = []
    #     return_history = dict(win=0, draw=0, lose=0)
    #     player = 1
    #     spGames = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]
        
    #     while len(spGames) > 0:
    #         states = np.stack([spg.state for spg in spGames])
    #         neutral_states = self.game.change_perspective(states, player)
            
    #         self.mcts.search(neutral_states, spGames)
            
    #         for i in range(len(spGames))[::-1]:
    #             spg = spGames[i]
                
    #             action_probs = np.zeros(self.game.action_size)
    #             for child in spg.root.children:
    #                 action_probs[child.action_taken] = child.visit_count
    #             action_probs /= np.sum(action_probs)

    #             spg.memory.append((spg.root.state, action_probs, player))

    #             temperature_action_probs = action_probs ** (1 / self.args['temperature'])
    #             temperature_action_probs /= np.sum(temperature_action_probs)
    #             action = np.random.choice(self.game.action_size, p=temperature_action_probs) # Divide temperature_action_probs with its sum in case of an error

    #             spg.state = self.game.get_next_state(spg.state, action, player)

    #             value, is_terminal = self.game.get_value_and_terminated(self.game.change_perspective(spg.state, player), action)

    #             if is_terminal:
    #                 if self.monitor:
    #                     if value * player == 1:
    #                         return_history['win'] += 1
    #                     elif value * player == -1:
    #                         return_history['lose'] += 1
    #                     else:
    #                         return_history['draw'] += 1
    #                 for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
    #                     hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
    #                     return_memory.append((
    #                         self.game.get_encoded_state(hist_neutral_state),
    #                         hist_action_probs,
    #                         hist_outcome
    #                     ))
    #                 del spGames[i]
                    
    #         player = self.game.get_opponent(player)
            
    #     return return_memory, return_history
                
    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory), batchIdx + self.args['batch_size'])] # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
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
                selfPlayIns = SelfPlay(self.model, self.game, self.args, self.monitor)
                return_memory, return_history = selfPlayIns.play()
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


# +
@ray.remote(num_cpus=1, num_gpus=0.05)
class SelfPlayRay:
    def __init__(self, model, game, args, monitor):
        self.model = model
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)
        self.monitor = monitor

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def play(self):
        return_memory = []
        return_history = dict(win=0, draw=0, lose=0, final_state=None, average_depth=[], max_depth=[])
        player = 1
        finish_games = 0
        random_number = np.random.randint(self.args['num_parallel_games'])
        spGames = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]
        
        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])
            neutral_states = self.game.change_perspective(states, player)
            
            self.mcts.search(neutral_states, spGames)

            average_depth = 0
            max_depth = 0
            cnt = 0
            for i in range(len(spGames))[::-1]:
                spg = spGames[i]

                cnt += 1
                average_depth *= (cnt - 1) / cnt
                average_depth += spg.search_depth_average / cnt
                max_depth *= (cnt - 1) / cnt
                max_depth += spg.search_depth_max / cnt
                spg.reset_depth()
                
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
                        if finish_games == random_number:
                            return_history['final_state'] = spg.state
                        finish_games += 1
                        if value * player == 1:
                            return_history['win'] += 1
                        elif value * player == -1:
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

            return_history['average_depth'].append(average_depth)
            return_history['max_depth'].append(max_depth)
            
            player = self.game.get_opponent(player)
            
        return return_memory, return_history

class AlphaZeroParallelRay:
    def __init__(self, model, optimizer, game, args, monitor=False, log_dir="logs"):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)
        self.monitor = monitor
        self.history = dict(win=0, draw=0, lose=0, average_depth=[], max_depth=[])
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
    
    # @ray.remote(num_gpus=0.2)
    # def selfPlay(self):
    #     return_memory = []
    #     return_history = dict(win=0, draw=0, lose=0)
    #     player = 1
    #     spGames = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]
        
    #     while len(spGames) > 0:
    #         states = np.stack([spg.state for spg in spGames])
    #         neutral_states = self.game.change_perspective(states, player)
            
    #         self.mcts.search(neutral_states, spGames)
            
    #         for i in range(len(spGames))[::-1]:
    #             spg = spGames[i]
                
    #             action_probs = np.zeros(self.game.action_size)
    #             for child in spg.root.children:
    #                 action_probs[child.action_taken] = child.visit_count
    #             action_probs /= np.sum(action_probs)

    #             spg.memory.append((spg.root.state, action_probs, player))

    #             temperature_action_probs = action_probs ** (1 / self.args['temperature'])
    #             temperature_action_probs /= np.sum(temperature_action_probs)
    #             action = np.random.choice(self.game.action_size, p=temperature_action_probs) # Divide temperature_action_probs with its sum in case of an error

    #             spg.state = self.game.get_next_state(spg.state, action, player)

    #             value, is_terminal = self.game.get_value_and_terminated(self.game.change_perspective(spg.state, player), action)

    #             if is_terminal:
    #                 if self.monitor:
    #                     if value * player == 1:
    #                         return_history['win'] += 1
    #                     elif value * player == -1:
    #                         return_history['lose'] += 1
    #                     else:
    #                         return_history['draw'] += 1
    #                 for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
    #                     hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
    #                     return_memory.append((
    #                         self.game.get_encoded_state(hist_neutral_state),
    #                         hist_action_probs,
    #                         hist_outcome
    #                     ))
    #                 del spGames[i]
                    
    #         player = self.game.get_opponent(player)
            
    #     return return_memory, return_history
                
    def train(self, memory, num_iteration, num_epoch):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory), batchIdx + self.args['batch_size'])] # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
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
#                 self.history['policy_losses'].append(policy_loss.detach().cpu().item())
#                 self.history['value_losses'].append(value_loss.detach().cpu().item())
                self.log_scalar(f"loss/{num_iteration}", loss.detach().cpu().item(), num_epoch*(len(memory)//self.args['batch_size'])+batchIdx/self.args['batch_size'])
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def learn(self):
        distributed_num = self.args["num_selfPlay_iterations"] // self.args["num_parallel_games"]
        cpu_per_actor = (os.cpu_count() - 1) / distributed_num
        gpu_per_actor = 1 / distributed_num if torch.cuda.is_available() else 0

        actors = [
            SelfPlayRay.options(num_cpus=cpu_per_actor, num_gpus=gpu_per_actor).remote(self.model, self.game, self.args, self.monitor) for _ in range(distributed_num)]

        for iteration in range(self.args["num_iterations"]):
            print(f"Iteration {iteration} Start...")

            memory = []

            current_weights = ray.put(self.model.cpu().state_dict())
            [actor.set_weights.remote(current_weights) for actor in actors]

            self.model.to(self.args['device'])
            self.model.eval()

            futures = [actor.play.remote() for actor in actors]
            memory_list = ray.get(futures)

            for i in range(distributed_num):
                return_memory, return_history = memory_list[i]
                memory += return_memory
                self.add_history(return_history)
                self.log_image(f'final_state/{iteration}', self.game.get_visualized_state(return_history['final_state']), i)
                print(len(return_memory))
            
            self.log_scalars('wining_rate', {'win': self.history['win'] / self.args['num_selfPlay_iterations'],
                                             'lose': self.history['lose'] / self.args['num_selfPlay_iterations'],
                                             'draw': self.history['draw'] / self.args['num_selfPlay_iterations']}, iteration)
            
            self.log_list(f'average_depth/{iteration}', self.calculate_average(self.history['average_depth']))
            self.log_list(f'max_depth/{iteration}', self.calculate_average(self.history['max_depth']))
            
            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory, iteration, epoch)

            # if self.monitor:
            #     file = open(f"./saved_history/history_{iteration}_{self.game}.pickle", "wb")
            #     pickle.dump(self.history, file)
            #     file.close()
            #     # print(self.history)
            #     self.history = dict(win=0, draw=0, lose=0, policy_losses=[], value_losses=[])

            torch.save(self.model.state_dict(), f"./saved_model/model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), f"./saved_model/optimizer_{iteration}_{self.game}.pt")
            self.reset_history()

    def add_history(self, return_history):
        for key, value in return_history.items():
            if type(value) is int:
                self.history[key] += value
            elif type(value) is list:
                self.history[key].append(value)

    def reset_history(self):
        self.history = dict(win=0, draw=0, lose=0, average_depth=[], max_depth=[])

    def calculate_average(self, depth_lists):
        return_list = []
        length_list = []
        for depth_list in depth_lists:
            length_list.append(len(depth_list))
        length_list = np.array(length_list)
        for i in range(max(length_list)):
            val = 0
            for depth_list in depth_lists:
                if len(depth_list) > i:
                    val += depth_list[i]
            return_list.append(val / sum(length_list > i))
        return return_list

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag, values, step):
        self.writer.add_scalars(tag, values, step)

    def log_list(self, tag, value_list):
        for i in range(len(value_list)):
            self.writer.add_scalar(tag, value_list[i], i)

    def log_image(self, tag, value, step):
        self.writer.add_image(tag, value, step)

    def close_writer(self):
        self.writer.close()


# -

class SPG:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None
        self.search_depth_average = 0
        self.search_depth_num = 0
        self.search_depth_max = 0

    def reset_depth(self):
        self.search_depth_average = 0
        self.search_depth_num = 0
        self.search_depth_max = 0
