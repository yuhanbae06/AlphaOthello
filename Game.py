import numpy as np

class TicTacToe:
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count
        
    def __repr__(self):
        return "TicTacToe"
        
    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))
    
    def get_next_state(self, state, action, player):
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = player
        return state
    
    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        if action == None:
            return False
        
        row = action // self.column_count
        column = action % self.column_count
        player = state[row, column]
        
        return (
            np.sum(state[row, :]) == player * self.column_count
            or np.sum(state[:, column]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count
        )
    
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        return state * player
    
    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        
        return encoded_state


class ConnectFour:
    def __init__(self):
        self.row_count = 6
        self.column_count = 7
        self.action_size = self.column_count
        self.in_a_row = 4
        
    def __repr__(self):
        return "ConnectFour"
        
    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))
    
    def get_next_state(self, state, action, player):
        row = np.max(np.where(state[:, action] == 0))
        state[row, action] = player
        return state
    
    def get_valid_moves(self, state):
        return (state[0] == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        if action == None:
            return False
        
        row = np.min(np.where(state[:, action] != 0))
        column = action
        player = state[row][column]

        def count(offset_row, offset_column):
            for i in range(1, self.in_a_row):
                r = row + offset_row * i
                c = action + offset_column * i
                if (
                    r < 0 
                    or r >= self.row_count
                    or c < 0 
                    or c >= self.column_count
                    or state[r][c] != player
                ):
                    return i - 1
            return self.in_a_row - 1

        return (
            count(1, 0) >= self.in_a_row - 1 # vertical
            or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1 # horizontal
            or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1 # top left diagonal
            or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1 # top right diagonal
        )
    
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        return state * player
    
    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        
        return encoded_state


'''
변수 이름

state: 판의 상태이다. row*col의 2차원 np.array로 구현. 1,-1은 돌이며 0은 아무것도 없음음
action: state를 변화 시키는 것(ex: 새로운 돌을 놓고 그 돌로 인해 뒤집어지는거 계산). np.array 1차원 행렬 [x,y]
player: int로 표현; 1돌의 주인이 1이고 -1돌의 주인이 -1
'''


class Othello:
    def __init__(self):
        self.row_count = 6 #should be even!
        self.column_count = 6
        self.action_size = self.row_count * self.column_count + 1
        self.interval = 16

    def __repr__(self):
        return "Othello"
    
    def get_initial_state(self):
        initialstate=np.zeros(shape=(self.row_count,self.column_count),dtype=int)
        initialstate[self.row_count//2-1:self.row_count//2+1,self.column_count//2-1:self.column_count//2+1]=np.array([[-1,1],[1,-1]])
        # print(initialstate)
        return initialstate
    
    def restrict(self, row, col):
        return row >= 0 and row < self.row_count and col >= 0 and col < self.column_count
    
    def get_next_state(self, state, action, player):
        #둘다 -1 이면 돌 못 놓은거
        if action == self.action_size - 1:
            pass
        else:
            row = action // self.column_count
            col = action % self.column_count
            state[row,col]=player
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    if dx==0 and dy==0: continue
                    x=row
                    y=col
                    while True:
                        x+=dx
                        y+=dy
                        # print(x,y,dx,dy)
                        if (not self.restrict(x,y)) or state[x,y]==0:
                            # print('fuck')
                            break
                        elif state[x,y]==player:
                            x-=dx
                            y-=dy
                            while x!=row or y!=col:
                                state[x,y]=player
                                x-=dx
                                y-=dy
                            break

        return state

    def get_valid_moves(self, state):
        ans=np.zeros(shape=self.action_size,dtype=np.uint8)
        for ij in range(self.row_count*self.column_count):
            i=ij//self.column_count
            j=ij%self.column_count

            if(state[i,j]): continue

            for dx in [-1,0,1]:
                if(ans[ij]): break
                for dy in [-1,0,1]:
                    if(ans[ij]): break
                    if dx==0 and dy==0: continue
                    x=i
                    y=j
                    flip = False
                    while True:
                        x+=dx
                        y+=dy
                        if (not self.restrict(x,y)) or state[x,y]==0:
                            break
                        if state[x,y]==1:
                            if flip:
                                ans[ij]=1
                                break
                            else:
                                break
                        elif state[x,y]==-1:
                            flip=True
        if np.sum(ans) == 0:
            ans[-1] = 1
        return ans

    def check_finish(self, state):
        return self.get_valid_moves(state)[-1] == 1 and self.get_valid_moves(self.change_perspective(state, -1))[-1] == 1
    
    def check_winner(self, state):
        res = np.sum(state)
        if res > 0:
            return 1
        elif res < 0:
            return -1
        else:
            return 0

    def get_value_and_terminated(self, state, action):
        if self.check_finish(state):
            return self.check_winner(state), True
        return 0, False

    def change_perspective(self, state, player:int):
        return state*player

    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value

    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        
        return encoded_state

    def get_visualized_state(self, state):
        size = 16
        visualized_state = np.zeros((3, self.row_count*size, self.row_count*size))
        for i in range(self.row_count):
            for j in range(self.column_count):
                if state[i, j] == 1:
                    visualized_state[0, i*self.interval:(i+1)*self.interval, j*self.interval:(j+1)*self.interval] = np.ones((self.interval, self.interval)) * 0
                    visualized_state[1, i*self.interval:(i+1)*self.interval, j*self.interval:(j+1)*self.interval] = np.ones((self.interval, self.interval)) * 0
                    visualized_state[2, i*self.interval:(i+1)*self.interval, j*self.interval:(j+1)*self.interval] = np.ones((self.interval, self.interval)) * 0
                elif state[i, j] == 0:
                    visualized_state[0, i*self.interval:(i+1)*self.interval, j*self.interval:(j+1)*self.interval] = np.ones((self.interval, self.interval)) * 0
                    visualized_state[1, i*self.interval:(i+1)*self.interval, j*self.interval:(j+1)*self.interval] = np.ones((self.interval, self.interval)) * 1
                    visualized_state[2, i*self.interval:(i+1)*self.interval, j*self.interval:(j+1)*self.interval] = np.ones((self.interval, self.interval)) * 0
                elif state[i, j] == -1:
                    visualized_state[0, i*self.interval:(i+1)*self.interval, j*self.interval:(j+1)*self.interval] = np.ones((self.interval, self.interval)) * 1
                    visualized_state[1, i*self.interval:(i+1)*self.interval, j*self.interval:(j+1)*self.interval] = np.ones((self.interval, self.interval)) * 1
                    visualized_state[2, i*self.interval:(i+1)*self.interval, j*self.interval:(j+1)*self.interval] = np.ones((self.interval, self.interval)) * 1

        return visualized_state

class GomokuNaive:
    def __init__(self):
        self.row_count = 10
        self.column_count = 10
        self.action_size = self.row_count * self.column_count
        self.interval = 16

    def __repr__(self):
        return "GomokuNaive"

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count), dtype=int)

    def get_next_state(self, state, action, player):
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = player
        return state


    def get_valid_moves(self, state):
        valid_moves = (state.reshape(-1) == 0).astype(np.uint8)
        return valid_moves

    def get_max_continuous(self, state, r, c, player):
        max_c = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for direction in [1, -1]:
                nr, nc = r + dr * direction, c + dc * direction
                while 0 <= nr < self.row_count and 0 <= nc < self.column_count and state[nr, nc] == player:
                    count += 1
                    nr += dr * direction
                    nc += dc * direction
            max_c = max(max_c, count)
        return max_c

    def check_win(self, state, action):
        if action is None:
            return False

        row = action // self.column_count
        col = action % self.column_count
        player = state[row, col]
        count = self.get_max_continuous(state, row, col, player)

        return count >= 5

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(state == 0) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return state * player

    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state

    def get_visualized_state(self, state):
        size = 16
        visualized_state = np.zeros((3, self.row_count * size, self.row_count * size))

        for i in range(self.row_count):
            for j in range(self.column_count):
                r0, r1 = i * self.interval, (i + 1) * self.interval
                c0, c1 = j * self.interval, (j + 1) * self.interval

                if state[i, j] == 1:
                    visualized_state[0, r0:r1, c0:c1] = np.ones((self.interval, self.interval)) * 0
                    visualized_state[1, r0:r1, c0:c1] = np.ones((self.interval, self.interval)) * 0
                    visualized_state[2, r0:r1, c0:c1] = np.ones((self.interval, self.interval)) * 0

                elif state[i, j] == 0:
                    visualized_state[0, r0:r1, c0:c1] = np.ones((self.interval, self.interval)) * 0
                    visualized_state[1, r0:r1, c0:c1] = np.ones((self.interval, self.interval)) * 1
                    visualized_state[2, r0:r1, c0:c1] = np.ones((self.interval, self.interval)) * 0

                elif state[i, j] == -1:
                    visualized_state[0, r0:r1, c0:c1] = np.ones((self.interval, self.interval)) * 1
                    visualized_state[1, r0:r1, c0:c1] = np.ones((self.interval, self.interval)) * 1
                    visualized_state[2, r0:r1, c0:c1] = np.ones((self.interval, self.interval)) * 1

        return visualized_state

class Play:
    def __init__(self):
        self.game = Othello()
        self.state = self.game.get_initial_state()
        self.player = 1

    def play(self):
        while not self.game.check_finish(self.game.change_perspective(self.state, self.player)):
            print(self.game.get_valid_moves(self.game.change_perspective(self.state, self.player)))
            print("{0}의 차례".format(self.player))
            self.print_state()
            row = int(input("행: ")) - 1
            col = int(input("열: ")) - 1
            if row == -1 and col == -1:
                action = self.game.action_size - 1
            else:
                action = row * self.game.row_count + col
            self.state = self.game.get_next_state(self.state, action, self.player)
            self.player = self.game.get_opponent(self.player)
        self.print_state()
        print("{0}의 승리!".format(self.game.check_winner(self.state)))
    
    def print_state(self):
        print(self.state)
