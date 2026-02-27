import numpy as np


class Gomoku10:
    def __init__(self):
        self.row_count = 10
        self.column_count = 10
        self.action_size = self.row_count * self.column_count
        self.input_channels = 3
        self.encoded_state_size = self.input_channels * self.row_count * self.column_count
        self.interval = 16
        self.game_name = "gomoku"

    def __repr__(self):
        return "Gomoku10"

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count), dtype=np.int8)

    def get_next_state(self, state, action, player):
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = player
        return state

    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)

    def _count_in_direction(self, state, r, c, dr, dc, player):
        count = 1
        nr, nc = r + dr, c + dc
        while 0 <= nr < self.row_count and 0 <= nc < self.column_count and state[nr, nc] == player:
            count += 1
            nr += dr
            nc += dc
        nr, nc = r - dr, c - dc
        while 0 <= nr < self.row_count and 0 <= nc < self.column_count and state[nr, nc] == player:
            count += 1
            nr -= dr
            nc -= dc
        return count

    def check_win(self, state, action):
        if action is None:
            return False
        row = action // self.column_count
        col = action % self.column_count
        player = state[row, col]
        if player == 0:
            return False
        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        for dr, dc in directions:
            if self._count_in_direction(state, row, col, dr, dc, player) >= 5:
                return True
        return False

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
        encoded_state = np.stack((state == -1, state == 0, state == 1)).astype(np.float32)
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        return encoded_state

    def reshape_encoded_state(self, flat):
        return flat.reshape(self.input_channels, self.row_count, self.column_count)

    def get_visualized_state(self, state):
        size = self.interval
        visualized_state = np.zeros((3, self.row_count * size, self.column_count * size), dtype=np.float32)
        for i in range(self.row_count):
            for j in range(self.column_count):
                r0, r1 = i * size, (i + 1) * size
                c0, c1 = j * size, (j + 1) * size
                if state[i, j] == 1:
                    visualized_state[:, r0:r1, c0:c1] = 0.0
                elif state[i, j] == -1:
                    visualized_state[:, r0:r1, c0:c1] = 1.0
                else:
                    visualized_state[1, r0:r1, c0:c1] = 1.0
        return visualized_state

    def final_state_size(self):
        return self.row_count * self.column_count

    def decode_final_state(self, raw):
        return np.frombuffer(raw, dtype=np.int8).copy().reshape(self.row_count, self.column_count)


class Quoridor7:
    def __init__(self):
        self.row_count = 7
        self.column_count = 7
        self.action_size = 77
        self.input_channels = 6
        self.encoded_state_size = self.input_channels * self.row_count * self.column_count
        self.interval = 20
        self.game_name = "quoridor7"

    def __repr__(self):
        return "Quoridor7"

    def reshape_encoded_state(self, flat):
        return flat.reshape(self.input_channels, self.row_count, self.column_count)

    def get_encoded_state(self, state):
        encoded_state = np.zeros(
            (self.input_channels, self.row_count, self.column_count), dtype=np.float32
        )
        encoded_state[0] = (state == 1).astype(np.float32)
        encoded_state[1] = (state == -1).astype(np.float32)
        encoded_state[4].fill(1.0)
        encoded_state[5].fill(1.0)
        return encoded_state

    def get_visualized_state(self, state):
        if isinstance(state, dict):
            board = np.asarray(state.get("board"), dtype=np.int8)
            walls_h = np.asarray(state.get("walls_h"), dtype=np.int8)
            walls_v = np.asarray(state.get("walls_v"), dtype=np.int8)
        else:
            board = np.asarray(state, dtype=np.int8)
            walls_h = np.zeros((self.row_count + 1, self.column_count + 1), dtype=np.int8)
            walls_v = np.zeros((self.row_count + 1, self.column_count + 1), dtype=np.int8)

        size = self.interval
        height = self.row_count * size + 1
        width = self.column_count * size + 1
        visualized_state = np.zeros((3, height, width), dtype=np.float32)

        # board background
        visualized_state[0].fill(0.95)
        visualized_state[1].fill(0.86)
        visualized_state[2].fill(0.70)

        # grid lines
        grid_color = 0.25
        for r in range(self.row_count + 1):
            y = r * size
            y1 = min(height, y + 1)
            visualized_state[:, y:y1, :] = grid_color
        for c in range(self.column_count + 1):
            x = c * size
            x1 = min(width, x + 1)
            visualized_state[:, :, x:x1] = grid_color

        # walls from walls_h / walls_v (black thick line)
        wall_thickness = max(2, size // 4)
        for wr in range(walls_h.shape[0]):
            for wc in range(walls_h.shape[1]):
                if walls_h[wr, wc] <= 0:
                    continue
                y = wr * size
                x0 = (wc - 1) * size
                x1 = (wc + 1) * size
                yy0 = max(0, y - wall_thickness // 2)
                yy1 = min(height, y + wall_thickness // 2 + 1)
                xx0 = max(0, x0)
                xx1 = min(width, x1 + 1)
                if yy0 < yy1 and xx0 < xx1:
                    visualized_state[:, yy0:yy1, xx0:xx1] = 0.0

        for wr in range(walls_v.shape[0]):
            for wc in range(walls_v.shape[1]):
                if walls_v[wr, wc] <= 0:
                    continue
                x = wc * size
                y0 = (wr - 1) * size
                y1 = (wr + 1) * size
                xx0 = max(0, x - wall_thickness // 2)
                xx1 = min(width, x + wall_thickness // 2 + 1)
                yy0 = max(0, y0)
                yy1 = min(height, y1 + 1)
                if yy0 < yy1 and xx0 < xx1:
                    visualized_state[:, yy0:yy1, xx0:xx1] = 0.0

        # pawns
        radius = max(2, size // 3)
        rr = radius * radius
        for r in range(self.row_count):
            for c in range(self.column_count):
                val = int(board[r, c])
                if val == 0:
                    continue
                cy = r * size + size // 2
                cx = c * size + size // 2
                for dy in range(-radius, radius + 1):
                    yy = cy + dy
                    if yy < 0 or yy >= height:
                        continue
                    for dx in range(-radius, radius + 1):
                        if dx * dx + dy * dy > rr:
                            continue
                        xx = cx + dx
                        if xx < 0 or xx >= width:
                            continue
                        if val > 0:
                            visualized_state[0, yy, xx] = 0.10
                            visualized_state[1, yy, xx] = 0.30
                            visualized_state[2, yy, xx] = 0.95
                        else:
                            visualized_state[0, yy, xx] = 0.92
                            visualized_state[1, yy, xx] = 0.20
                            visualized_state[2, yy, xx] = 0.15
        return visualized_state

    def final_state_size(self):
        wall_area = (self.row_count + 1) * (self.column_count + 1)
        return self.row_count * self.column_count + 2 * wall_area

    def decode_final_state(self, raw):
        area = self.row_count * self.column_count
        wall_area = (self.row_count + 1) * (self.column_count + 1)
        arr = np.frombuffer(raw, dtype=np.int8).copy()
        if arr.size == area:
            # backward compatibility: old format had only board plane
            board = arr.reshape(self.row_count, self.column_count)
            walls_h = np.zeros((self.row_count + 1, self.column_count + 1), dtype=np.int8)
            walls_v = np.zeros((self.row_count + 1, self.column_count + 1), dtype=np.int8)
            return {"board": board, "walls_h": walls_h, "walls_v": walls_v}
        expected = area + 2 * wall_area
        if arr.size != expected:
            raise ValueError(f"Unexpected quoridor final_state size: got {arr.size}, expected {expected}")
        board = arr[:area].reshape(self.row_count, self.column_count)
        walls_h = arr[area : area + wall_area].reshape(self.row_count + 1, self.column_count + 1)
        walls_v = arr[area + wall_area : area + 2 * wall_area].reshape(
            self.row_count + 1, self.column_count + 1
        )
        return {"board": board, "walls_h": walls_h, "walls_v": walls_v}

    def get_initial_state(self):
        raise NotImplementedError("Quoridor Python play/test path is not implemented")

    def get_next_state(self, state, action, player):
        raise NotImplementedError("Quoridor Python play/test path is not implemented")

    def get_valid_moves(self, state):
        raise NotImplementedError("Quoridor Python play/test path is not implemented")

    def get_value_and_terminated(self, state, action):
        raise NotImplementedError("Quoridor Python play/test path is not implemented")

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return state


class Quoridor9(Quoridor7):
    def __init__(self):
        self.row_count = 9
        self.column_count = 9
        self.action_size = 133
        self.input_channels = 6
        self.encoded_state_size = self.input_channels * self.row_count * self.column_count
        self.interval = 16
        self.game_name = "quoridor9"

    def __repr__(self):
        return "Quoridor9"


class Quoridor5(Quoridor7):
    def __init__(self):
        self.row_count = 5
        self.column_count = 5
        self.action_size = 37
        self.input_channels = 6
        self.encoded_state_size = self.input_channels * self.row_count * self.column_count
        self.interval = 24
        self.game_name = "quoridor5"

    def __repr__(self):
        return "Quoridor5"


def make_game(game_name: str):
    normalized = str(game_name).strip().lower()
    if normalized in ("gomoku", "gomoku10"):
        return Gomoku10()
    if normalized in ("quoridor5", "quoridor_5"):
        return Quoridor5()
    if normalized in ("quoridor", "quoridor7"):
        return Quoridor7()
    if normalized in ("quoridor9", "quoridor_9"):
        return Quoridor9()
    raise ValueError(f"Unsupported game: {game_name}")
