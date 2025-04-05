import numpy as np
import gymnasium as gym
import random
from gymnasium.spaces import Discrete, Box
import pygame

# Actions
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Rendering constants
CELL_SIZE = 50
AGENT_COLOR = (0, 0, 255)  # Blue
GOAL_COLOR = (0, 255, 0)   # Green
WALL_COLOR = (139, 69, 19) # Brown
EMPTY_COLOR = (255, 255, 255) # White
GRID_COLOR = (200, 200, 200)  # Light gray

class DirtEnv(gym.Env):
    def __init__(self, n=5, p=0.8, dirt_count = 1):
        super().__init__()
        self.n = n
        self.p = p  # Probability of empty space vs wall
        self.observation_space = Discrete(n * n)
        self.action_space = Discrete(4)
        self.window = None
        self.clock = None
        self.initial_dirt_count = dirt_count

    def inc(self, row, col, action):
        new_row, new_col = row, col
        if action == LEFT:
            new_col = max(0, col - 1)
        elif action == RIGHT:
            new_col = min(self.n - 1, col + 1)
        elif action == UP:
            new_row = max(0, row - 1)
        elif action == DOWN:
            new_row = min(self.n - 1, row + 1)
        
        # Check if new position is a wall
        if self.board[new_row][new_col] == 'W':
            return row, col  # Stay in current position if wall
        return new_row, new_col

    def step(self, action):
        row = self.obs // self.n
        col = self.obs % self.n
        new_row, new_col = self.inc(row, col, action)
        self.obs = new_row * self.n + new_col

        reward = -1
        done = False

        # Distance-based penalty: sum of Manhattan distances
        distance_penalty = sum(abs(gx - new_row) + abs(gy - new_col) for gx, gy in self.goal_positions)
        reward -= 2*distance_penalty/(len(self.goal_positions) * self.n)
        
        if self.board[new_row][new_col] == 'G':
            reward = 10
            self.dirt_count -= 1
            self.goal_positions.remove((new_row, new_col))
            self.board[new_row][new_col] = 'F'  # Mark as cleaned
            if self.dirt_count == 0:
                done = True

        return self.obs, reward, done, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.dirt_count = self.initial_dirt_count
        self.board, self.goal_positions = self.generate_random_map(seed)
        self.obs = 0  # Always start at (0,0)
        return self.obs

    def generate_random_map(self, seed=None):
        valid = False
        rng = np.random.default_rng(seed)
        while not valid:
            tmp = rng.choice(['F', 'W'], (self.n, self.n), p=[self.p, 1 - self.p])
            tmp[0][0] = 'S'  # Start position

            random.seed(seed)
            goals = set()
            while len(goals) < self.dirt_count:
                x, y = random.randint(1, self.n - 1), random.randint(1, self.n - 1)
                if tmp[x][y] != 'W' and (x, y) != (0, 0):
                    tmp[x][y] = 'G'
                    goals.add((x, y))

            valid = self.is_valid(tmp)

        return tmp, goals

    def is_valid(self, board):
        """Check if all dirt spots are reachable from start using BFS"""
        start = (0, 0)
        goals = [(i, j) for i in range(self.n) for j in range(self.n) if board[i][j] == 'G']
        
        if not goals:
            return False
            
        actions = [(1, 0), (0, 1), (0, -1), (-1, 0)]
        visited = set()
        queue = [start]
        found_goals = set()
        
        while queue and len(found_goals) < len(goals):
            r, c = queue.pop(0)
            
            if (r, c) in goals and (r, c) not in found_goals:
                found_goals.add((r, c))
                
            for dr, dc in actions:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.n and 0 <= nc < self.n and 
                    (nr, nc) not in visited and 
                    board[nr][nc] != 'W'):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        
        return len(found_goals) == len(goals)

    def render(self, mode="human"):
        if self.window is None and mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((self.n * CELL_SIZE, self.n * CELL_SIZE))
            pygame.display.set_caption("DirtEnv")
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.n * CELL_SIZE, self.n * CELL_SIZE))
        canvas.fill(EMPTY_COLOR)

        for row in range(self.n):
            for col in range(self.n):
                cell_val = self.board[row][col]
                color = EMPTY_COLOR
                if cell_val == 'G':
                    color = GOAL_COLOR
                elif cell_val == 'W':
                    color = WALL_COLOR

                pygame.draw.rect(
                    canvas,
                    color,
                    pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                )

                pygame.draw.rect(
                    canvas,
                    GRID_COLOR,
                    pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                    1
                )

        # Draw agent
        agent_row = self.obs // self.n
        agent_col = self.obs % self.n
        pygame.draw.circle(
            canvas,
            AGENT_COLOR,
            (agent_col * CELL_SIZE + CELL_SIZE//2, agent_row * CELL_SIZE + CELL_SIZE//2),
            CELL_SIZE//3
        )

        if mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(5)
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window:
            pygame.quit()
            self.window = None