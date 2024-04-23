import numpy as np
import random

class SnakeEnv:
    def __init__(self, grid_size=(10, 10)):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.snake = [(self.grid_size[0]//2, self.grid_size[1]//2)]
        self.score = 0
        self.food = None
        self._place_food()
        self.done = False
        self.direction = (0, 1)  # Start moving right
        return self._get_observation()

    def _place_food(self):
        while True:
            food = (random.randint(0, self.grid_size[0]-1), random.randint(0, self.grid_size[1]-1))
            if food not in self.snake:
                self.food = food
                break

    def _get_observation(self):
        """ Create a grid representation of the snake and food """
        obs = np.zeros(self.grid_size, dtype=int)
        for segment in self.snake:
            obs[segment] = 1
        if self.food:
            obs[self.food] = 2
        return obs

    def step(self, action):
        # Define movement
        direction_map = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # UP, RIGHT, DOWN, LEFT
        if ((action == 0 and self.direction != (1, 0)) or
            (action == 2 and self.direction != (-1, 0)) or
            (action == 1 and self.direction != (0, -1)) or
            (action == 3 and self.direction != (0, 1))):
            self.direction = direction_map[action]

        # Move snake
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        if (new_head in self.snake or
            new_head[0] < 0 or new_head[0] >= self.grid_size[0] or
            new_head[1] < 0 or new_head[1] >= self.grid_size[1]):
            self.done = True
            return self._get_observation(), -1, self.done, {}

        self.snake.insert(0, new_head)
        reward = 0

        if new_head == self.food:
            self.score += 1
            reward = 1
            self._place_food()
        else:
            self.snake.pop()

        return self._get_observation(), reward, self.done, {}

    def render(self, mode='human'):
        if mode == 'human':
            for i in range(self.grid_size[0]):
                print(''.join(['#' if (i, j) in self.snake else 'O' if (i, j) == self.food else '.' for j in range(self.grid_size[1])]))
            print("Score:", self.score)
