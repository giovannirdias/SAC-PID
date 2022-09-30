import random
import numpy as np

# Replay buffer
class buffer:
    def __init__(self, capacidade):
        self.capacidade = capacidade
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, val):
        if len(self.buffer) < self.capacidade:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, val)
        self.pos = (self.pos+1) % self.capacidade

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, val = map(np.stack, zip(*batch))
        return state, action, reward, next_state, val
