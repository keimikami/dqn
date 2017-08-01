import numpy as np
import random

class ReplayBuffer:
	def __init__(self, size):
		self.memory = []
		self.max = size
		self.idx = 0

	def add(self, state, action, reward, next_state, done):
		data = (state, action, reward, next_state, done)

		if self.idx >= len(self.memory):
			self.memory.append(data)
		else:
			self.memory[self.idx] = data
		self.idx = (self.idx + 1) % self.max

	def sample(self, batch_size):
		idxes = np.random.choice(len(self.memory), batch_size)
		state_batch, action_batch, reward_batch, next_state_batch, done_batch = [], [], [], [], []

		for i in idxes:
			state, action, reward, next_state, done = self.memory[i]
			state_batch.append(np.array(state, copy=False))
			action_batch.append(np.array(action, copy=False))
			reward_batch.append(reward)
			next_state_batch.append(np.array(next_state, copy=False))
			done_batch.append(done)

		return np.array(state_batch), np.array(action_batch), np.array(reward_batch), np.array(next_state_batch), np.array(done_batch)