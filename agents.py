import gym
from gym import wrappers
import numpy as np

from replay_buffer import ReplayBuffer
from models import build_network
from utils import preprocess

import sys
import random
from datetime import datetime

ENV_NAME = "Breakout-v0" # Environment name
NUM_EPISODES = 100000  # Number of episodes
NUM_EPOCHS = 1 # number of iterations over the data set

FRAME_WIDTH = 84  # Frame width
FRAME_HEIGHT = 84  # Frame height
BATCH_SIZE = 32  # Mini batch size

GAMMA = 0.99  # Discount factor

WAITING_STEPS = 30  # Maximum number of "do nothing" at the start of an episode
TRAIN_INTERVAL = 4  # The agent selects 4 actions between successive updates
TARGET_UPDATE_INTERVAL = 10000  # The frequency with which the target network is updated

INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy
EXPLORATION_STEPS = 1000000  # Number of steps to decay epsilon

INITIAL_REPLAY_SIZE = 50000  # Number of steps to fill replay memory
NUM_REPLAY_MEMORY = 1000000  # Size of replay memory

LOAD_NETWORK = False
SAVE_PATH = 'saved_models/'
SAVE_INTERVAL = 1000
DEBUG_INTERVAL = 50
RENDER = True
RECORD = False

class Agent:

	def __init__(self, num_actions):
		self.num_actions = num_actions
		self.step = 0
		self.replay_buffer = ReplayBuffer(NUM_REPLAY_MEMORY)
		self.epsilon = INITIAL_EPSILON
		self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS

		self.model = build_network(FRAME_WIDTH, FRAME_HEIGHT, 1, self.num_actions)
		self.target_network = build_network(FRAME_WIDTH, FRAME_HEIGHT, 1, self.num_actions)

	def select_action(self, state):
		action = 0
		if random.random() <= self.epsilon:
			action = random.randrange(self.num_actions)
		else:
			q_values = self.model.predict(np.reshape(state, (1, FRAME_WIDTH, FRAME_HEIGHT, 1)))
			action = np.argmax(q_values[0])

		if self.epsilon > FINAL_EPSILON and self.step >= INITIAL_REPLAY_SIZE:
			self.epsilon -= self.epsilon_step

		return action

	def train(self):
		(state_batch, action_batch, reward_batch, next_state_batch, done_batch) = self.replay_buffer.sample(BATCH_SIZE)
		
		state_batch = np.float32(state_batch / 255.0)
		next_state_batch = np.float32(next_state_batch / 255.0)
		done_batch = done_batch + 0

		q_batch = reward_batch + (1 - done_batch) * GAMMA * np.amax(self.target_network.predict(next_state_batch))
		y_batch = self.model.predict(state_batch)

		for i in range(BATCH_SIZE):
			y_batch[i][action_batch[i]] = q_batch[i]

		self.model.fit(state_batch, y_batch, epochs=NUM_EPOCHS, verbose=0)

	def train_target(self):
		latest_weights = self.model.get_weights()
		self.target_network.set_weights(latest_weights)

	def reinfoce(self, state, action, reward, next_state, done):
		clipped_reward = np.sign(reward)
		self.replay_buffer.add(state, action, clipped_reward, next_state, done)

		if self.step >= INITIAL_REPLAY_SIZE:
			if self.step % TRAIN_INTERVAL == 0:
				self.train()

			if self.step % TARGET_UPDATE_INTERVAL == 0:
				self.train_target()

		self.step += 1

	def load(self, name):
		self.model.load_weights(name)

	def save(self, name):
		self.model.save_weights(name)

if __name__ == "__main__":
	env = gym.make(ENV_NAME)
	if RECORD:
		env = wrappers.Monitor(env, '/tmp/' + ENV_NAME + '-experiment')
	agent = Agent(env.action_space.n)

	if LOAD_NETWORK:
		args = sys.argv
		if len(args) == 2:
			load_path = args[1]
			agent.load(load_path)
			agent.train_target()
			print('Successfully loaded: ' + load_path)
		else:
			print('$ agents.py <load_path>')
			quit()

	for e in range(1, NUM_EPISODES + 1):
		observation = env.reset()
		last_observation = None
		done = False

		for _ in range(random.randint(1, WAITING_STEPS)):
			last_observation = observation
			observation, _, _, _ = env.step(0)
		state = preprocess(observation, last_observation, FRAME_WIDTH, FRAME_HEIGHT)

		t = 0
		while not done:
			action = agent.select_action(state)
			last_observation = observation
			observation, reward, done, _ = env.step(action)
			if RENDER:
				env.render()	
			next_state = preprocess(observation, last_observation, FRAME_WIDTH, FRAME_HEIGHT)
			agent.reinfoce(state, action, reward, next_state, done)
			state = next_state
			t += 1

		if e % DEBUG_INTERVAL == 0:
			print("episode: {}/{}, t: {}, epsilon: {:.5f}".format(e, NUM_EPISODES, t, agent.epsilon))

		if e % SAVE_INTERVAL == 0:
			save_path = SAVE_PATH + ENV_NAME + "-" + str(e)
			agent.save(save_path)
			print('Successfully saved: ' + save_path)


