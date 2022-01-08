from memory import ReplayMemory, Transition
from nn_model import DQN

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as weight_init
import os.path
import copy
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Agent:
	def __init__(self):
		self.state_size = 24+3;
		self.memory = ReplayMemory(2000)
		self.T = 96
		self.action_size = 3
		self.batch_size = 16
		self.gamma = 0.99
		self.epsilon = 0.1
		self.q_network=DQN(self.state_size).to(device)
		self.target_net= DQN(self.state_size).to(device)
		self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.00025)
		#self.q_network.model.load_weights("D:/Sukuna AI/Model/q_nn")
		#self.target_net.model.load_weights("D:/Sukuna AI/Model/target_nn")
		for param_p in self.q_network.parameters(): 
				weight_init.normal_(param_p)

	def store(self, state, action, new_state, reward):
		self.memory.push(state, action, new_state, reward)

	def make_action(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		tensor = torch.FloatTensor(state).to(device)
		tensor = tensor.unsqueeze(0)
		options = self.target_net(tensor)
		return (np.argmax(options[-1].detach().cpu().numpy()))
	def optimize(self, step):
		if self.memory.__len__() < self.batch_size*10:
			return
		transitions = self.memory.sample(self.batch_size)
		# Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
		# detailed explanation). This converts batch-array of Transitions
		# to Transition of batch-arrays.
		batch = Transition(*zip(*transitions))

		# Compute a mask of non-final states and concatenate the batch elements
		# (a final state would've been the one after which simulation ended)
		next_state = torch.FloatTensor(batch.next_state).to(device)
		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state)))
		non_final_next_states = torch.cat([s for s in next_state if s is not None])

		state_batch = torch.FloatTensor(batch.state).to(device)
		action_batch = torch.LongTensor(batch.action).to(device)
		reward_batch = torch.FloatTensor(batch.reward).to(device)

		# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
		# columns of actions taken. These are the actions which would've been taken
		# for each batch state according to q_network
		l = self.q_network(state_batch).size(0)
		state_action_values = self.q_network(state_batch)[95:l:96].gather(1, action_batch.reshape((self.batch_size, 1)))
		state_action_values = state_action_values.squeeze(-1)

		# Compute V(s_{t+1}) for all next states.
		# Expected values of actions for non_final_next_states are computed based
		# on the "older" target_net; selecting their best reward with max(1)[0].
		# This is merged based on the mask, such that we'll have either the expected
		# state value or 0 in case the state was final.
		next_state_values = torch.zeros(self.batch_size, device=device)
		next_state_values[non_final_mask] = self.target_net(next_state)[95:l:96].max(1)[0].detach()
		# Compute the expected Q values
		expected_state_action_values = (next_state_values * self.gamma) + reward_batch

		# Compute the loss
		loss = torch.nn.MSELoss()(expected_state_action_values, state_action_values)

		# Optimize the model
		
		loss.backward()
		
		for param in self.q_network.parameters():
				param.grad.data.clamp_(-1, 1)
		
		self.optimizer.step()
		
		if step % self.T == 0:
			# print('soft_update')
			gamma = 0.001
			target_update = copy.deepcopy(self.target_net.state_dict())
			for k in target_update.keys():
				target_update[k] = self.target_net.state_dict()[k] * (1 - gamma) + self.q_network.state_dict()[k] * gamma
			self.target_net.load_state_dict(target_update)
