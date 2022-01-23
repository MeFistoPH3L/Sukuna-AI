from memory import ReplayMemory, Transition
from nn_model import DQN
from torchinfo import summary
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as weight_init
import os.path
import copy
from shutil import copyfile
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Agent:
	def __init__(self):
		self.state_size = 8+3
		self.memory = ReplayMemory(480)
		self.T = 160
		self.action_size = 3
		self.batch_size = 160
		self.gamma = 0.04
		self.epsilon = 0.4
		if os.path.exists(r'D:\Sukuna AI\Model\target_nn_5m.pth'):
			self.q_network = torch.load(r'D:\Sukuna AI\Model\q_nn_5m.pth')
			self.target_net = torch.load(r'D:\Sukuna AI\Model\target_nn_5m.pth')
			self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.00025)
			self.optimizer.load_state_dict(torch.load(r'D:\Sukuna AI\Model\optimizer_5m.pth'))
		else:
			self.q_network=DQN(self.state_size).to(device)
			self.target_net= DQN(self.state_size).to(device)
			for param_p in self.q_network.parameters(): 
				weight_init.normal_(param_p, 0, 0.001)
			self.target_net.load_state_dict(self.q_network.state_dict())
			self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.00025)
		self.hs = torch.zeros(1, 1, 25)
		self.cs = torch.zeros(1, 1, 25)

		#summary(self.q_network,input_size = (1, 96, 99))
		#summary(self.q_network,input_size = (1, 96, 99))
	def store(self, state, actions, new_states, rewards, action, step, old_hs, old_cs, new_hs, new_cs):
		for n in range(len(actions)):
			self.memory.push(state, actions[n]+1, new_states[n], rewards[n], old_hs.tolist(), old_cs.tolist(), new_hs.tolist(), new_cs.tolist())
	def reset (self):
		self.memory = ReplayMemory(24000)
	def make_action(self, state, step):
		#if np.random.rand() <= self.epsilon:
		#	return random.randrange(self.action_size)
		tensor = torch.FloatTensor(state).to(device)
		tensor = tensor.unsqueeze(0)
		options = self.q_network(tensor, self.hs, self.cs)
		self.hs = options[1]
		self.cs = options[2]
		return (np.argmax(options[0].detach().cpu().numpy()))
	def optimize(self, step):
		if self.memory.__len__() < 480:
			return
		if step % self.T == 0:
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
			old_hs = torch.FloatTensor(batch.old_hs).to(device)
			old_cs = torch.FloatTensor(batch.old_cs).to(device)
			new_hs = torch.FloatTensor(batch.new_hs).to(device)
			new_cs = torch.FloatTensor(batch.new_cs).to(device)
			action = []
			for x, hs, cs in zip(batch.next_state, new_hs, new_cs):
				tensor = torch.FloatTensor(x).to(device)
				tensor = tensor.unsqueeze(0)
				options = self.q_network(tensor,hs,cs)
				action.append((np.argmax(options[0].detach().cpu().numpy())))
			next_state_action = torch.LongTensor(action).to(device)
			# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
			# columns of actions taken. These are the actions which would've been taken
			# for each batch state according to q_network
			state_action_values = []
			for x, hs, cs in zip(batch.state, old_hs, old_cs):
				tensor = torch.FloatTensor(x).to(device)
				tensor = tensor.unsqueeze(0)
				options = self.q_network(tensor,hs,cs)
				state_action_values.append(options[0][0])
			state_action_values = torch.stack(state_action_values).gather(1, action_batch.reshape((self.batch_size, 1)))
			state_action_values = state_action_values.squeeze(-1)

			# Compute V(s_{t+1}) for all next states.
			# Expected values of actions for non_final_next_states are computed based
			# on the "older" target_net; selecting their best reward with max(1)[0].
			# This is merged based on the mask, such that we'll have either the expected
			# state value or 0 in case the state was final.
			next_state_values = []
			for x, hs, cs in zip(batch.next_state, new_hs, new_cs):
				tensor = torch.FloatTensor(x).to(device)
				tensor = tensor.unsqueeze(0)
				options = self.target_net(tensor,hs,cs)
				next_state_values.append(options[0][0])
			next_state_values = torch.stack(next_state_values).gather(1, next_state_action.reshape((self.batch_size, 1)))
			next_state_values = next_state_values.squeeze(-1)
			# Compute the expected Q values
			expected_state_action_values = (next_state_values * self.gamma) + reward_batch
			# Compute the loss
			loss_func = torch.nn.MSELoss()
			loss = loss_func(expected_state_action_values, state_action_values)
			#print(str(loss.item())+'\n')
			self.optimizer.zero_grad()
			# Optimize the model
			loss.backward()
			par = list(self.q_network.parameters())
			#for param in self.q_network.parameters():
			#		param.grad.data.clamp_(-1, 1)
			self.optimizer.step()

		
	# print('soft_update')
		gamma = 0.001
		target_update = copy.deepcopy(self.target_net.state_dict())
		for k in target_update.keys():
			target_update[k] = self.target_net.state_dict()[k] * (1 - gamma) + self.q_network.state_dict()[k] * gamma
		self.target_net.load_state_dict(target_update)

		if os.path.exists(r'D:\Sukuna AI\Model\target_nn_5m.pth'):
			copyfile(r'D:\Sukuna AI\Model\q_nn_5m.pth', r'D:\Sukuna AI\Model\q_nn_5m_copy.pth')
			copyfile(r'D:\Sukuna AI\Model\target_nn_5m.pth', r'D:\Sukuna AI\Model\target_nn_5m_copy.pth')
			copyfile(r'D:\Sukuna AI\Model\optimizer_5m.pth', r'D:\Sukuna AI\Model\optimizer_5m_copy.pth')
		torch.save(self.q_network, r'D:\Sukuna AI\Model\q_nn_5m.pth')
		torch.save(self.target_net, r'D:\Sukuna AI\Model\target_nn_5m.pth')
		torch.save(self.optimizer.state_dict(), r'D:\Sukuna AI\Model\optimizer_5m.pth')