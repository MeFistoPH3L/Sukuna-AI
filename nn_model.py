import torch
import torch.nn as nn

class DQN(nn.Module):
	def __init__(self, state_size):
		super(DQN, self).__init__()
		self.first_two_layers = nn.Sequential(
			nn.Linear(state_size, 50),
			nn.ELU(),
			nn.Linear(50, 50),
			nn.ELU()
		)
		self.lstm = nn.LSTM(50, 50, 1, batch_first=True)
		self.last_linear = nn.Linear(50, 3)
		nn.init.constant_(self.first_two_layers[0].bias,0)
		nn.init.constant_(self.first_two_layers[2].bias,0)
		nn.init.constant_(self.last_linear.bias,0)
		self.lstm.bias_hh_l0.data.fill_(0)
		self.lstm.bias_ih_l0.data.fill_(0)
		for names in self.lstm._all_weights:
			for name in filter(lambda n: "bias" in n,  names):
				bias = getattr(self.lstm, name)
				n = bias.size(0)
				start, end = n//4, n//2
				bias.data[start:end].fill_(1.)

# Data Flow Protocol:
# 1. network input shape: (batch_size, seq_length, num_features)
# 2. LSTM output shape: (batch_size, seq_length, hidden_size)
# 3. Linear input shape:  (batch_size * seq_length, hidden_size)
# 4. Linear output: (batch_size * seq_length, out_size)

	def forward(self, input, hs, cs):
		# rint(input.size())

		x = self.first_two_layers(input)
		# print(x.size())
		
		lstm_out, (hs, cs) = self.lstm(x, (hs,cs))
		# print(lstm_out.size())

		batch_size, seq_len, mid_dim = lstm_out.shape
		linear_in = lstm_out.contiguous().view(seq_len * batch_size, mid_dim)
		# linear_in = lstm_out.contiguous().view(-1, lstm_out.size(2))

		# linear_in = lstm_out.reshape(-1, hidden_size) 
		return self.last_linear(linear_in), hs, cs