# Sukuna-AI

Financial Trading as a Game:
A Deep Reinforcement Learning Approach

An automatic program that generates constant profit from the financial market is lucrative
for every market practitioner. Recent advance in deep reinforcement learning provides a
framework toward end-to-end training of such trading agent. In this paper, we propose an
Markov Decision Process (MDP) model suitable for the financial trading task and solve it
with the state-of-the-art deep recurrent Q-network (DRQN) algorithm. We propose several
modifications to the existing learning algorithm to make it more suitable under the financial
trading setting, namely 1. We employ a substantially small replay memory (only a few
hundreds in size) compared to ones used in modern deep reinforcement learning algorithms
(often millions in size.) 2. We develop an action augmentation technique to mitigate the
need for random exploration by providing extra feedback signals for all actions to the
agent. This enables us to use greedy policy over the course of learning and shows strong
empirical performance compared to more commonly used -greedy exploration. However,
this technique is specific to financial trading under a few market assumptions. 3. We
sample a longer sequence for recurrent neural network training. A side product of this
mechanism is that we can now train the agent for every T steps. This greatly reduces
training time since the overall computation is down by a factor of T. We combine all of
the above into a complete online learning algorithm and validate our approach on the spot
foreign exchange market.
Keywords: deep reinforcement learning, deep recurrent Q-network, financial trading,
foreign exchange

Based on:
[link_1](https://arxiv.org/pdf/1807.02787.pdf)
[link_2](https://arxiv.org/pdf/1507.06527v4.pdf)
Зависимости:

python-binance

numpy

requests

scipy

pytorch

