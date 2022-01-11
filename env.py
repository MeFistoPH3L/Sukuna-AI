import numpy as np
import Agent
from copy import deepcopy


class TradingEnv:
    def __init__(self):
        self.initial_value = 100000
        self.balance_history = [float(self.initial_value)]
        self.spread = 0.01
        self.position = [0]*3
        self.trade_size = 8000
        self.balance = 100000
        self.actions = [0]
        self.last_action = 0
    @staticmethod
    def hot_encoding(a):
        a_ = np.zeros(3, dtype=np.float32)
        a_[a + 1] = 1.
        return a_

    def merge_state_action(self, state, a_variable):
        T = len(state)
        actions_for_state = []
        actions_for_state.append(a_variable)
        diff = T - len(actions_for_state)
        if diff > 0:
            actions_for_state.extend([a_variable] * diff)

        result = []
        for s, a in zip(state, actions_for_state):
            new_s = deepcopy(s)
            new_s.extend(TradingEnv.hot_encoding(a))
            result.append(new_s)

        result = np.asarray(result)
        return result

    def reset(self):
        self.balance_history = [float(self.initial_value)]
        self.position = 0

    def step(self, state, open_, close, step, action):
        actions = [-1,0,1]
        price_move = close/open_ - 1
        v_old = self.balance_history[-1]
        new_states = []
        for a in actions:
            new_states.append(self.merge_state_action(state, a))
        v_new = []
        for a in actions:
            commission = self.trade_size * np.abs(a - self.actions[-1]) * 0.0002
            v_new.append(v_old + a * self.trade_size * price_move - commission)
        v_new = np.asarray(v_new)
        rewards = []
        for i in range(len(v_new)):
            if v_new[i] * v_old > 0 and v_old != 0:
                rewards.append(np.log(v_new[i] / v_old))
            else:
                rewards.append(-1)
        rewards = np.asarray(rewards) 
        act = ""
        if action == 0:
            if self.actions[-1] == 1:
                act = 'Close Long'
            elif self.actions[-1]==-1:
                act = 'Close Short'
        if action == 1:
            if self.actions[-1] == 0:
                act = 'Open Long'
            elif self.actions[-1]==-1:
                act = 'Close Short and Open Long'
        if action == -1:
            if self.actions[-1] == 0:
                act = 'Open Short'
            elif self.actions[-1]==1:
                act = 'Close Long and Open Short'
        self.actions.pop(0)
        self.actions.append(int(action))
        self.balance_history.append(float(v_new[action+1]))
        while True:
            try:
                f = open("D:/Sukuna AI/Results/ethusdt_log_5m.csv", 'a')
                f.write(f"{int(v_old)},{act},{open_}\n")
                f.close()
            except:
                continue
            break
        print(f"{step} {int(v_new[action+1])} {action}")
        return actions, rewards, new_states, new_states[action+1]
