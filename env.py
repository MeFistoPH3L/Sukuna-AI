import numpy as np
import Agent

class TradingEnv:
    def __init__(self, agent: Agent.Agent):
        self.initial_value = 100000
        self.balance_history = [float(self.initial_value)]
        self.spread = 0.01
        self.position = [0]*3
        self.trade_size = 8000
        self.agent = agent
        self.balance = 100000
        self.actions = [[0,0,0]]*96
        self.last_action = 0

    def reset(self):
        self.balance_history = [float(self.initial_value)]
        self.position = 0

    def step(self, state, next_state, open_, close, step, end):
        v_old = self.balance_history[-1]
        if end == True:
            return
        tmp_states = np.empty(shape = [0,27])
        tmp_states_new = np.empty(shape = [0,27])
        for i in range(len(state)):
            tmp_states = np.append(tmp_states,[np.append(state[i],self.actions[i])], axis = 0)
            if i != len(state)-1:
                tmp_states_new = np.append(tmp_states_new,[np.append(next_state[i],self.actions[i+1])], axis = 0)
            else:
                tmp_states_new = np.append(tmp_states_new,[np.append(next_state[i],[0,0,0])], axis = 0)

        temp = self.position[:]
        price_move = close/open_ - 1

        v_new = v_old + self.trade_size*price_move - self.trade_size * 0.0002 * abs(1-self.last_action)
        tmp_states_new[-1][-3]=0
        tmp_states_new[-1][-2]=0
        tmp_states_new[-1][-1]=1
        self.agent.store(tmp_states, 2, tmp_states_new, v_new/v_old)

        v_new = v_old + -1*self.trade_size*price_move - self.trade_size * 0.0002 * abs(-1-self.last_action)
        tmp_states_new[-1][-3]=1
        tmp_states_new[-1][-2]=0
        tmp_states_new[-1][-1]=0
        self.agent.store(tmp_states, 0, tmp_states_new, v_new/v_old)

        v_new = v_old - self.trade_size * 0.0002 * abs(self.last_action)
        tmp_states_new[-1][-1]=0
        tmp_states_new[-1][-2]=0
        tmp_states_new[-1][-3]=0
        self.agent.store(tmp_states, 1, tmp_states_new, v_new/v_old)

        action = self.agent.make_action(tmp_states) - 1
        commision = self.trade_size * 0.0002 * abs(action - self.last_action)
        
        v_new = v_old + action*self.trade_size*price_move-commision
        self.position = [0,0,0]
        if action + 1 != 1:
            self.position[action + 1] = 1
        self.balance_history = np.append(self.balance_history,[v_new])
        if step <= 10000:
            print(f"\n{step}   {v_old}   {action}   {self.position}\n")
        if step > 10000:
            while True:
                try:
                    print(int(self.balance))
                    path ='D:/Sukuna AI/Results/ethusdt_log.csv'
                    f = open(path, 'a')
                    f.write(str(self.balance)+',' + str(action)+','+str(self.position[0]) + ',' + str(self.position[2])+'\n')
                    f.close()
                    self.balance += action*self.trade_size*price_move-commision
                except:
                    continue
                break
        self.actions.pop(0)
        self.actions.append(self.position)
        self.last_action = action
        self.agent.optimize(step)