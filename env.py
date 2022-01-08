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

    def reset(self):
        self.balance_history = [float(self.initial_value)]
        self.position = 0

    def step(self, state, next_state, open_, close, step, end):
        v_old = self.balance_history[-1]
        print(step)

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
        if temp[2] == 0: 
            temp[0] += 1
            reward = -1*price_move*temp[0]*self.trade_size + price_move * temp[2]*self.trade_size - self.trade_size*0.0002
            tmp_states_new[-1][-1]=temp[-1]
            tmp_states_new[-1][-2]=temp[-2]
            tmp_states_new[-1][-3]=temp[-3]
            self.agent.store(tmp_states, 0, tmp_states_new, v_old/(reward+v_old) - 1)
        else:
            temp = [ 0, 0, 0 ]
            reward = -1 * self.trade_size * 0.0002 * self.position[2]
            tmp_states_new[-1][-1]=temp[-1]
            tmp_states_new[-1][-2]=temp[-2]
            tmp_states_new[-1][-3]=temp[-3]
            self.agent.store(tmp_states, 0, tmp_states_new, v_old/(reward+v_old) - 1)

        temp = self.position[:]
        reward = -1*price_move*temp[0] + price_move * temp[2]
        tmp_states_new[-1][-1]=temp[-1]
        tmp_states_new[-1][-2]=temp[-2]
        tmp_states_new[-1][-3]=temp[-3]
        self.agent.store(tmp_states, 1, tmp_states_new, v_old/(reward+v_old) - 1)

        temp = self.position[:]
        if temp[0] == 0: 
            temp[2] += 1
            reward = -1*price_move*temp[0]*self.trade_size + price_move * temp[2]*self.trade_size - self.trade_size*0.0002
            tmp_states_new[-1][-1]=temp[-1]
            tmp_states_new[-1][-2]=temp[-2]
            tmp_states_new[-1][-3]=temp[-3]
            self.agent.store(tmp_states, 2, tmp_states_new, v_old/(reward+v_old) - 1)
        else:
            temp = [ 0, 0, 0 ]
            reward = -1 * self.trade_size * 0.0002 * self.position[0]
            tmp_states_new[-1][-1]=temp[-1]
            tmp_states_new[-1][-2]=temp[-2]
            tmp_states_new[-1][-3]=temp[-3]
            self.agent.store(tmp_states, 2, tmp_states_new, v_old/(reward+v_old) - 1)




        action = self.agent.make_action(tmp_states)
        commision = 0
        if action == 0 and self.position[2] != 0:
            commision = self.position[2]*self.trade_size*0.0002
            self.position[2] = 0
        elif action == 2 and self.position[0] != 0:
            commision = self.position[0]*self.trade_size*0.0002
            self.position[0]=0
        elif action == 1:
            None
        else:
            self.position[action]+=1
            commision = abs(action-1)*self.trade_size*0.02/100
        v_new = v_old + -1*price_move*self.position[0]*self.trade_size + price_move * self.position[2]*self.trade_size - commision
        self.balance_history = np.append(self.balance_history,[v_new])
        if step > 10000:
            path ='D:/Sukuna AI/Results/ethusdt_log.csv'
            f = open(path, 'a')
            f.write(str(self.balance)+',' + str(action-1)+','+str(self.position[0]) + ',' + str(self.position[2])+'\n')
            f.close()
            self.balance += -1*price_move*self.position[0]*self.trade_size + price_move * self.position[2]*self.trade_size - commision
        self.actions.pop(0)
        self.actions.append(self.position)
        self.agent.optimize(step)