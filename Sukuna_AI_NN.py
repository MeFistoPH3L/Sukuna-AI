import torch
import os
import numpy as np
import pandas as pd
from datetime import timezone
import datetime
import time
from sortedcontainers import SortedList
import requests
import math
import json
import csv
import binance
from requests.structures import CaseInsensitiveDict
import random
import Agent
import env
import scipy.stats as stats
from numpy import savetxt
import json, codecs
from copy import deepcopy

path = r"D:\Sukuna AI\Candles"
binance_api_url = "https://fapi.binance.com/fapi/v1/"

def save_data(data, name):
    json.dump(data.tolist(), codecs.open('D:/Sukuna AI/Data/' + name, 'a', encoding='utf-8'), sort_keys=True, indent=4)

def load_data(name):
    return json.load(codecs.open('D:/Sukuna AI/Data/' + name, 'r', encoding='utf-8'))


def hot_encoding(a):
    a_ = np.zeros(3, dtype=np.float32)
    a_[a + 1] = 1.
    return a_

def merge_state_action(state, a_variable):
    T = len(state)
    actions_for_state = []
    actions_for_state.append(a_variable)

    diff = T - len(actions_for_state)
    if diff > 0:
        actions_for_state.extend([a_variable] * diff)

    result = []
    for s, a in zip(state, actions_for_state):
        new_s = deepcopy(s)
        new_s.extend(hot_encoding(a))
        result.append(new_s)

    result = np.asarray(result)
    return result

market_data = load_data('close_ethusdt_5m_8.txt')
"""
f = open ('D:/Sukuna AI/Data/' + 'ethusdt_states.txt', 'a')
f.write('[')
f.close()
for i in np.arange(96,len(market_data)-1):
    print(len(market_data)-3-i)
    states_arr = np.empty(shape = [0,24])
    next_state = np.empty(shape = [0,24])
    
    states_arr = z_trans(market_data[i-95:i])
    #next_state = next_state.flatten()
    #next_state = np.reshape(next_state,(-1,1))
    save_data(states_arr, 'ethusdt_states.txt')
    #save_data(next_state, 'ethusdt_next_states.txt')
    f = open ('D:/Sukuna AI/Data/' + 'ethusdt_states.txt', 'a')
    if i != len(market_data)-2:
        f.write(',')
    f.close()
f = open ('D:/Sukuna AI/Data/' + 'ethusdt_states.txt', 'a')
f.write(']')
f.close()
"""
i = 0   
states = load_data('z_score_log_ethusdt_5m_8.txt')
SEED = 2037
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

t = [1, 0.1, 0.01, 0.001, 0.00025]
cnt = [50, 50, 50, 50, 10000]
for rate, count in zip(t,cnt):
    for ep in range (count):
        trades = 0
        agent = Agent.Agent(rate)
        new_rate = False
        state = merge_state_action([states[0]],0)
        environment = env.TradingEnv()
        steps = len(states)-1
        for i in np.arange(1, 8641):
            action = agent.make_action(state, i)
            trades += abs(action-1)
            o = market_data[i-1]
            c = market_data[i]
            old_state = state
            actions, rewards, new_states, state, done = environment.step([states[i]],o,c,i, action-1)
            steps = i
            if done == True:
                break
            agent.store(old_state, actions, new_states, rewards, action, i)
            agent.optimize(i)

        f = open ('D:/Sukuna AI/Results/log.txt', 'a')
        f.write(str(steps) + '    ' + str(environment.balance_history[-1]) + '    ' + str(trades) + '\n')
        f.close()