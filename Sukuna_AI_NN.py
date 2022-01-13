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
tickers = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'SUSHIUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT', 'IOTAUSDT', 'MATICUSDT', 'ATOMUSDT', 'LINKUSDT', 'NEARUSDT']
current_price = 0.0000
def current_price_func(msg):
    global current_price
    current_price=msg

def read_history(tf, currency):
    while True:
        try:
            os.chdir(path+"/"+currency+"/"+tf)
            market_data = np.empty(shape = [0,12])
            start_time = 0
            for file in os.listdir():
                market_data_tmp = np.empty(shape = [0,12])
                file_path = f"{path}/{currency}/{tf}/{file}"
                f = pd.read_csv(file_path, index_col=0, header = None)
                with open(file_path, newline='') as f:
                    reader = csv.reader(f)
                    lst = list(reader)
                    for line in lst:
                        string_list = line[:-1]
                        start_time=int(line[0])/1000
                        tmp = np.array([[float(val) for val in line]])
                        market_data_tmp = np.append(market_data_tmp,tmp, axis = 0)
                market_data = np.append(market_data,market_data_tmp, axis = 0)
            time_ = 0
            if tf == "5m":
                time_ = 5
            if tf == "15m":
                time_ = 15
            if tf == "1h":
                time_ = 60
            if tf == "30m":
                time_ = 30
            if tf == "1d":
                time_ = 24*60
            start_time += time_*60
            dt = datetime.datetime.now(timezone.utc)
            utc_time = dt.replace(tzinfo=timezone.utc)
            utc_timestamp = utc_time.timestamp()
            # print(round(utc_timestamp))
            set_count = int(math.ceil((round(utc_timestamp)-start_time)/(time_ * 60 * 1000)))
            for i in range(set_count):
                market_data_tmp = np.empty(shape = [0,12])
                req = requests.get(url = binance_api_url+"klines", params = {'symbol':currency, 'interval' : tf, 'limit' : str(1000), 'startTime' : str(int(start_time) * 1000 + i * time_ * 60 * 1000 * 1000)})
                
                klines = json.loads(req.content)
                for candle in klines:
                    market_data_tmp = np.append(market_data_tmp,[np.array([float(val) for val in candle])], axis = 0)
                market_data = np.append(market_data,market_data_tmp, axis = 0)
        except:
            time.sleep(1);
            continue
        break
    return market_data[:-1]


def save_data(data, name):
    json.dump(data.tolist(), codecs.open('D:/Sukuna AI/Data/' + name, 'a', encoding='utf-8'), sort_keys=True, indent=4)

def load_data(name):
    return json.load(codecs.open('D:/Sukuna AI/Data/' + name, 'r', encoding='utf-8'))

def z_trans(x):
    temp_arr=np.append([val[4] for val in x],[val1[5] for val1 in x], axis = 0)
    temp_arr = np.reshape(temp_arr,(2,-1))
    temp_arr = stats.zscore(temp_arr, axis = 1)
    #temp_arr = np.reshape(temp_arr,(-1,1))
    temp_arr = np.append(temp_arr[0][-12:], temp_arr[1][-12:], axis = 0)
    return temp_arr


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
for ep in range (1000):
    state = merge_state_action([states[0]],0)
    environment = env.TradingEnv()
    agent = Agent.Agent(True)
    steps = 0
    for i in np.arange(1, len(states)-1):
        action = agent.make_action(state, i)
        o = market_data[i-1]
        c = market_data[i]
        old_state = state
        actions, rewards, new_states, state, done = environment.step([states[i]],o,c,i, action-1)
        if done == True:
            steps = i
            break
        agent.store(old_state, actions, new_states, rewards, action, i)
        agent.optimize(i)

    f = open ('D:/Sukuna AI/Results/log.txt', 'a')
    f.write(str(steps) + '    ' + str(environment.balance_history[-1])+'\n')
    f.close()