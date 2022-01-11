import numpy as np
import csv
import json, codecs
import scipy.stats as stats
import os
import math
import pandas as pd
from datetime import timezone
import datetime
import time
import requests
from collections import OrderedDict, deque

path = r"D:\Sukuna AI\Candles"
binance_api_url = "https://fapi.binance.com/fapi/v1/"
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
            if tf == '30m':
                time_ = 15
            if tf == "1h":
                time_ = 60
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
    json.dump(data, codecs.open('D:/Sukuna AI/Data/' + name, 'a', encoding='utf-8'), sort_keys=True, indent=4)

def load_data(name):
    return json.load(codecs.open('D:/Sukuna AI/Data/' + name, 'r', encoding='utf-8'))

def z_trans(x):
    temp_arr= x
    temp_arr = stats.zscore(temp_arr)
    #temp_arr = np.reshape(temp_arr,(-1,1))
    temp_arr = (temp_arr[-8:])
    return temp_arr

def calc_z_scores_parameters(cluster):
    cluster0 = np.asarray(cluster)
    mean = np.mean(cluster0, axis=0)
    variance = np.var(cluster0, axis=0)
    return mean, variance

def z_transform(value, mean, variance):
    result = (np.asarray(value) - mean) / variance
    return result.tolist()
market_data = read_history('5m', 'ETHUSDT')

n = 8
log_returns = []
vol_returns = []
lag = deque()
vol = deque()
last_price = None
last_v = None
close_data = []
for v in market_data:
    if last_price is not None:
        lag.append(math.log(v[4] / last_price))
        if v[5]!=0:
            vol.append(math.log(v[5] / last_v))
        else:
            vol.append(0)
        while len(lag) > n:
            lag.popleft()
            vol.popleft()
        if len(lag) == n:
            log_returns.append(list(lag))
            close_data.append(v[4])
            vol_returns.append(list(vol))
    last_price = v[4]
    if v[5] != 0:
        last_v = v[5]

# print(log_returns[0:5])

# z-score normalization, group 96 states into a cluster
z_score_clusters = OrderedDict()
z_score_vol = OrderedDict()
for n, t_vs in enumerate(log_returns):
    i = n // 96
    if i not in z_score_clusters:
        z_score_clusters[i] = []
    z_score_clusters[i].append(t_vs)

for n, t_vs in enumerate(vol_returns):
    i = n // 96
    if i not in z_score_vol:
        z_score_vol[i] = []
    z_score_vol[i].append(t_vs)

z_score_transformed = []
z_score_t_vol = []
for n, t_vs in enumerate(log_returns):
    i = n // 96
    mean, variance = calc_z_scores_parameters(z_score_clusters[i])
    z_score_transformed.append(z_transform(t_vs, mean, variance))

for n, t_vs in enumerate(vol_returns):
    i = n // 96
    mean, variance = calc_z_scores_parameters(z_score_vol[i])
    z_score_t_vol.append(z_transform(t_vs, mean, variance))

return_list = [np.append(a,b).tolist() for a,b in zip(z_score_t_vol,z_score_transformed)]

save_data(return_list, 'z_score_log_ethusdt_5m_with_v.txt')
save_data(close_data, 'close_ethusdt_5m_with_v.txt')
#f = open ('D:/Sukuna AI/Data/' + 'ethusdt_states_1h_without_vol.txt', 'a')
#f.write('[')
#f.close()
#for i in np.arange(96,len(market_data)-1):
#    print(len(market_data)-3-i)
#    states_arr = np.empty(shape = [0,24])
#    next_state = np.empty(shape = [0,24])
    
#    states_arr = z_trans(market_data[i-96:i])
#    #next_state = next_state.flatten()
#    #next_state = np.reshape(next_state,(-1,1))
#    save_data(states_arr, 'ethusdt_states_1h_without_vol.txt')
#    #save_data(next_state, 'ethusdt_next_states.txt')
#    f = open ('D:/Sukuna AI/Data/' + 'ethusdt_states_1h_without_vol.txt', 'a')
#    if i != len(market_data)-2:
#        f.write(',')
#    f.close()
#f = open ('D:/Sukuna AI/Data/' + 'ethusdt_states_1h_without_vol.txt', 'a')
#f.write(']')
#f.close()
