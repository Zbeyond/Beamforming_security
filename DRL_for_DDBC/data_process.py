""" visualize the simulation results """

import json
import matplotlib.pyplot as plt
import numpy as np

# 画最后的结果图
def data_visualization():
    window = 500
    # 将训练的结果数据画出来
    filename = 'data/drl_performance.json'
    r = []
    with open(filename, 'r') as f:
        data = np.array(json.load(f))
        for i in range(len(data) - window + 1):
            r.append(np.mean(data[i:i + window]))

    r = np.array(r)
    plt.plot(r, label='DQN Solution')

    filename = 'data/fp_performance.json'

    r = []
    with open(filename, 'r') as f:
        data = np.array(json.load(f))
        for i in range(len(data) - window + 1):
            r.append(np.mean(data[i:i + window]))
    r = np.array(r)
    # plt.plot(r, label='ideal Solution')

    filename = 'data/greedy_performance.json'
    r = []
    with open(filename, 'r') as f:
        data = np.array(json.load(f))
        for i in range(len(data) - window + 1):
            r.append(np.mean(data[i:i + window]))
    r = np.array(r)
    # plt.plot(r, label='greedy')

    filename = 'data/random_performance.json'
    r = []
    with open(filename, 'r') as f:
        data = np.array(json.load(f))
        for i in range(len(data) - window + 1):
            r.append(np.mean(data[i:i + window]))
    r = np.array(r)
    plt.plot(r, label='random')

    plt.xlabel('number of time slots')
    # plt.ylabel('average achievable spectrum efficiency (bps/Hz)')
    plt.ylabel('The system security capacity')
    plt.legend(loc=0)
    plt.grid(True)
    plt.show()


data_visualization()
