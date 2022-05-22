""" 深度强化学习训练过程主函数 """

from cellular_network import CellularNetwork as CN
import json
import random
import numpy as np
import scipy.io as sio
from config import Config
import os
os.environ['MKL_NUM_THREADS'] = '1'


c = Config()
random.seed(c.random_seed)
np.random.seed(c.random_seed)
cn = CN()
utility = []
cn.draw_topology()
rate_m = []
for _ in range(c.total_slots):
    print(_)
    s = cn.observe() #获取当前各个基站的状态，包括当前状态下基站所消耗的功率、上一次迭代过程中选择的波束成形矩阵、产生的保密容量等等。
    actions = cn.choose_actions(s) # 选择当前动作
    cn.update(ir_change=False, actions=actions)
    utility.append(cn.get_ave_utility()) #计算当前动作产生的SREE值
    rate_m.append(cn.get_all_rates())
    cn.update(ir_change=True)
    r = cn.give_rewards() #当前选择动作获得的回报（考虑未来的Q值）
    s_ = cn.observe() # 选择的下一个动作
    cn.save_transitions(s, actions, r, s_)

    if _ > 256:
        cn.train_dqns()


# save data
filename = 'data/drl_performance.json'
with open(filename, 'w') as f:
    json.dump(utility, f)
rate_m = np.array(rate_m)
sio.savemat('rates/drl_rates.mat', {'drl_rates': rate_m})
