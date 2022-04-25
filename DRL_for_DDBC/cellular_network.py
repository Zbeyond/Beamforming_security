""" 定义整个通信网络 """

from base_station import BaseStation as BS
from user_equipment import UserEquipment as UE
from eva import Eva as EVA
from config import Config
from legal_channel import LegalChannel
from illegal_channel import illegalChannel
import functions as f
import numpy as np
import matplotlib.pyplot as plt
import operator
import random


class CellularNetwork:

    def __init__(self):
        """ initialize the whole communication network """
        self.config = Config()
        self._generate_bs_()
        self._generate_ue_()
        self._generate_eva_()
        self._establish_legal_channels_()
        self._establish_illegal_channels_()
        self._reset_()

    def _generate_bs_(self):
        """ generate the BSs given the generated locations """
        """ 确定基站的坐标，以极坐标的方式生成各个基站的坐标，存入列表里"""
        self.bs_list = []
        r = self.config.cell_radius
        bs_locations = [[0, 0]] #基准坐标，即坐标系的原点
        theta_1 = 2 * np.pi / 6 * np.arange(0, 6) #起点为0终点为6，步长取默认值1，创建等差数组
        theta_2 = 2 * np.pi / 6 * np.arange(0, 6) - np.pi / 6
        r1 = 2 * r
        r21 = 4 * r
        r22 = 4 * r / 2 * np.sqrt(3)
        #升维 -> 转置 -> 将矩阵转化为列表
        positions = r1 * np.vstack((np.cos(theta_2), np.sin(theta_2))).transpose()
        positions = positions.tolist()
        bs_locations += positions
        positions = r21 * np.vstack((np.cos(theta_2), np.sin(theta_2))).transpose()
        positions = positions.tolist()
        bs_locations += positions
        positions = r22 * np.vstack((np.cos(theta_1), np.sin(theta_1))).transpose()
        positions = positions.tolist()
        bs_locations += positions
        bs_locations = np.array(bs_locations)
        self.bs_locations = bs_locations
        for index in range(bs_locations.shape[0]):
            self.bs_list.append(BS(bs_locations[index, :], index))

    def _generate_ue_(self):
        """ 生成合法用户并存入列表 """
        self.ue_list = []
        for bs in self.bs_list:
            self.ue_list.append(UE(bs))

    def _establish_legal_channels_(self):
        """ establish the channels (direct link and interference channel) between every pair of BS and UE """
        """ 生成合法用户和基站之间的信道并存入列表 """
        self.channels = []
        for bs in self.bs_list:
            for ue in self.ue_list:
                self.channels.append(LegalChannel(bs, ue))
        self._get_links_()

    def _generate_eva_(self):
        """ 生成非法用户并存入列表 """
        self.evalist = []
        for bs in self.bs_list:
            self.evalist.append(EVA(bs))
    
    def _establish_illegal_channels_(self):
        """ establish the channels (direct link and interference channel) between every pair of BS and EVA """
        """ 生成非法用户和基站之间的信道并存入列表 """
        self.illegal_channels = []
        for bs in self.bs_list:
            for eva in self.evalist:
                self.illegal_channels.append(illegalChannel(bs,eva))
        self._get_links_()
        
    def _get_links_(self):
        """ get the link set """
        self.links = []
        for channel in self.channels:
            if channel.is_link:
                self.links.append(channel)

    def get_channel_list(self, bs_index=None, ue_index=None):
        """ 对于系统中的每一对基站和用户，从一堆信道中选择直接连接的，排除非直接连接的信道 """
        channel_list = []

        if bs_index is not None and ue_index is None:
            for channel in self.channels:
                if bs_index == channel.bs.index:
                    channel_list.append(channel)
        elif bs_index is None and ue_index is not None:
            for channel in self.channels:
                if ue_index == channel.ue.index:
                    channel_list.append(channel)
        elif bs_index is not None and ue_index is not None:
            for channel in self.channels:
                if bs_index == channel.bs.index and ue_index == channel.ue.index:
                    return channel

        return channel_list

    def get_link(self, link_index):
        """ Search for the direct link that meets the given conditions """
        for link in self.links:
            if link.ue.index == link_index:
                return link

    def get_link_interferers(self, link):
        """ get the set of all the interferers """
        interferers = []
        channels = self.get_channel_list(ue_index=link.ue.index)
        for channel in channels:
            if not channel.is_link:
                interferers.append(channel)
        return interferers

    def get_interferer_neighbors(self, link):
        """ get the set of the interferers given the cardinality constraint, i.e., U """
        i = []
        channels = self.get_link_interferers(link)
        cmpfun = operator.attrgetter('r_power')
        channels.sort(key=cmpfun, reverse=True)
        for channel in channels[0:self.config.U]:
            i.append(channel.bs.index)
        return np.array(i)

    def get_interfered_neighbors(self, link):
        """ get the set of the interfered neighbors given the cardinality constraint, i.e., U """
        o = []
        interfered = []
        channels = self.get_channel_list(bs_index=link.bs.index)
        for channel in channels:
            if not channel.is_link:
                interfered.append(channel)
        cmpfun = operator.attrgetter('r_power')
        interfered.sort(key=cmpfun, reverse=True)
        for channel in interfered[0:self.config.U]:
            o.append(channel.ue.index)
        return np.array(o)

    def _evaluate_link_performance_(self):
        """ evaluate the performance of the direct link """
        for link in self.links:
            IN = self.config.noise_power
            interferers = self.get_link_interferers(link)
            for interferer in interferers:
                IN += interferer.r_power

            link.IN = IN
            link.SINR = link.r_power / link.IN
            link.utility = np.log2(1 + link.SINR) 
            link.interferer_neighbors = self.get_interferer_neighbors(link)
            link.interfered_neighbors = self.get_interfered_neighbors(link)

    def update(self, ir_change, actions=None, weights=None):
        """ update the cellular network status due to channel fading or beamformers update"""
        """ 在信道特征或者波束成形矩阵发生变化后，需要更新网络的状态 """
        if ir_change:
            for channel in self.channels:
                channel.update(ir_change)
        else:
            if actions is not None:
                self._take_actions_(actions=actions)
            if weights is not None:
                self._take_actions_(weights=weights)
            for channel in self.channels:
                channel.update(ir_change)
        self._evaluate_link_performance_()

    def random_choose_actions(self): # 用于随机选择算法
        """ random take actions"""
        actions = []
        for _ in range(self.config.n_links):
            actions.append(random.randint(0, self.config.n_actions - 1))
        return np.array(actions)

    def _take_actions_(self, actions=None, weights=None):
        """ BSs take the given actions"""
        if actions is not None:
            for index in range(actions.shape[0]):
                self.bs_list[index].take_action(action=actions[index])
        if weights is not None:
            for index in range(weights.shape[1]):
                self.bs_list[index].take_action(weight=weights[:, index])

    def _reset_(self):
        """ reset the cellular network to guarantee the channel variations are the same in different schemes"""
        for _ in range(10):
            actions = self.random_choose_actions()
            self.update(ir_change=False, actions=actions)
            self.update(ir_change=True)

    def observe(self):
        """ obtain the states of the BSs"""
        # normalization factors for the elements in states
        n_r_power = 1e-9
        n_gain = 1e-9
        n_IN = 1e-7
        power_max = f.dB2num(self.config.bs_power)
        n_links = self.config.n_links - 1
        n_ulitity = 5

        observations = []
        for link in self.links:
            local_information, interferer_information, interfered_information = [], [], []

            local_information = np.hstack((link.bs.power / power_max, link.bs.code_index,
                                           link.utility11 / n_ulitity,
                                           link.gain / n_gain, link.gain10 / n_gain,
                                           link.IN / n_IN, link.IN10 / n_IN)).tolist()

            for link_index in link.interferer_neighbors11:
                channel = self.get_channel_list(bs_index=link_index, ue_index=link.ue.index)
                interferer_information.append(channel.bs.index / n_links)
                interferer_information.append(channel.r_power11 / n_r_power)
                interferer_information.append(channel.bs.code_index)
                interferer_information.append(self.get_link(link_index).utility11 / n_ulitity)

            for link_index in link.interferer_neighbors21:
                channel = self.get_channel_list(bs_index=link_index, ue_index=link.ue.index)
                interferer_information.append(channel.bs.index / n_links)
                interferer_information.append(channel.r_power21 / n_r_power)
                interferer_information.append(channel.bs.code_index1)
                interferer_information.append(self.get_link(link_index).utility21 / n_ulitity)

            for link_index in link.interfered_neighbors11:
                channel = self.get_channel_list(bs_index=link.bs.index, ue_index=link_index)
                interfered_information.append(channel.gain11 / n_gain)
                interfered_information.append(self.get_link(link_index).utility11 / n_ulitity)
                interfered_information.append(channel.r_power11 / self.get_link(link_index).IN11)

            observation = local_information + interferer_information + interfered_information
            observations.append(observation)
        return np.array(observations)

    def give_rewards(self):
        """ calculated the rewards of all the BSs"""
        rewards = []
        for link in self.links:
            penalty = 0
            for link_index in link.interfered_neighbors11:
                interfered_link = self.get_link(link_index)
                channel = self.get_channel_list(bs_index=link.bs.index, ue_index=link_index)
                penalty += -interfered_link.utility11 + \
                           np.log2(1 + interfered_link.r_power11 / (interfered_link.IN11 - channel.r_power11))
            reward = link.utility11 - penalty
            rewards.append(reward)

        return np.array(rewards)

    def save_transitions(self, s, a, r, s_):
        """ save the experience of each BS"""
        for bs in self.bs_list:
            i = bs.index
            bs.dqn.save_transition(s[i, :], a[i], r[i], s_[i, :])

    def train_dqns(self):
        """ train the DQN of each BS"""
        for bs in self.bs_list:
            bs.dqn.learn()

    def choose_actions(self, s):
        """ choose actions with the DTDE DRL-based scheme """
        actions = []
        for bs in self.bs_list:
            actions.append(bs.dqn.choose_action(s[bs.index, :]))
        return np.array(actions)

    def draw_topology(self):
        """ 画网络拓扑图 """
        x, y = [], []
        for bs in self.bs_list:
            x.append(bs.location[0])
            y.append(bs.location[1])
        plt.scatter(x, y, marker='^', c='red', s=100, edgecolor=None, label='BS')

        x, y = [], []
        for ue in self.ue_list:
            x.append(ue.location[0])
            y.append(ue.location[1])
        plt.scatter(x, y, c='blue', s=10, edgecolor=None, label='User')
        
        x, y = [], []
        for eva in self.evalist:
            x.append(eva.location[0])
            y.append(eva.location[1])
        plt.scatter(x, y, c='green', s=10, edgecolor=None, label='Eav')
        
        plt.legend(loc=0)
        plt.show()

    def get_ave_utility(self):
        """ calculate the average throughput of all the direct links"""
        s = 0
        for link in self.links:
            s += link.utility

        return s / self.config.n_links

    def get_all_rates(self):
        rates = []
        for link in self.links:
            rates.append(link.utility)
        return rates

    def get_H(self):
        """ a function of FP approach, to get the global CSI of the cellular network"""
        M = self.config.n_links
        K = self.config.n_antennas
        H = np.zeros((M, M, K), dtype=np.complex)
        for i in range(M):
            for j in range(M):
                H[i, j, :] = self.get_channel_list(bs_index=i, ue_index=j).H

        return H
