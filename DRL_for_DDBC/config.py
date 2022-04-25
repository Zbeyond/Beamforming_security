""" 初始化仿真的参数 """

import functions as f
import numpy as np


class Config:

    def __init__(self):
        # 基站
        self.n_antennas = f.get_codebook().shape[0]  # 发射天线数
        self.codebook_size = f.get_codebook().shape[1]  # number of codes 表示选择主瓣方向时有多少个备选项
                                                        # 在[0,2π]之间选择若干个点作为备选的方向
        self.n_power_levels = 5  # number of discrete power levels
        self.n_actions = self.codebook_size * self.n_power_levels   # number of available actions to choose
        self.bs_power = 40  # maximum transmit power of base stations 基站的功率上限
        self.rss_min = 40   # 各个用户的保密容量下限

        # 信道
        self.angular_spread = 3 / 180 * np.pi  # angular spread 由于多径反射、散射，信号在接收天线上的到达角度会展宽
        self.multi_paths = 4  # number of multi-paths 径数
        self.rho = 0.64  # channel correlation coefficient 连续时隙之间的相关系数
        self.noise_power = f.dB2num(-114)  # noise power  高斯白噪声功率

        # 通信网络
        self.cell_radius = 200  # cell radius 整个通信系统的的半径
        self.n_links = 19  # number of simulated direct links in the simulation 仿真中直连信道的数量
        self.inner_cell_radius = 10  # inner cell radius   一个小区的半径

        # 其他仿真参数
        self.slot_interval = 0.02  # 每个时隙的间隔
        self.random_seed = 2000  # random seed to control the simulated cellular network
        self.total_slots = 100000   # 总共的迭代次数
        self.U = 5  # number of neighbors taken into consideration
