""" 模拟窃听者 """

from config import Config
import numpy as np


class Eva:

    def __init__(self, bs):
        """ initialize the attributes of a user eavesdroppers """

        c = Config()

        self.azimuth = np.random.rand() * 2 * np.pi # 随机生成方位角
        self.bs = bs
        self.index = bs.index
        # 随机生成和基站之间的距离
        self.distance_to_bs = np.random.rand() * (c.cell_radius - c.inner_cell_radius) + c.inner_cell_radius
        # 根据极坐标生成各个窃听者的坐标。
        self.location = bs.location + self.distance_to_bs * np.array([np.cos(self.azimuth), np.sin(self.azimuth)])