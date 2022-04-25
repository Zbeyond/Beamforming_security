""" 模拟合法信道 """

from config import Config
import functions as f
import numpy as np

# 合法用户和基站之间的信道
class LegalChannel:

    def __init__(self, bs, ue):

        self.bs = bs
        self.ue = ue
        self.index = np.array([bs.index, ue.index])

        self.norminal_aod = f.get_azimuth(bs.location, ue.location) # 合法用户和基站之间的方位角
        self.angular_spread = Config().angular_spread # 散射角
        self.multi_paths = Config().multi_paths # 径数
        self.rho = Config().rho # 频道相关系数

        self.d = np.linalg.norm(self.bs.location - self.ue.location)
        self.path_loss = 1 / f.dB2num(120.9 + 37.6 * np.log10(self.d / 1000) + np.random.normal(0, 8)) #路径损耗
        self._check_is_link_()
        self._generate_steering_vector_() # 生成导向矢量，这些导向矢量组成响应矩阵
        self.g = (np.random.randn(1, self.multi_paths) + np.random.randn(1, self.multi_paths) * 1j) / np.sqrt(2 * self.multi_paths)
        self._cal_csi_(ir_change=True)
        # 存储历史csi信息值
        self.h1, self.h2 = None, None
        self.r_power10, self.r_power11, self.r_power20, self.r_power21 = None, None, None, None
        self.gain10, self.gain11, self.gain20, self.gain21 = None, None, None, None

    def _generate_steering_vector_(self):
        self.aod = self.norminal_aod + (np.random.rand(self.multi_paths) - 0.5) * self.angular_spread #用户和基站之间的方位角
        self.sv = np.zeros((self.multi_paths, self.bs.n_antennas), dtype=complex)
        for i in range(self.multi_paths):
            self.sv[i, :] = np.exp(1j * np.pi * np.cos(self.aod[i]) * np.arange(self.bs.n_antennas)) \
                              / np.sqrt(self.bs.n_antennas)

    def _cal_csi_(self, ir_change):
        "计算信号在每条传输路径上的信道因子，即信道增益矩阵H中每个元素的值"
        if ir_change:
            self.h = np.matmul(self.g, self.sv) #矩阵乘法
            self.H = self.h.reshape((3, )) * np.sqrt(self.path_loss) # 信道矩阵

        self.gain = self.path_loss * np.square(np.linalg.norm(np.matmul(self.h, self.bs.code))) #信道增益
        self.r_power = self.bs.power * self.gain 

    def _check_is_link_(self):
        """ determine whether the channel is a direct link for data transmission or a interference channel, then
        initialize some extra attributes for the direct link. 
        确定该通道是数据传输的直连通道还是干扰通道，然后为直连链路初始化一些额外的属性"""

        if self.bs.index == self.ue.index: #直连通道
            self.is_link = True
            self.utility, self.utility10, self.utility11, self.utility20, self.utility21 = None, None, None, None, None
            self.SINR, self.SINR10, self.SINR11, self.SINR20, self.SINR21 = None, None, None, None, None
            self.IN, self.IN10, self.IN11, self.IN20, self.IN21 = None, None, None, None, None
            self.interferer_neighbors, self.interferer_neighbors10, self.interferer_neighbors11, \
            self.interferer_neighbors20, self.interferer_neighbors21 = None, None, None, None, None
            self.interfered_neighbors, self.interfered_neighbors10, self.interfered_neighbors11, \
            self.interfered_neighbors20, self.interfered_neighbors21 = None, None, None, None, None
        else:
            self.is_link = False #干扰通道

    def _save_csi_(self, ir_change):
        """ save historical CSI """

        if ir_change:

            self.h2 = self.h1
            self.h1 = self.h

            self.r_power21 = self.r_power11
            self.r_power11 = self.r_power

            self.gain21 = self.gain11
            self.gain11 = self.gain

            if self.is_link:
                self.IN21 = self.IN11
                self.IN11 = self.IN

                self.SINR21 = self.SINR11
                self.SINR11 = self.SINR

                self.utility21 = self.utility11
                self.utility11 = self.utility

                self.interferer_neighbors21 = self.interferer_neighbors11
                self.interferer_neighbors11 = self.interferer_neighbors

                self.interfered_neighbors21 = self.interfered_neighbors11
                self.interfered_neighbors11 = self.interfered_neighbors
        else:

            self.r_power20 = self.r_power10
            self.r_power10 = self.r_power

            self.gain20 = self.gain10
            self.gain10 = self.gain

            if self.is_link:
                self.IN20 = self.IN10
                self.IN10 = self.IN

                self.SINR20 = self.SINR10
                self.SINR10 = self.SINR

                self.utility20 = self.utility10
                self.utility10 = self.utility

                self.interferer_neighbors20 = self.interferer_neighbors10
                self.interferer_neighbors10 = self.interferer_neighbors

                self.interfered_neighbors20 = self.interfered_neighbors10
                self.interfered_neighbors10 = self.interfered_neighbors

    def update(self, ir_change):
        """ 当CSI由于信道块衰落或波束更新而发生变化时，保存历史CSI并计算新的CSI。 
         参数'ir_change'是一个bool变量，表示CSI由于信道衰落或n而发生变化。  """
        self._save_csi_(ir_change)
        if ir_change:
            # Fading
            e = (np.random.randn(1, self.multi_paths) + np.random.randn(1, self.multi_paths) * 1j) \
                * np.sqrt(1 - np.square(self.rho)) / np.sqrt(2)
            self.g = self.rho * self.g + e

        self._cal_csi_(ir_change)
        