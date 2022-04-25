""" simulator for channels"""

from config import Config
import functions as f
import numpy as np

# 非法窃听者和基站之间的信道
class LegalChannel:
    def __init__(self, bs, ue):
        """ establish a channel given a BS and a EVAs """

        self.bs = bs
        self.ue = ue
        self.index = np.array([bs.index, ue.index])

        self.norminal_aod = f.get_azimuth(bs.location, ue.location)
        self.angular_spread = Config().angular_spread
        self.multi_paths = Config().multi_paths
        self.rho = Config().rho