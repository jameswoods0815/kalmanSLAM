import itertools
import numpy as np
import scipy.linalg
from common import *

class predictIMU:
    def __init__(self,
        prior_pose_conv,
        sensor_noise_conv):
        if prior_pose_conv is None:
            prior_pose_conv=1e-3 *np.eye(6)

        self.T=np.eye(4)
        self.sensor_noise_conv=sensor_noise_conv
        self.Sigma=prior_pose_conv

    def predict(self,controlInput,tau):
        tmp=scipy.linalg.expm(tau*getSkewMatrix(controlInput))
        self.T=self.T@tmp

        # for conv
        tmp1=scipy.linalg.expm(-tau*getSkew_6x6(controlInput))
        self.Sigma=tmp1@self.Sigma
        self.Sigma=self.Sigma@tmp1.T+self.sensor_noise_conv





