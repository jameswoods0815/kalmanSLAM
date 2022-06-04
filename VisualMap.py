import itertools
import numpy as np
import scipy.linalg
from common import *

class EKFMapping:
    def __init__(self,
                 num_landmarks,
                 car_imu_T_cam,
                 car_cam_intrinsic,
                 car_b,
                 imu_noise_conv=None,
                 observation_noise_conv=None,
                 prior_pose_conv=None,
                 prior_landmark_conv=None):

        # landmark x,y,z 3 demension
        if prior_landmark_conv is None:
            prior_landmark_conv= 1e-3 *np.eye(3)
        #pose x,y,z,w1,w1,w3 6 dimension
        if prior_pose_conv is None:
            prior_pose_conv=1e-3*np.eye(6)
        #observation 4 l1,l2,r1,r2
        if observation_noise_conv is None:
            observation_noise_conv=1e-3*np.eye(4)
        #IMU sensing nosie 6 demsion
        if imu_noise_conv is None:
            imm_noise_conv=1e-3 *np.eye(6)

        #robot pose to the world
        self.w_T_car=np.eye(4)
        self.imu_noise_conv=imm_noise_conv
        self.num_landmarks=num_landmarks
        self.n_initlized=0
        self.init_mask=np.zeros((num_landmarks),dtype=bool)
        self.init_maxid=0
        self.mapPosition=np.zeros((num_landmarks,3))
        self.totalObservationSigma=np.kron(np.eye(num_landmarks),prior_landmark_conv)
        self.singleObservationCov= observation_noise_conv
        self.imu_T_cam=car_imu_T_cam
        K=car_cam_intrinsic
        self.car_b=car_b
        self.Ks=np.block([[K[:2, :], np.array([[0, 0]]).T], [K[:2, :], np.array([[-K[0, 0] * self.car_b, 0]]).T]])

    def get_init_max_id(self):
      return  self.init_maxid

    def get_num_init(self):
        return self.n_initlized

    def getPositionFromTransform(self):
        return self.w_T_car[:3,3].reshape(-1,1)

    def world_T_cam(self):
        return self.w_T_car@self.imu_T_cam

    def predict(self, u, tau):
        tmp=scipy.linalg.expm(tau*getSkewMatrix(u))
        self.w_T_car=self.w_T_car@tmp

    #input  4* num_landmarks
    #return a index for data:
    def pick_useful_observe(self, observation):
        return np.array(np.where(observation.sum(axis=0)>-4), dtype=np.int32).reshape(-1)

    # init the landmark:
    #if the id is new, insert to the map:
    # if not, skip:

    def init_landmarks(self,observation,datamask):
        mask=np.invert(self.init_mask[datamask])
        data=datamask[mask]

        if data.size>0:
            wTo=self.world_T_cam()
            self.init_mask[data]=True
            observation=observation[:,data]
            Ks=self.Ks
            b=self.car_b
            worldCoor=np.ones((4,data.size))
            worldCoor[0, :] = (observation[0, :] - Ks[0, 2]) * b / (observation[0, :] - observation[2, :])
            worldCoor[1, :] = (observation[1, :] - Ks[1, 2]) * (-Ks[2, 3]) / (Ks[1, 1] * (observation[0, :] - observation[2, :]))
            worldCoor[2, :] = -Ks[2, 3] / (observation[0, :] - observation[2, :])
            world=wTo@worldCoor

            self.mapPosition[data,:]=world[:3,:].T

            self.n_initlized=np.sum(self.init_mask)
            self.init_maxid=max(data.max()+1, self.init_maxid)
    def get_H(self, datamap):
        num_observe=datamap.size
        n_update=self.init_maxid
        P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
        mapPostion=np.hstack([self.mapPosition[datamap,:],np.ones((num_observe,1))])
        H=np.zeros((num_observe*4,n_update*3))

        cam_T_world=np.linalg.inv(self.world_T_cam())

        for i in range(num_observe):
            tmp=datamap[i]
            H[i*4:(i+1)*4, tmp*3:(tmp+1)*3]=self.Ks@diff_func_pi(cam_T_world@mapPostion[i,:].reshape(-1,1))@cam_T_world@P.T
        return H

    def get_postion_and_map_total_simga(self):
        n_updates=self.init_maxid
        postion=self.mapPosition[:n_updates,:]
        conv=self.totalObservationSigma[:n_updates*3,:n_updates*3]
        return postion, conv

    def get_observation(self,observation, datamap):
        return observation[:,datamap].reshape(-1,1,order='F')

    def get_predict_observation(self, datamap):
        num_observation=datamap.size
        cam_T_world = np.linalg.inv(self.world_T_cam())
        position=np.hstack([self.mapPosition[datamap,:],np.ones((num_observation,1))])
        optical_predict=self.Ks@pi_function(cam_T_world@position.T)
        return optical_predict.reshape(-1,1,order='F')

    def update_value_postion_and_sigma(self,postion,sigma):
        n_updates=self.init_maxid
        self.mapPosition[:n_updates,:]=postion
        self.totalObservationSigma[:n_updates*3,:n_updates*3]=sigma

    def updateMap(self,observation):
        datamap=self.pick_useful_observe(observation)
        if datamap.size>0:
            num_observation=datamap.size
            self.init_landmarks(observation,datamap)

            H=self.get_H(datamap)
            position,sigma=self.get_postion_and_map_total_simga()
            predict_observe=self.get_predict_observation(datamap)

            obsevReal=self.get_observation(observation,datamap)
            V=np.kron(np.eye(num_observation),self.singleObservationCov)
            PHT=sigma@H.T
            Kt=np.linalg.solve((H@PHT+V).T,PHT.T).T

            position+=(Kt@(obsevReal-predict_observe)).reshape(-1,3)
            finalSigma=(np.eye(Kt.shape[0])-Kt@H)@sigma
            self.update_value_postion_and_sigma(position, finalSigma)