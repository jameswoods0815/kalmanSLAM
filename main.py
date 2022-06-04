import numpy as np
from pr3_utils import *
from utils import *
from slam import *
from common import *
from predict import *
from VisualMap import *
from EKFSLAM import *

def testPredictIMU():
	filename = "./data/10.npz"
	t, features, linear_velocity, angular_velocity, K, b, imu_T_cam = load_data(filename)
	ekf=predictIMU(5e-3*np.eye(6),1e-3*np.eye(6))
	u = np.vstack([linear_velocity, angular_velocity])
	num = linear_velocity.shape[-1]
	pose = []
	for i in range(1, num):
		ekf.predict(u[:, i], (t[0, i] - t[0, i - 1]))
		pose.append(ekf.T)
	pose = np.stack(pose, -1) if isinstance(pose, list) else pose
	visualize_trajectory_2d_1(pose, None, show_ori=True)


def testVisualMap():
	filename = "./data/03.npz"
	t, features, linear_velocity, agularVelocity, K, b, imu_T_cam = load_data(filename)
	ekf=EKFMapping(features.shape[1],imu_T_cam,K,b,None,1000*np.eye(4),1e-3*np.eye(6),1e-3*np.eye(3))
	u = np.vstack([linear_velocity, agularVelocity])
	num = linear_velocity.shape[-1]
	pose = []
	for i in range(1,num):
		ekf.predict(u[:,i],(t[0,i]-t[0,i-1]))
		ekf.updateMap(features[:,:,i])
		pose.append(ekf.w_T_car)
		print (i)

		if (i+1)%50==0:
		  tmp = np.stack(pose, -1) if isinstance(pose, list) else pose
		  visualize_trajectory_2d_1(tmp, ekf.mapPosition)

def testEKFslam():
	filename = "./data/03.npz"
	t, features, linear_velocity, agularVelocity, K, b, imu_T_cam = load_data(filename)
	ekf = slam(features.shape[1], imu_T_cam, K, b, 1e-3 *np.eye(6), 100* np.eye(4), 1e-3 * np.eye(6), 1e-3 * np.eye(3))
	u = np.vstack([linear_velocity, agularVelocity])
	Time = linear_velocity.shape[-1]
	pose = []

	for i in range(1, Time):
		ekf.predict(u[:, i], (t[0, i] - t[0, i - 1]))
		ekf.update(features[:, :, i])
		pose.append(ekf.w_T_car)

		print (i)

		if (i+1)%50==0:
		  tmp = np.stack(pose, -1) if isinstance(pose, list) else pose
		  visualize_trajectory_2d_1(tmp, ekf.mapPosition)




def testSLAM():
	filename = "./data/03.npz"
	t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu = load_data(filename)
	ekf = EKFSLAM(
		n_landmarks=features.shape[1],
		robot_cam_T_imu=cam_T_imu,
		robot_cam_intrinsic_calib=K,
		robot_cam_baseline=b,
		observation_noise_covariance=1e-3* np.eye(4),
		prior_landmark_covariance=1e-3 * np.eye(3),
		process_noise_covariance=1e-3 * np.eye(6)
	)

	u = np.vstack([linear_velocity, rotational_velocity])
	T = linear_velocity.shape[-1]
	pose = []

	for i in range(1, T):
		ekf.predict(u[:, i], (t[0, i] - t[0, i - 1]))
		zmap = ekf.update(features[:, :, i])
		pose.append(np.linalg.inv(ekf.xU))

		if i % 1 == 0:
			print(f'------------ {i} ----------------------')
			print('INITIALIZED LANDMARK:::', ekf.n_initialized)

		if (i + 1) % 500== 0:
			visualize_trajectory_2d(np.stack(pose, -1),
									landmarks=ekf.xm,
									initialized=ekf.is_initialized,
									observed=zmap,
									save_fig_name=None,
									xlim=None,
									ylim=None,
									show_navigation=True)

	pose = np.stack(pose, -1) if isinstance(pose, list) else pose
	visualize_trajectory_2d(pose, ekf.xm, show_ori=True)
	pose = np.stack(pose, -1) if isinstance(pose, list) else pose
	fig, ax = visualize_trajectory_2d(pose, ekf.xm)



if __name__ == '__main__':

	# Load the measurements
	#filename = "./data/10.npz"
	#t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
	#testPredict()
	#testVisualMap()
	testEKFslam()
	#testSLAM()


	print(1)

	# (a) IMU Localization via EKF Prediction

	# (b) Landmark Mapping via EKF Update

	# (c) Visual-Inertial SLAM

	# You can use the function below to visualize the robot pose over time
	# visualize_trajectory_2d(world_T_imu, show_ori = True)


