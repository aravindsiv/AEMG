import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import sys
import os

from pendulum import PendulumNoCtrl
from tqdm import tqdm

from models import *
from config import *

env = PendulumNoCtrl()
tf_bounds = env.get_transformed_state_bounds()
gt_bounds = env.state_bounds

MODEL_PATH = f'{ROOT_PATH}/models'
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

encoder = Encoder(high_dims,low_dims)
dynamics = LatentDynamics(low_dims)
decoder = Decoder(low_dims,high_dims)

encoder = torch.load(f"{MODEL_PATH}/encoder_{mode}_0.1_20steps_1M{'_warmup' if warmup > 0 else ''}_64.pt",map_location=torch.device('cpu'))
decoder = torch.load(f"{MODEL_PATH}/decoder_{mode}_0.1_20steps_1M{'_warmup' if warmup > 0 else ''}_64.pt",map_location=torch.device('cpu'))
dynamics = torch.load(f"{MODEL_PATH}/dynamics_{mode}_0.1_20steps_1M{'_warmup' if warmup > 0 else ''}_64.pt",map_location=torch.device('cpu'))


xs = np.linspace(-np.pi, np.pi, 20)
ys = np.linspace(-2*np.pi, 2*np.pi, 20)
num_steps = 20

plt.figure(figsize=(8,8))

for i in tqdm(range(xs.shape[0])):
    for j in range(ys.shape[0]):
        x = np.array([xs[i], ys[j]])
        x = env.transform(x)
        x = (x - tf_bounds[:,0]) / (tf_bounds[:,1] - tf_bounds[:,0])
        x = torch.tensor(x, dtype=torch.float32)
        x = encoder(x)
        x = x.detach().numpy()
        traj = [x]
        for k in range(num_steps):
            x = torch.tensor(x, dtype=torch.float32)
            x = dynamics(x)
            x = x.detach().numpy()
            traj.append(x)
        traj = np.array(traj)
        # plt.plot(traj[:,0], traj[:,1], c='black')
        # plt.scatter(traj[0,0], traj[0,1], c='r')
        tf_traj = []
        for k in range(traj.shape[0]):
            x = traj[k]
            x = torch.tensor(x, dtype=torch.float32)
            x = decoder(x)
            x = x.detach().numpy()
            x = x * (tf_bounds[:,1] - tf_bounds[:,0]) + tf_bounds[:,0]
            x = env.inverse_transform(x)
            tf_traj.append(x)
        tf_traj = np.array(tf_traj)
        # Iterate through tf_traj. If the absolute diff between consecutive
        # thetas is greater than theta_thresh, then split the trajectory
        # and plot each segment separately.
        start = 0
        for k in range(1, tf_traj.shape[0]):
            if np.abs(tf_traj[k,0] - tf_traj[k-1,0]) > theta_thresh:
                plt.plot(tf_traj[start:k,0], tf_traj[start:k,1], c='black')
                start = k
        if start == 0:
            plt.plot(tf_traj[:,0], tf_traj[:,1], c='black')
        # plt.plot(tf_traj[:,0], tf_traj[:,1], c='black')
        plt.scatter(tf_traj[0,0], tf_traj[0,1], color='red')

plt.xlim(-np.pi,np.pi)
plt.ylim(-2*np.pi,2*np.pi)
# plt.xlim(-0.4,0.6)
# plt.ylim(-0.4,0.8)
plt.xlabel("theta")
plt.ylabel("theta_dot")
plt.show()
