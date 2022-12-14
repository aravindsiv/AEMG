import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from pendulum import PendulumNoCtrl
from tqdm import tqdm
import pickle as pkl

env = PendulumNoCtrl()
tf_bounds = env.get_transformed_state_bounds()
gt_bounds = env.state_bounds

class AutoEncoder(nn.Module):
    def __init__(self,input_shape,lower_shape):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, lower_shape),
            nn.Tanh())
        self.decoder = nn.Sequential(
            nn.Linear(lower_shape, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, input_shape),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = torch.load("pendulum_ae.pt")

mode = "lqr"

if mode == "noctrl":
    step = 15
    xs = np.linspace(-np.pi,np.pi,step)
    ys = np.linspace(-2*np.pi,2*np.pi,step)

    trajs = []
    for i in tqdm(range(xs.shape[0])):
        for j in range(ys.shape[0]):
            traj = [np.array([xs[i],ys[j]])]
            for k in range(100):
                traj.append(env.step(traj[-1]))
            traj = np.vstack(traj)
            trajs.append(traj)

if mode == "lqr":
    fname = "LQRTraj_GP.pkl"

    with open(fname, "rb") as f:
        dat = pkl.load(f)

    x = np.vstack(dat["obs"])
    next = np.vstack(dat["next_obs"])

    trajs = []
    prev_idx = 0
    for i in range(x.shape[0]-1):
        if not np.allclose(x[i+1],next[i]):
            # Trajectory ended at i
            trajs.append(x[prev_idx:i+1])
            prev_idx = i+1

plt.figure(figsize=(8,8))
for traj in tqdm(trajs):
    plt.plot(traj[:,0],traj[:,1],color='black')
    plt.scatter(traj[0,0],traj[0,1],c="r")

plt.xlim(-np.pi,np.pi)
plt.ylim(-2*np.pi,2*np.pi)
plt.xlabel("theta")
plt.ylabel("theta_dot")
plt.title("Trajectories in phase space")
plt.show()


plt.figure(figsize=(8,8))
for traj in tqdm(trajs):
    tf_traj = np.vstack([env.transform(s) for s in traj])
    tf_traj = (tf_traj - tf_bounds[:,0]) / (tf_bounds[:,1] - tf_bounds[:,0])
    tf_traj = torch.from_numpy(tf_traj).float()
    encoding = model.encoder(tf_traj).detach().numpy()
    plt.plot(encoding[:,0],encoding[:,1],color='black')
    plt.scatter(encoding[0,0],encoding[0,1],c="r")

plt.xlim(-1.1,1.1)
plt.ylim(-1.1,1.1)
plt.xlabel("encoding 1")
plt.ylabel("encoding 2")
plt.title("Trajectories in latent space")
plt.show()

'''

# Transform the trajectory
tf_traj = np.vstack([env.transform(s) for s in traj])
tf_traj = (tf_traj - tf_bounds[:,0]) / (tf_bounds[:,1] - tf_bounds[:,0])

# Encode the trajectory
tf_traj = torch.from_numpy(tf_traj).float()
encoding = model.encoder(tf_traj).detach().numpy()

plt.figure(figsize=(8,8))
plt.xlim(-1.1,1.1)
plt.ylim(-1.1,1.1)
plt.plot(encoding[:,0],encoding[:,1])
plt.scatter(encoding[0,0],encoding[0,1],c="r")
plt.show()
'''
