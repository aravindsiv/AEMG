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

x = torch.tensor([0.1,0.1],dtype=torch.float32)
x = model.decoder(x)
x = x.detach().numpy()
x = x * (tf_bounds[:,1] - tf_bounds[:,0]) + tf_bounds[:,0]
print("x,y,xdot,ydot: ",x)
x = env.inverse_transform(x)
print("theta, thetadot: ",x)
y = env.transform(x)
print("x,y,xdot,ydot: ",y)

traj = [x]
for i in range(100):
    x = env.step(x)
    traj.append(x)

traj = np.array(traj)

plt.figure(figsize=(8,8))
plt.xlim(-np.pi,np.pi)
plt.ylim(-2*np.pi,2*np.pi)
plt.xlabel("theta")
plt.ylabel("theta_dot")
plt.plot(traj[:,0],traj[:,1])
plt.scatter(traj[0,0],traj[0,1],c="r")
plt.show()

plt.figure(figsize=(8,8))

tf_traj = np.vstack([env.transform(s) for s in traj])
tf_traj = (tf_traj - tf_bounds[:,0]) / (tf_bounds[:,1] - tf_bounds[:,0])
tf_traj = torch.from_numpy(tf_traj).float()
encoding = model.encoder(tf_traj).detach().numpy()
plt.plot(encoding[:,0],encoding[:,1],color='black')
plt.scatter(encoding[0,0],encoding[0,1],c="r")

plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel("encoding 1")
plt.ylabel("encoding 2")
plt.show()