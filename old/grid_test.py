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


xs = np.linspace(-np.pi, np.pi, 100)
ys = np.linspace(-2*np.pi, 2*np.pi, 20)

plt.figure(figsize=(8,8))

for i in tqdm(range(xs.shape[0])):
    for j in range(ys.shape[0]):
        plt.scatter(xs[i], ys[j])

plt.xlim(-np.pi,np.pi)
plt.ylim(-2*np.pi,2*np.pi)
plt.xlabel("theta")
plt.ylabel("theta_dot")
plt.title("Original State Space")
plt.show()

plt.figure(figsize=(8,8))
pts = []
for i in tqdm(range(xs.shape[0])):
    for j in range(ys.shape[0]):
        x = np.array([xs[i], ys[j]])
        x = env.transform(x)
        x = (x - tf_bounds[:,0]) / (tf_bounds[:,1] - tf_bounds[:,0])
        x = torch.tensor(x, dtype=torch.float32)
        x = encoder(x)
        x = x.detach().numpy()
        pts.append(x)
        plt.scatter(x[0], x[1])

plt.xlim(-1.1,1.1)
plt.ylim(-1.1,1.1)
plt.xlabel("encoding 1")
plt.ylabel("encoding 2")
plt.title("Latent Space")
plt.show()

'''

# Project back to original space
pts = np.vstack(pts)
pts = torch.tensor(pts, dtype=torch.float32)
pts = model.decoder(pts)
pts = pts.detach().numpy()
plt.figure(figsize=(8,8))
for pt in pts:
    pt = pt * (tf_bounds[:,1] - tf_bounds[:,0]) + tf_bounds[:,0]
    pt = env.inverse_transform(pt)
    plt.scatter(pt[0], pt[1], c='r')
plt.xlim(-np.pi,np.pi)
plt.ylim(-2*np.pi,2*np.pi)
plt.xlabel("theta")
plt.ylabel("theta_dot")
plt.title("Original State Space")
plt.show()


'''



# class CircleEquation(nn.Module):
#     def __init__(self,center=[0,0],radius=1):
#         super(CircleEquation,self).__init__()
#         # Create the center and radius parameters as differentiable variables
#         self.center = nn.Parameter(torch.tensor(center, dtype=torch.float32), requires_grad=True)
#         self.radius = nn.Parameter(torch.tensor(radius, dtype=torch.float32), requires_grad=True)

#     def forward(self,x):
#         return torch.norm(x - self.center, dim=1) - self.radius

# # Create a circle equation
# circle = CircleEquation()

# # Create an optimizer
# optimizer = torch.optim.Adam(circle.parameters(), lr=0.01)

# # Create a loss function
# loss_fn = nn.MSELoss()

# pts = torch.tensor(pts, dtype=torch.float32)

# # Train the circle equation
# for i in tqdm(range(1000)):
#     optimizer.zero_grad()
#     loss = loss_fn(circle(pts), torch.zeros(pts.shape[0]))
#     loss.backward()
#     optimizer.step()

# # Plot the circle
# plt.figure(figsize=(8,8))

# from matplotlib.patches import Circle
# circle_center = circle.center.detach().numpy()
# circle_radius = circle.radius.detach().numpy()
# circle = Circle(circle_center, circle_radius, fill=False)
# ax = plt.gca()
# ax.add_patch(circle)

# plt.scatter(pts[:,0], pts[:,1], c='r')
# plt.xlim(-1.1,1.1)
# plt.ylim(-1.1,1.1)
# plt.xlabel("encoding 1")
# plt.ylabel("encoding 2")
# plt.title("Latent Space")
# plt.show()

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
