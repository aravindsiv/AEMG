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


def f(g, X):
    # Now x is in [-1,1]^2
    x = torch.tensor([0.1,0.1],dtype=torch.float32)
    # This brings x to [0,1]^4
    x = model.decoder(x)
    x = x.detach().numpy()
    # This brings x to R^4
    x = x * (tf_bounds[:,1] - tf_bounds[:,0]) + tf_bounds[:,0]
    # This brings x to [-pi,pi] x [-2pi,2pi]
    x = env.inverse_transform(x)
    # This computes the image (using TimeMap)
    x = g(X)
    # Bring x to R^4
    x = env.transform(x)
    # Bring x to [0,1]^4
    x = (x - tf_bounds[:,0]) / (tf_bounds[:,1] - tf_bounds[:,0])
    x = torch.from_numpy(x).float()
    # Bring x to [-1,1]^2
    x = model.encoder(x).detach().numpy()
    return x
