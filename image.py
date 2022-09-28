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
x = env.inverse_transform(x)

for i in range(100):
    # This is the g function
    x = env.step(x)

x = (x - tf_bounds[:,0]) / (tf_bounds[:,1] - tf_bounds[:,0])
x = torch.from_numpy(x).float()
encoding = model.encoder(x).detach().numpy()
print(x)
