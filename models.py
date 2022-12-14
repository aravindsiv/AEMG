import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self,input_shape,lower_shape):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.ReLU(True), 
            nn.Linear(32, 32), 
            nn.ReLU(True), 
            nn.Linear(32, lower_shape),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder(nn.Module):
    def __init__(self,lower_shape,input_shape):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(lower_shape, 32),
            nn.ReLU(True),
            nn.Linear(32, 32),
            nn.ReLU(True), 
            nn.Linear(32, input_shape),
            nn.Sigmoid() 
            )

    def forward(self, x):
        x = self.decoder(x)
        return x

class LatentDynamics(nn.Module):
    # Takes as input an encoding and returns a latent dynamics
    # vector which is just another encoding
    def __init__(self,lower_shape):
        super(LatentDynamics, self).__init__()
        self.dynamics = nn.Sequential(
            nn.Linear(lower_shape, 32),
            nn.ReLU(True),
            nn.Linear(32, 32),
            nn.ReLU(True), 
            nn.Linear(32, lower_shape), 
            nn.Tanh()
            )
    
    def forward(self, x):
        x = self.dynamics(x)
        return x


class FullModel(nn.Module):
    def __init__(self, input_shape, lower_shape):
        super(FullModel, self).__init__()
        self.encoder = Encoder(input_shape, lower_shape)
        self.decoder = Decoder(lower_shape, input_shape)
        self.dynamics = LatentDynamics(lower_shape)

    def forward(self, x_t, x_tau):
        z_t = self.encoder(x_t)
        z_tau_dyn = self.dynamics(z_t)
        x_tau_dyn = self.dynamics(z_tau_dyn)

        x_t_pred = self.decoder(z_t)
        z_tau_pred = self.encoder(x_tau)
        
        return x_tau_dyn, x_t_pred, z_tau_dyn, z_tau_pred



