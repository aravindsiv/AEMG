import numpy as np 
import torch
import torch.nn as nn

from AEMG.systems.utils import get_system
from AEMG.models import *

import os

class MorseGraphUtils:
    def __init__(self, config):
        self.system = get_system(config['system'])

        if config["use_limits"]:
            raise NotImplementedError
        else:
            self.X_min = np.loadtxt(os.path.join(config['model_dir'], 'X_min.txt'), delimiter=',')
            self.X_max = np.loadtxt(os.path.join(config['model_dir'], 'X_max.txt'), delimiter=',')

        self.encoder  = Encoder(config['high_dims'], config['low_dims'])
        self.decoder  = Decoder(config['low_dims'], config['high_dims'])
        self.dynamics = LatentDynamics(config['low_dims'])

        self.encoder = torch.load(os.path.join(config['model_dir'], 'encoder.pt'))
        self.decoder = torch.load(os.path.join(config['model_dir'], 'decoder.pt'))
        self.dynamics = torch.load(os.path.join(config['model_dir'], 'dynamics.pt'))
    
    def f(self, z):
        # This function takes as input a latent state and returns the next latent state
        z = torch.tensor(z, dtype=torch.float32)
        return self.dynamics(z).detach().numpy()
    
    def encode(self, x):
        # This function takes as input a raw state (un-normalized)
        # and returns the latent state
        x = (x - self.X_min) / (self.X_max - self.X_min)
        x = torch.tensor(x, dtype=torch.float32)
        return self.encoder(x).detach().numpy()
    
    def decode(self, x):
        # This function takes as input a latent state
        # and returns the raw state (un-normalized)
        x = torch.tensor(x, dtype=torch.float32)
        x = self.decoder(x).detach().numpy()
        return x * (self.X_max - self.X_min) + self.X_min
        