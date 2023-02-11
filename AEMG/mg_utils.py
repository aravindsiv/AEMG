import numpy as np 
import torch
import torch.nn as nn

from AEMG.systems.utils import get_system
from AEMG.models import *

import os

class MorseGraphOutputProcessor:
    def __init__(self, config):
        mg_fname = os.path.join(config['out_dir'], 'mg_output.csv')
        
        self.dims = config['low_dims']

        # Check if the file exists
        if not os.path.exists(mg_fname):
            raise FileNotFoundError("Morse Graph output file does not exist")
        with open(mg_fname, 'r') as f:
            lines = f.readlines()
            # Find indices where the first character is an alphabet
            self.indices = []
            for i, line in enumerate(lines):
                if line[0].isalpha():
                    self.indices.append(i)
            self.box_size = np.array(lines[self.indices[0]+1].split(',')).astype(np.float32)
            self.morse_nodes_data = np.vstack([np.array(line.split(',')).astype(np.float32) for line in lines[self.indices[1]+1:self.indices[2]]])
            self.attractor_nodes_data = np.vstack([np.array(line.split(',')).astype(np.float32) for line in lines[self.indices[2]+1:]])

        self.morse_nodes = np.unique(self.morse_nodes_data[:, 1])
        self.attractor_nodes = np.unique(self.attractor_nodes_data[:, 1])
    
    def get_corner_points_of_attractor(self, id):
        # Get the attractor nodes
        attractor_nodes = self.attractor_nodes_data[self.attractor_nodes_data[:, 1] == id]
        return attractor_nodes[:, 2:]        
    
    def which_morse_set(self, point):
        assert point.shape[0] == self.dims
        for i in range(self.morse_nodes_data.shape[0]):
            corner_point_low  = self.morse_nodes_data[i, 2:2+self.dims]
            corner_point_high = self.morse_nodes_data[i, 2+self.dims:]
            if np.all(point >= corner_point_low) and np.all(point <= corner_point_high):
                return self.morse_nodes_data[i, 1]
        return -1

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
        