import AEMG
from AEMG.systems.utils import get_system

import numpy as np
from tqdm import tqdm 
from torch.utils.data import Dataset
import torch

import os
import sys

class DynamicsDataset(Dataset):
    def __init__(self, config):
        Xt = []
        Xnext = []

        step = config['step']

        for f in tqdm(os.listdir(config['data_dir'])):
            data = np.loadtxt(os.path.join(config['data_dir'], f), delimiter=',')
            for i in range(data.shape[0] - step):
                Xt.append(data[i])
                Xnext.append(data[i + step])
            
        self.Xt = np.array(Xt)
        self.Xnext = np.array(Xnext)

        # Normalize the data
        if config['use_limits']:
            raise NotImplementedError
        else:
            # Get bounds from the max of both Xt and Xnext
            self.X_min = np.min(np.concatenate((self.Xt, self.Xnext), axis=0), axis=0)
            self.X_max = np.max(np.concatenate((self.Xt, self.Xnext), axis=0), axis=0)

        self.Xt = (self.Xt - self.X_min) / (self.X_max - self.X_min)
        self.Xnext = (self.Xnext - self.X_min) / (self.X_max - self.X_min)

        # If model_dir does nto exist, create it
        if not os.path.exists(config['model_dir']):
            os.makedirs(config['model_dir'])
        
        # Write the normalization parameters to a file
        np.savetxt(os.path.join(config['model_dir'], 'X_min.txt'), self.X_min, delimiter=',')
        np.savetxt(os.path.join(config['model_dir'], 'X_max.txt'), self.X_max, delimiter=',')

        # Convert to torch tensors
        self.Xt = torch.from_numpy(self.Xt).float()
        self.Xnext = torch.from_numpy(self.Xnext).float()
    
    def __len__(self):
        return len(self.Xt)
    
    def __getitem__(self, idx):
        return self.Xt[idx], self.Xnext[idx]

class TrajectoryDataset:
    # Useful for plotting
    def __init__(self, config):
        self.trajs = []

        for f in tqdm(os.listdir(config['data_dir'])):
            data = np.loadtxt(os.path.join(config['data_dir'], f), delimiter=',')
            self.trajs.append(data)
