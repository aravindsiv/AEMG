import AEMG
from AEMG.systems.utils import get_system

import numpy as np
from tqdm import tqdm 
from torch.utils.data import Dataset
from AEMG.systems.utils import get_system
import torch

import os
import sys

class DynamicsDataset(Dataset):
    def __init__(self, config):
        Xt = []
        Xnext = []

        step = config['step']
        subsample = config['subsample']
        system = get_system(config['system'], config['high_dims'])
        print("Getting data for: ",system.name)

        for f in tqdm(os.listdir(config['data_dir'])):
            data = np.loadtxt(os.path.join(config['data_dir'], f), delimiter=',')
            indices = np.arange(data.shape[0])
            subsampled_indices = indices % subsample == 0
            subsampled_data_untransformed = data[subsampled_indices]
            subsampled_data = system.transform(subsampled_data_untransformed)
            Xt.append(subsampled_data[:-step])
            Xnext.append(subsampled_data[step:])
            # for i in range(subsampled_data.shape[0] - step): 
            #     Xt.append(system.transform(subsampled_data[i]))
            #     Xnext.append(system.transform(subsampled_data[i + step]))
            
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
    def __init__(self, config, labels_fname=None):
        self.trajs = []
        subsample = config['subsample']

        system = get_system(config['system'], config['high_dims'])
        print("Getting data for: ",system.name)

        self.labels_dict = {}
        self.labels = []
        if labels_fname is not None:
            labels = np.loadtxt(labels_fname, delimiter=',', dtype=str)
            # Create dict with key as fname and value as label
            for i in range(len(labels)):
                self.labels_dict[labels[i,0]] = int(labels[i,1])

        for f in tqdm(os.listdir(config['data_dir'])):
            raw_data = np.loadtxt(os.path.join(config['data_dir'], f), delimiter=',')
            if len(raw_data) == 0: continue
            indices = np.arange(raw_data.shape[0])
            subsampled_indices = indices % subsample == 0
            subsampled_data = raw_data[subsampled_indices]
            # Transform each state in the trajectory
            self.trajs.append(system.transform(subsampled_data))
            # data = []
            # for i in range(subsampled_data.shape[0]):
            #     data.append(system.transform(subsampled_data[i]))
            # data = np.array(data)
            # self.trajs.append(data)
            if labels_fname is not None:
                try:
                    self.labels.append(self.labels_dict[f])
                except KeyError:
                    print("No label found for ", f)
                    self.labels.append(-1)
    
    def __len__(self):
        return len(self.trajs)
    
    def __getitem__(self, idx):
        return self.trajs[idx]
    
    def get_label(self,index):
        return self.labels[index]
    
    def get_attracting_final_points(self):
        assert len(self.trajs) == len(self.labels)
        final_points = []
        for i in range(len(self.trajs)):
            if self.labels[i] == 1:
                final_points.append(self.trajs[i][-1])
        return np.array(final_points)
