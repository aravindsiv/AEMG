import AEMG
from AEMG.systems.utils import get_system, multi_dim_tensor_cartesian

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
            
        self.Xt = np.vstack(Xt)
        self.Xnext = np.vstack(Xnext)
        assert len(self.Xt) == len(self.Xnext), "Xt and Xnext must have the same length"

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

class LabelsDataset(Dataset):
    def __init__(self,config):
        labels = np.loadtxt(config['labels_fname'], delimiter=',', dtype=str)
        labels_dict = {}
        for i in range(len(labels)):
            labels_dict[labels[i,0]] = int(labels[i,1])

        self.final_points = []
        self.labels = []

        for f in tqdm(os.listdir(config['data_dir'])):
            data = np.loadtxt(os.path.join(config['data_dir'], f), delimiter=',')
            self.final_points.append(data[-1])
            try:
                self.labels.append(labels_dict[f])
            except KeyError:
                print("No label found for ", f)
                self.labels.append(0)

        self.final_points = np.array(self.final_points)
        system = get_system(config['system'], config['high_dims'])
        self.final_points = system.transform(self.final_points)
        X_max = np.loadtxt(os.path.join(config['model_dir'], 'X_max.txt'), delimiter=',')
        X_min = np.loadtxt(os.path.join(config['model_dir'], 'X_min.txt'), delimiter=',')
        self.final_points = (self.final_points - X_min) / (X_max - X_min)
        self.final_points = torch.from_numpy(self.final_points).float()

        self.labels = np.array(self.labels)
        self.labels = torch.from_numpy(self.labels).long()

        self.generate_contrastive_triplets()

    def __len__(self):
        return len(self.contrastive_triplets)

    def __getitem__(self, idx):
        return self.contrastive_triplets[idx]

    def generate_contrastive_triplets(self):
        """
        Generates the contrastive triplets for the dataset where each triplet has the form (anchor, positive, negative)
        The anchor can be either a success or failure point, the positive is a point of the same class as the anchor, and the negative is a point of the opposite class
        """

        # Gathering the indices of the success and failure labels
        success_indices = torch.where(self.labels == 1)[0]
        failure_indices = torch.where(self.labels == 0)[0]

        # Generating all possible combinations of success and failure indices for positive anchor pairs
        success_anchor_pairs = torch.stack(torch.meshgrid(success_indices, success_indices)).T.reshape(-1,2)
        failure_anchor_pairs = torch.stack(torch.meshgrid(failure_indices, failure_indices)).T.reshape(-1,2)

        # Removing instances with the same index
        success_anchor_pairs = success_anchor_pairs[success_anchor_pairs[:, 0] != success_anchor_pairs[:, 1]]
        failure_anchor_pairs = failure_anchor_pairs[failure_anchor_pairs[:, 0] != failure_anchor_pairs[:, 1]]

        success_indices = success_indices[:, None]
        failure_indices = failure_indices[:, None]

        # Adding negative values to the positive anchor pairs
        success_failure_triplets = multi_dim_tensor_cartesian(success_anchor_pairs, failure_indices)
        failure_success_triplets = multi_dim_tensor_cartesian(failure_anchor_pairs, success_indices)

        # Concatenating all triplets
        self.contrastive_triplets = torch.cat([success_failure_triplets, failure_success_triplets], dim=0)

    def collate_fn(self, batch):
        """
        Processes the batch of triplets to create a batch of points present in the triplets and a new batch of triplets is created with the updated indices of the points
        :param batch: batch of triplets
        :return: updated batch of triplets and the batch of points
        """
        triplets = np.array([triplet.to('cpu').numpy() for triplet in batch])

        # Getting the points present in the batch of triplets
        unique_indices = np.unique(triplets)
        x_batch = self.final_points[unique_indices]

        # Updating the indices of the triplets to match the indices of the points in the batch
        updated_triplets = []
        for anchor_idx, pos_idx, neg_idx in triplets:
            updated_anchor_idx = np.where(unique_indices == anchor_idx)[0][0]
            updated_pos_idx = np.where(unique_indices == pos_idx)[0][0]
            updated_neg_idx = np.where(unique_indices == neg_idx)[0][0]
            updated_triplets.append([updated_anchor_idx, updated_pos_idx, updated_neg_idx])

        return torch.tensor(updated_triplets), x_batch

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
    
    def get_successful_initial_conditions(self):
        assert len(self.trajs) == len(self.labels)
        initial_points = []
        for i in range(len(self.trajs)):
            if self.labels[i] == 1:
                initial_points.append(self.trajs[i][0])
        return np.array(initial_points)

    def get_unsuccessful_initial_conditions(self):
        assert len(self.trajs) == len(self.labels)
        initial_points = []
        for i in range(len(self.trajs)):
            if self.labels[i] == 0:
                initial_points.append(self.trajs[i][0])
        return np.array(initial_points)
    
    def get_successful_final_conditions(self):
        assert len(self.trajs) == len(self.labels)
        final_points = []
        for i in range(len(self.trajs)):
            if self.labels[i] == 1:
                final_points.append(self.trajs[i][-1])
        return np.array(final_points)

    def get_unsuccessful_final_conditions(self):
        assert len(self.trajs) == len(self.labels)
        final_points = []
        for i in range(len(self.trajs)):
            if self.labels[i] == 0:
                final_points.append(self.trajs[i][-1])
        return np.array(final_points)