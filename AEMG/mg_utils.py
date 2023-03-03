import numpy as np 
import torch
import torch.nn as nn

from AEMG.systems.utils import get_system

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
        