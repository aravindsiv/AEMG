import os

from AEMG.data_utils import DynamicsDataset
from AEMG.models import *
from AEMG.training import Training

import numpy as np 
from tqdm import tqdm
import pickle
import argparse

from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',help='Config file inside config/',type=str,default='discrete_map.txt')

    args = parser.parse_args()
    config_fname = "config/" + args.config

    with open(config_fname) as f:
        config = eval(f.read())
    
    dataset = DynamicsDataset(config)

    train_size = int(0.8*len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)
    print("Train size: ", len(train_dataset))
    print("Test size: ", len(test_dataset))

    loaders = {'train': train_loader, 'test': test_loader}

    experiment = Training(config, loaders)

    experiment.train_encoder_decoder(config["epochs"], config["patience"], loss='ae1')
    experiment.train_dynamics(config["epochs"], config["patience"])
    experiment.train_all(config["epochs"], config["patience"])

    experiment.save_models()