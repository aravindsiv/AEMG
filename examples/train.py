from AEMG.data_utils import DynamicsDataset
from AEMG.models import *
from AEMG.training import Training

import numpy as np 
from tqdm import tqdm
import pickle
import argparse

from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir',help='Directory of config files',type=str,default='config/')
    parser.add_argument('--config',help='Config file inside config_dir',type=str,default='discrete_map.txt')

    args = parser.parse_args()
    config_fname = args.config_dir + args.config

    with open(config_fname) as f:
        config = eval(f.read())
    
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    
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
    exp_ids = config['experiment'].split('*')

    for i, e in enumerate(exp_ids):
        exp = e.split('_')

        if len(exp)==3:
            weight = [int(j) for j in exp]
            print(weight)
            experiment.train(config["epochs"], config["patience"], weight)
        else:
            print("Invalid training setting")
            exit()
        experiment.save_logs(suffix = "Step" + e)
        experiment.reset_losses()

    experiment.save_models()

if __name__ == "__main__":
    main()