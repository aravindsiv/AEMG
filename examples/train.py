from AEMG.data_utils import DynamicsDataset
from AEMG.models import *
from AEMG.training import Training, TrainingConfig

import numpy as np 
from tqdm import tqdm
import pickle
import argparse

from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir',help='Directory of config files',type=str,default='config/')
    parser.add_argument('--config',help='Config file inside config_dir',type=str,default='discrete_map.txt')
    parser.add_argument('--verbose',help='Print training output',type=int,default=1)

    args = parser.parse_args()
    config_fname = args.config_dir + args.config

    with open(config_fname) as f:
        config = eval(f.read())
    
    torch.manual_seed(config["seed"])
    
    dataset = DynamicsDataset(config)
    
    np.random.seed(config["seed"])

    train_size = int(0.8*len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)

    if args.verbose:
        print("Train size: ", len(train_dataset))
        print("Test size: ", len(test_dataset))

    loaders = {'train': train_loader, 'test': test_loader}

    trainer = Training(config, loaders, bool(int(args.verbose)))
    experiment = TrainingConfig(config['experiment'])

    for i,exp in enumerate(experiment):
        if args.verbose:
            print("Training loss weights: ", exp)
        trainer.train(config["epochs"], config["patience"], exp)
        trainer.save_logs(suffix =str(i))
        trainer.reset_losses()
    
    trainer.save_models()

if __name__ == "__main__":
    main()