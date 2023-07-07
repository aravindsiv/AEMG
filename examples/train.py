from AEMG.data_utils import DynamicsDataset
from AEMG.models import *
from AEMG.training import Training, TrainingConfig

import numpy as np
import scipy
from tqdm import tqdm
import pickle
import argparse

from torch.utils.data import DataLoader


def check_collapse(encoder, dataset):
    
    dim_high = len(dataset[0][0])

    dim_low = len(encoder(dataset[0][0]))

    epsilon = 0.05
    epsilon = np.power(epsilon, 1/dim_low)
    distance = 0.5

    test_freq = int(min(10000, (len(dataset)-dim_low)/dim_low))

    # train_dataset = np.random.shuffle(train_dataset)
    for test_index in range(test_freq):

        matrix3 = np.array([encoder(dataset[i + test_index * dim_low][0]).detach().numpy() for i in range(dim_low+1)])        
        a = scipy.spatial.distance_matrix(matrix3, matrix3)
        ind = np.unravel_index(np.argmax(a, axis=None), a.shape)

        if a[ind] < distance and a[ind] != 1:
            matrix = [encoder(dataset[i + test_index * dim_low][0]).detach().numpy() - encoder(dataset[dim_low+1 + test_index * dim_low][0]).detach().numpy() for i in range(dim_low)]
            # matrix = np.array(matrix)
            print(a[ind],"\n", np.linalg.det(matrix)**2,"\n", matrix)
            if np.linalg.det(matrix)**2 > epsilon**2:
                return False
            
    print("\033[91m Collapse")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir',help='Directory of config files',type=str,default='config/')
    parser.add_argument('--config',help='Config file inside config_dir',type=str,default='bistable.txt')
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


    # if not check_collapse(trainer.encoder, dataset):
    check_collapse(trainer.encoder, train_dataset)
    trainer.save_models()

    

if __name__ == "__main__":
    main()