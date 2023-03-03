import sys 
import os
from tqdm import tqdm
from AEMG.systems.utils import get_system
import Grid

import numpy as np
np.set_printoptions(suppress=True)
np.random.seed(101193)

import argparse

def sample_points(lower_bounds, upper_bounds, num_pts):
    # Sample num_pts in dimension dim, where each
    # component of the sampled points are in the
    # ranges given by lower_bounds and upper_bounds
    dim = len(lower_bounds)
    X = np.random.uniform(lower_bounds, upper_bounds, size=(num_pts, dim))
    return X

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', help='Trajectory length', type=float, default=2.0)
    parser.add_argument('--time_step', help='Time step', type=float, default=0.1)
    parser.add_argument('--num_trajs', help='Number of trajectories', type=int, default=10000)
    parser.add_argument('--mode', help='Mode', type=str, default='none')
    parser.add_argument('--save_dir', help='Save directory', type=str, default='data')

    args = parser.parse_args()
    
    system = get_system("discrete_map")

    num_trajs = args.num_trajs
    num_steps = int(args.time/args.time_step)

    lower_bounds = [-2]*10
    upper_bounds = [2]*10
    random_sample = True

    dim = len(upper_bounds)

    num_pts = num_trajs

    if random_sample:
        X = sample_points(lower_bounds, upper_bounds, num_pts)
    else:
        grid = Grid.Grid(lower_bounds, upper_bounds, dim + 1)
        X = grid.uniform_sample()

    num_steps = len(X)

    def f(X):
        Y0 = np.arctan(4*X[:,0]).reshape(-1,1)
        Y1 = X[:,1::]/2
        return np.concatenate((Y0, Y1), axis = 1)

    Y = f(f(f(f(f(f(X))))))

    data = np.concatenate((X, Y), axis = 0)
    
    # Create the data directory if it doesn't exist
    # save_dir = args.save_dir + "/discrete_map_" + str(args.num_trajs)
    save_dir = args.save_dir + "/discrete_map"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    counter = 0

    np.savetxt(f"{save_dir}/{counter}.txt",data,delimiter=",")    