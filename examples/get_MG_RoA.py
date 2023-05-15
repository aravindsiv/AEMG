from AEMG.systems.utils import get_system
from AEMG.models import *
from AEMG.dynamics_utils import DynamicsUtils
from AEMG.systems import pendulum
from AEMG.data_utils import DynamicsDataset
from torch.utils.data import DataLoader
import argparse

import CMGDB_util
import RoA
import Grid
import dyn_tools

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

import numpy as np

def write_experiments(morse_graph, experiment_name, output_dir, name="out_exp"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    name = f"{output_dir}/{name}"

    with open(name, "w") as file:
        file.write(experiment_name)

        S = set(morse_graph.vertices())

        counting_attractors = 0
        list_attractors = []
        while len(S) != 0:
            v = S.pop()
            if len(morse_graph.adjacencies(v)) == 0:
                counting_attractors += 1
                list_attractors.append(v)

        file.write(f":{list_attractors},{counting_attractors}\n")


def compute_roa(map_graph, morse_graph, base_name):

    startTime = datetime.now()

    roa = RoA.RoA(map_graph, morse_graph)

    print(f"Time to build the regions of attraction = {datetime.now() - startTime}")

    roa.save_file(base_name)

    fig, ax = roa.PlotTiles(name_plot=base_name)

    # RoA.PlotTiles(lower_bounds, upper_bounds,
    #               from_file=base_name, from_file_basic=True)

    # plt.savefig(base_name, bbox_inches='tight')
    plt.show()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir',help='Directory of config files',type=str,default='config/')
    parser.add_argument('--config',help='Config file inside config_dir',type=str,default='discrete_map.txt')
    parser.add_argument('--name_out',help='Name of the out file',type=str,default='out_exp')
    parser.add_argument('--RoA',help='Compute RoA',action='store_true')
    parser.add_argument('--sub',help='Select subdivision',type=int,default=14)


    args = parser.parse_args()
    config_fname = args.config_dir + args.config

    with open(config_fname) as f:
        config = eval(f.read())

    dyn_utils = DynamicsUtils(config)

    MG_util = CMGDB_util.CMGDB_util()

    sb = args.sub
    number_of_steps = np.ceil(12 / config['step'])  # at least 0.6 seconds in total
    if config['system'] == "discrete_map":
        number_of_steps = 1
    uniform_sample = True

    subdiv_init = subdiv_min = subdiv_max = sb  # non adaptive proceedure




    # Get the limits
    lower_bounds_original_space = np.loadtxt(os.path.join(config['model_dir'], 'X_min.txt'), delimiter=',').tolist()
    upper_bounds_original_space = np.loadtxt(os.path.join(config['model_dir'], 'X_max.txt'), delimiter=',').tolist()
    print("Bounds for decoded space", lower_bounds_original_space, upper_bounds_original_space)

    dim_original_space = config['high_dims']
    grid_original_space = Grid.Grid(lower_bounds_original_space, upper_bounds_original_space, sb + 2)

    dim_latent_space = config['low_dims']
    lower_bounds = [-1]*dim_latent_space
    upper_bounds = [1]*dim_latent_space
    print("Bounds for encoded space", lower_bounds, upper_bounds)

    grid = Grid.Grid(lower_bounds, upper_bounds, sb)

    if uniform_sample:
        grid_original_space_sample = grid_original_space.uniform_sample()

        print("uniform data on the state space grid", grid_original_space_sample.shape)
        print(grid_original_space_sample[1])

        grid_latent_space_sample = dyn_utils.encode(grid_original_space_sample)
        print("uniform data on the latent space grid", grid_latent_space_sample.shape)
        print(grid_latent_space_sample[1])

    else:  # sample from trajectories

        dataset = DynamicsDataset(config)
        grid_latent_space_sample = dyn_utils.encode(dataset.Xt.numpy())
        print("data on the latent space grid", grid_latent_space_sample.shape)

    valid_grid_latent_space = grid.valid_grid(grid_latent_space_sample)

    def g(X):
        return dyn_tools.iterate(dyn_utils.f, X, n=number_of_steps).tolist()

    phase_periodic = [False, False]

    K = [1.1]*2
    def F(rect):
        return MG_util.BoxMapK_valid(g, rect, K, valid_grid_latent_space, grid.point2cell)


    # base name for the output files.
    base_name = f"{config['output__dir']}/{args.name_out}"
    
    
    print(base_name)
    morse_graph, map_graph = MG_util.run_CMGDB(
        subdiv_min, subdiv_max, lower_bounds, upper_bounds, phase_periodic, F, base_name, subdiv_init)


   

    experiment_name = f"{config['experiment']}&{config['num_layers']}&{config['data_dir'][5::]}&{config['step']}&{args.sub}"
    
    write_experiments(morse_graph, experiment_name, config['output_dir'], args.name_out)

    if args.RoA:
     
     compute_roa(map_graph, morse_graph, base_name)

if __name__ == "__main__":
    main()