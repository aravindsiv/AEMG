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

def write_experiments(morse_graph, number_of_attactors, experiment_name, name_folder, name="out_exp"):

    new_out_dir = 'output/' + name_folder
    if not os.path.exists(new_out_dir):
        os.makedirs(new_out_dir)

    name = f"{new_out_dir}/{name}.txt"

    with open(name, "a") as file:
        file.write(experiment_name)

        S = set(morse_graph.vertices())

        counting_attractors = 0
        while len(S) != 0:
            v = S.pop()
            if len(morse_graph.adjacencies(v)) == 0:
                counting_attractors += 1

        if counting_attractors >= number_of_attactors:
            file.write(",1\n")
            return      


        # for u in range(morse_graph.num_vertices()):
            
        #     if len(morse_graph.adjacencies(u)) > 1:
        #         file.write(",1\n")
        #         return    
                

        file.write(",0\n")


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

    args = parser.parse_args()
    config_fname = args.config_dir + args.config

    with open(config_fname) as f:
        config = eval(f.read())

    dyn_utils = DynamicsUtils(config)

    MG_util = CMGDB_util.CMGDB_util()

    sb = 14
    number_of_steps = 5 * config['step']  # 0.5 seconds in total
    if config['system'] == "discrete_map":
        number_of_steps = 1
    uniform_sample = True

    subdiv_init = subdiv_min = subdiv_max = sb  # non adaptive proceedure

    # base name for the output files.
    base_name = f"{config['system']}_{config['control']}_{number_of_steps}_{subdiv_init}"
    print(base_name)


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

    K = [1.5]*2
    def F(rect):
        return MG_util.BoxMapK_valid(g, rect, K, valid_grid_latent_space, grid.point2cell)

    morse_graph, map_graph = MG_util.run_CMGDB(
        subdiv_min, subdiv_max, lower_bounds, upper_bounds, phase_periodic, F, base_name, subdiv_init)


    # compute_roa(map_graph, morse_graph, base_name)

    experiment_name = f"{config['experiment']}&{config['num_layers']}&{config['data_dir'][5::]}"
    

    if config['system'].count('pendulum') >= 1:
        number_of_attactors = 3
    else:
        number_of_attactors = 2


    write_experiments(morse_graph, number_of_attactors, experiment_name, config['data_dir'][5::], args.name_out)


if __name__ == "__main__":
    main()