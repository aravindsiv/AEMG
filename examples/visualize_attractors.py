from AEMG.data_utils import *
from AEMG.dynamics_utils import *
from AEMG.mg_utils import *

from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import pdb

if __name__ == "__main__":
    print(os.getcwd() )
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment',help='Directory of results inside output/',type=str,default='pendulum_lqr1k')
    parser.add_argument('--id', type=str, default="")
    parser.add_argument('--print', action='store_true')
    parser.add_argument('--output_dir',type=str, default="")
    parser.add_argument('--labels_file',help="Success/Failure labels inside output/",type=str, default='')
    args = parser.parse_args()

    if args.id == "":
        config_fnames = os.listdir(os.path.join("output/",args.experiment))
    else:
        config_fnames = [args.id]

    print("Assuming all configs have the same dataset.")
    config_fname = os.path.join("output",args.experiment,config_fnames[0], "config.txt")
    with open(config_fname, 'r') as f:
        config = eval(f.read())
    assert config['low_dims'] == 2, "Only 2D systems supported"

    attractors = None
    if config['system'] == 'pendulum':
        attractors = np.array([[-2.1, 0.0], [0.0, 0.0], [2.1, 0.0]])
    elif config['system'] == 'bistable':
        attractors = np.array([[-1.39]+[0.0]*9, [1.39]+[0.0]*9])
    elif config['system'] == 'cartpole':
        attractors = np.array([[0.0, 0.0, 0.0, 0.0],
                            [1.0, np.pi, 0.0, 0.0],
                            [-1.0, -np.pi, 0.0, 0.0]])
    else:
        if args.labels_file == '':
            print("No labels file provided")
            exit(0)
        else:
            trajectories = TrajectoryDataset(config, os.path.join("output", args.labels_file))
            attractors = trajectories.get_attracting_final_points()
    
    for fname in tqdm(config_fnames):
        config_fname = os.path.join("output",args.experiment,fname, "config.txt")

        with open(config_fname, 'r') as f:
            config = eval(f.read())
        
        dynamics = DynamicsUtils(config)

        try:
            mg_out_utils = MorseGraphOutputProcessor(config)
        except FileNotFoundError or ValueError:
            exit(0)

        # setting the size of the plot
        fig_w=8
        fig_h=8
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        lower_bounds = [-1,-1]
        upper_bounds = [1,1]
        d1=0
        d2=1
        fontsize=16
        tick=5
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        ax.set_xlim([lower_bounds[d1], upper_bounds[d1]])
        ax.set_ylim([lower_bounds[d2], upper_bounds[d2]])
        plt.xticks(np.linspace(lower_bounds[d1], upper_bounds[d1], tick))
        plt.yticks(np.linspace(lower_bounds[d2], upper_bounds[d2], tick))
        ax.set_xlabel(str(d1))
        ax.set_ylabel(str(d2))
        ax.xaxis.label.set_size(fontsize)
        ax.yaxis.label.set_size(fontsize)

        if args.labels_file == '':
            attractors = dynamics.system.transform(attractors)

        for i in range(len(attractors)):
            attractor = attractors[i]
            zt = dynamics.encode(attractor)
            plt.scatter(zt[0], zt[1], color='r', marker='x',s=100, label='GT Attractor' if i==0 else None)
        
        all_attractor_centers = []
        print('num attractors', mg_out_utils.get_num_attractors())
        for i in range(mg_out_utils.get_num_attractors()):
            attractor_tiles = mg_out_utils.get_corner_points_of_attractor(mg_out_utils.attractor_nodes[i])
            attractor_mean_corner_points = np.mean(attractor_tiles, axis=0)
            attractor_center = (attractor_mean_corner_points[:config['low_dims']] + attractor_mean_corner_points[config['low_dims']:]) / 2.0
            if args.print:
                print("Obtained Attractor {}:".format(i))
                print(dynamics.system.inverse_transform(dynamics.decode(attractor_center)))
            print(attractor_tiles.shape[0])
            for j in range(attractor_tiles.shape[0]):
                cp_low = attractor_tiles[j, :config['low_dims']]
                cp_high = attractor_tiles[j, config['low_dims']:]
                tile_center = (cp_low + cp_high) / 2.0
                plt.scatter(tile_center[0], tile_center[1], color='b', s = 100./attractor_tiles.shape[0], marker='.',label='MG Attractor' if i==0 and j==0 else None)
        
        plt.legend(loc='best')
        
        if args.output_dir:
            plt.savefig(os.path.join(args.output_dir,fname+"_attractors.png"))
            print(os.path.join(args.output_dir,fname+"_attractors.png"))
        else:
            plt.savefig(os.path.join(config['output_dir'],  "attractors.png"))
        plt.close()
