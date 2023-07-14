from AEMG.data_utils import *
from AEMG.dynamics_utils import *
from AEMG.mg_utils import *

from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, default="output/pendulum_lqr1k")
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--print', action='store_true')
    parser.add_argument('--output_dir',type=str, default="")
    args = parser.parse_args()

    config_fname = os.path.join(args.config_dir, args.id, "config.txt")

    with open(config_fname, 'r') as f:
        config = eval(f.read())
    assert config['low_dims'] == 2, "Only 2D systems supported"

    dynamics = DynamicsUtils(config)

    attractors = None
    if config['system'] == 'pendulum':
        attractors = np.array([[-2.1, 0.0], [0.0, 0.0], [2.1, 0.0]])
    elif config['system'] == 'bistable':
        attractors = np.array([[-1.39]+[0.0]*9, [1.39]+[0.0]*9])
    else:
        raise NotImplementedError

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

    for i in range(len(attractors)):
        attractor = attractors[i]
        xt = dynamics.system.transform(attractor)
        zt = dynamics.encode(xt)
        plt.scatter(zt[0], zt[1], color='r', marker='x',s=100, label='GT Attractor' if i==0 else None)
    
    all_attractor_centers = []
    for i in range(mg_out_utils.get_num_attractors()):
        attractor_tiles = mg_out_utils.get_corner_points_of_attractor(mg_out_utils.attractor_nodes[i])
        attractor_mean_corner_points = np.mean(attractor_tiles, axis=0)
        attractor_center = (attractor_mean_corner_points[:config['low_dims']] + attractor_mean_corner_points[config['low_dims']:]) / 2.0
        if args.print:
            print("Obtained Attractor {}:".format(i))
            print(dynamics.system.inverse_transform(dynamics.decode(attractor_center)))
        for j in range(attractor_tiles.shape[0]):
            cp_low = attractor_tiles[j, :config['low_dims']]
            cp_high = attractor_tiles[j, config['low_dims']:]
            tile_center = (cp_low + cp_high) / 2.0
            plt.scatter(tile_center[0], tile_center[1], color='b', s = 100./attractor_tiles.shape[0], marker='.',label='MG Attractor' if i==0 and j==0 else None)
    
    # plt.legend(loc='best')
    
    if args.output_dir:
        plt.savefig(os.path.join(args.output_dir, "attractors.png"))
    else:
        plt.savefig(os.path.join(config['output_dir'], "attractors.png"))
