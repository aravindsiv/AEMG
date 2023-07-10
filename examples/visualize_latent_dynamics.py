from AEMG.data_utils import *
from AEMG.dynamics_utils import *

from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, default="output/pendulum_lqr1k")
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--mode', type=str, default="trajectories", choices=["field", "trajectories", "all"])
    parser.add_argument('--output_dir',type=str, default="")
    parser.add_argument('--num_trajs', type=int, default=100)
    args = parser.parse_args()

    config_fname = os.path.join(args.config_dir, args.id, "config.txt")

    with open(config_fname, 'r') as f:
        config = eval(f.read())
    assert config['low_dims'] == 2, "Only 2D systems supported"

    dynamics = DynamicsUtils(config)
    dataset = TrajectoryDataset(config)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    assert len(dataset) >= args.num_trajs, "Not enough trajectories in dataset"
    idxes = np.random.choice(len(dataset), args.num_trajs, replace=False)

    if args.mode == "trajectories" or args.mode == "all":
        fig, ax = plt.subplots(figsize=(10,5))
        ax1 = plt.subplot(1,2,1)
        ax2 = plt.subplot(1,2,2)

        ax1.set_title("Encoded GT Trajectory")
        ax2.set_title("Latent Dynamics Trajectory")

        for idx in tqdm(idxes):
            traj = dataset[idx]
            z = dynamics.encode(traj)
            
            ax1.plot(z[:,0], z[:,1],color='black')
            ax1.scatter(z[0,0], z[0,1], color='r', marker='.')
            ax1.scatter(z[-1,0], z[-1,1], color='g', marker='x')

            z_curr = z[0,:]
            ax2.scatter(z_curr[0], z_curr[1], color='r', marker='.')
            for i in range(len(z)//config['step']):
                z_next = dynamics.f(z_curr)
                ax2.plot([z_curr[0], z_next[0]], [z_curr[1], z_next[1]], color='black')
                z_curr = z_next
            ax2.scatter(z_curr[0], z_curr[1], color='g', marker='x')


        if args.output_dir:
            plt.savefig(os.path.join(args.output_dir, args.id + "_trajectories.png"))
        else:
            plt.savefig(os.path.join(config['output_dir'], "trajectories.png"))
        plt.close()
    
    if args.mode == "field" or args.mode == "all":
        fig, ax = plt.subplots(figsize=(10,5))
        ax1 = plt.subplot(1,2,1)
        ax2 = plt.subplot(1,2,2)

        ax1.set_title("Encoded GT Vector Field")
        ax2.set_title("Latent Dynamics Vector Field")

        for idx in tqdm(idxes):
            traj = dataset[idx]
            z = dynamics.encode(traj)
            
            encoded_start = z[0,:]
            encoded_end = z[config['step'],:]
            ax1.arrow(encoded_start[0], encoded_start[1], encoded_end[0]-encoded_start[0], encoded_end[1]-encoded_start[1], color='black', head_width=1e-2)
            predicted_end = dynamics.f(encoded_start)
            ax2.arrow(encoded_start[0], encoded_start[1], predicted_end[0]-encoded_start[0], predicted_end[1]-encoded_start[1], color='black', head_width=1e-2)

        if args.output_dir != "":
            plt.savefig(os.path.join(args.output_dir, args.id + "_field.png"))
        else:
            plt.savefig(os.path.join(config['output_dir'], "field.png"))
        
