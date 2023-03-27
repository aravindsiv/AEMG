import sys 
import os
from tqdm import tqdm
from AEMG.systems.utils import get_system

sys.path.append(os.environ["DIRTMP_PATH"]+"/examples/tripods/")
import TimeMap

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
    parser.add_argument('--time', help='Trajectory length', type=float, default=1.0)
    parser.add_argument('--time_step', help='Time step', type=float, default=0.1)
    parser.add_argument('--num_trajs', help='Number of trajectories', type=int, default=10)
    parser.add_argument('--save_dir', help='Save directory', type=str, default='/data/discrete_map')
    parser.add_argument('--system', help='Select the system', type=str, default='discrete_map')

    args = parser.parse_args()
    
    system = get_system(args.system)

    num_trajs = args.num_trajs
    num_steps = int(args.time/args.time_step)

    true_bounds = system.get_true_bounds()

    dim = len(true_bounds) // 2

    lower_bounds = true_bounds[0:dim]
    upper_bounds = true_bounds[dim::]


    X = sample_points(lower_bounds, upper_bounds, num_trajs)

    if args.system == "ndpendulum":

        TM = TimeMap.TimeMap("pendulum_lc", args.time_step,
                                "examples/tripods/pendulum_lc.yaml")
        f = TM.pendulum_lqr

    elif args.system == "discrete_map":
        f = system.f


    # Create the data directory if it doesn't exist
    # save_dir = args.save_dir

    save_dir = os.getcwd() + args.save_dir
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    counter = 0
    for x in tqdm(X):
        # Get the full trajectory
        traj = [x]
        # traj = [system.transform(x)]
        state_temp = x
        for k in range(num_steps):
            state_temp = f(state_temp)
            traj.append(state_temp)
            # traj.append(system.transform(state_temp))
        
        traj = np.array(traj) 
        np.savetxt(f"{save_dir}/{counter}.txt",traj,delimiter=",")
        counter += 1 