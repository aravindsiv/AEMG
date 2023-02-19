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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', help='Trajectory length', type=float, default=2.0)
    parser.add_argument('--time_step', help='Time step', type=float, default=0.1)
    parser.add_argument('--num_trajs', help='Number of trajectories', type=int, default=1000)
    parser.add_argument('--mode', help='Mode', type=str, default='none')
    parser.add_argument('--save_dir', help='Save directory', type=str, default='data')

    args = parser.parse_args()

    TM = TimeMap.TimeMap("pendulum_lc", args.time_step,
                            "examples/tripods/pendulum_lc.yaml")
    
    system = get_system("pendulum")

    num_trajs = args.num_trajs
    num_steps = int(args.time/args.time_step)

    theta = np.linspace(-np.pi, np.pi, int(np.ceil(np.sqrt(num_trajs))))
    dtheta = np.linspace(-2*np.pi, 2*np.pi, int(np.ceil(np.sqrt(num_trajs))))

    # Create the data directory if it doesn't exist
    save_dir = args.save_dir + "/pendulum_" + args.mode + '_' + str(args.time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    counter = 0
    for i in tqdm(range(theta.shape[0])):
        for j in range(dtheta.shape[0]):
            # Get the full trajectory
            traj = [(theta[i], dtheta[j])]
            for k in range(num_steps):
                state = traj[k]
                if args.mode == 'lqr':
                    traj.append(TM.pendulum_lqr(state))
                else:
                    traj.append(TM.pendulum_no_ctrl(state))
            
            traj = np.array(traj)
            np.savetxt(f"{save_dir}/{counter}.txt",traj,delimiter=",")
            counter += 1


    