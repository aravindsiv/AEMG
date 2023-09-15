import os

import numpy as np

path = '/Users/htnamus/All_Stuff/Programming_Stuff/AEMG/examples/data/pendulum_roa_unwrapped.txt'

output_dir_path = '/Users/htnamus/All_Stuff/Programming_Stuff/AEMG/examples/data/pendulum_roa'
labels_dir_path = '/Users/htnamus/All_Stuff/Programming_Stuff/AEMG/examples/data'

if __name__ == '__main__':
    with open(path) as file:
        data = file.read()
        data = data.split('\n')

    trajectory = []
    trajectories = []
    success_details = []

    for line in data:
        if line == '':
            if len(trajectory): trajectories.append(trajectory)
            trajectory = []
            continue

        line = line.split(' ')
        for x in line:
            if x == '':
                line.remove(x)
        line = np.array([float(x) for x in line if len(x) > 0])



        trajectory.append(line)
    if len(trajectory): trajectories.append(trajectory)

    for idx, trajectory in enumerate(trajectories):
        final_state = trajectory[-1]
        is_success = np.linalg.norm(final_state) <  0.1

        success_details.append(1 if is_success else 0)

        traj_file_name = 'state_trajectory_' + str(idx) + '.txt'

        traj_file_path = os.path.join(output_dir_path, traj_file_name)

        with open(traj_file_path, 'w') as file:
            for state in trajectory:
                file.write(', '.join([str(val) for val in state]) + '\n')

    success_details_file_name = 'pendulum_roa_success.txt'

    labels_file_path = os.path.join(labels_dir_path, success_details_file_name)
    with open(labels_file_path, 'w') as file:
        for idx, final_state in enumerate(success_details):
            traj_file_name = 'state_trajectory_' + str(idx) + '.txt'
            file.write(traj_file_name + ', ' + str(final_state) + '\n')
