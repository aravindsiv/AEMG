import sys
import os
from tqdm import tqdm
sys.path.append(os.environ["DIRTMP_PATH"]+"examples/tripods/")
import TimeMap
import numpy as np 
np.set_printoptions(suppress=True)
np.random.seed(101193)
import matplotlib.pyplot as plt

from pendulum import PendulumNoCtrl

time = 5
time_step = 0.5
num_steps = int(time/time_step)

TM = TimeMap.TimeMap("pendulum_lc", time_step,
                     "examples/tripods/pendulum_lc.yaml")

env = PendulumNoCtrl()

num_points = int(1e5)
data = np.zeros((int(num_points*num_steps),4))
print(data.shape)

for i in tqdm(range(num_points)):
    # data[i*num_steps,:2] = env.sample_state()
    # for j in range(1,time,time_step):
    #     state = list(data[i*num_steps + j-1, :2])
    #     data[i*num_steps + j-1, 2:] = TM.pendulum_lqr(state)
    #     data[i*num_steps + j, :2] = data[i*num_steps + j-1, 2:]
    # state = list(data[i*num_steps + time - 1, :2])
    # data[i*num_steps + time - 1, 2:] = TM.pendulum_lqr(state)
    # Get the full trajectory
    traj = [env.sample_state()]
    for j in range(num_steps):
        state = list(traj[j])
        # traj.append(TM.pendulum_lqr(state))
        traj.append(TM.pendulum_no_ctrl(state))
    # Each pair of consecutive states is a data point
    for j in range(num_steps):
        data[i*num_steps + j, :2] = traj[j]
        data[i*num_steps + j, 2:] = traj[j+1]


# plt.figure(figsize=(8,8))
# plt.scatter(data[:,0],data[:,1],c='r')
# plt.scatter(data[:,2],data[:,3],c='b')
# plt.xlim([-np.pi,np.pi])
# plt.ylim([-2*np.pi,2*np.pi])
# plt.show()

# Save the data
np.savetxt("pendulum_data.txt",data,delimiter=",")
