import sys
import os
sys.path.append(os.environ["DIRTMP_PATH"]+"examples/tripods/")
import TimeMap
import numpy as np 
import matplotlib.pyplot as plt

from pendulum import PendulumNoCtrl

time = 1

TM = TimeMap.TimeMap("pendulum_lc", time,
                     "examples/tripods/pendulum_lc.yaml")

env = PendulumNoCtrl()

num_points = 1000
data = np.zeros((num_points,4))

for i in range(num_points):
    data[i,:2] = env.sample_state()
    state = list(data[i,:2])
    data[i,2:] = TM.pendulum_lqr(state)

# plt.figure(figsize=(8,8))
# plt.scatter(data[:,0],data[:,1],c='r')
# plt.scatter(data[:,2],data[:,3],c='b')
# plt.xlim([-np.pi,np.pi])
# plt.ylim([-2*np.pi,2*np.pi])
# plt.show()

# Save the data
np.savetxt("pendulum_data.txt",data,delimiter=",")
