import numpy as np 
from AEMG.systems.system import BaseSystem

class Cartpole(BaseSystem):
    def __init__(self,**kwargs):
        # x, theta, xdot, thetadot
        super().__init__(**kwargs)
        self.name = "cartpole"

    def transform(self,x):
        theta = x[1]
        # x[1] = np.arctan2(np.sin(theta),np.cos(theta))
        x = [x[0], np.cos(theta), np.sin(theta), x[2], x[3]]
        return x
    
    def achieved_goal(self,s):
        diff = np.sqrt(s[1]*s[1] + s[3]*s[3])
        return (diff < 0.1)
    
    def get_true_bounds(self):
        # bounds normalized by arctan2
        return np.array([[-1, 1], [-np.pi, np.pi], [-1.27, 1.27], [-3.04, 3.04]])
    
    def get_bounds(self):
        return NotImplementedError