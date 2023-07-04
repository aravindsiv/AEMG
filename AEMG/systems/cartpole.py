import numpy as np 
from AEMG.systems.system import BaseSystem

class Cartpole(BaseSystem):
    def __init__(self,**kwargs):
        # x, theta, xdot, thetadot
        super().__init__(**kwargs)
        self.name = "cartpole"
    
    def transform(self,x):
        return np.array([x[0],np.cos(x[1]),np.sin(x[1]),x[2],x[3]])

    def inverse_transform(self,x):
        return np.array([x[0],np.arctan2(x[2],x[1]),x[3],x[4]])
    
    def achieved_goal(self,s):
        diff = np.sqrt(s[1]*s[1] + s[3]*s[3])
        return (diff < 0.1)
    
    def get_true_bounds(self):
        return NotImplementedError
    
    def get_bounds(self):
        return NotImplementedError