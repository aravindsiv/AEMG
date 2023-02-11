import numpy as np 
from AEMG.systems.system import BaseSystem

class Cartpole(BaseSystem):
    def __init__(self,**kwargs):
        # x, theta, xdot, thetadot
        super().__init__(**kwargs)
        self.name = "cartpole"
    
    def achieved_goal(self,s):
        diff = np.sqrt(s[1]*s[1] + s[3]*s[3])
        return (diff < 0.1)
    
    def get_true_bounds(self):
        return NotImplementedError
    
    def get_bounds(self):
        return NotImplementedError