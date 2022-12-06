import numpy as np 
from AEMG.systems.system import BaseSystem

class PhysicsPendulum(BaseSystem):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.name = "physics_pendulum"

    def transform(self,s):
        # The original state is [x, xdot, cos, sin, thetadot]
        # We want to transform it to [x, xdot, theta, thetadot]
        theta = np.arctan2(s[2],s[3])
        thetadot = s[4]
        return np.array([s[0],s[1],theta,thetadot])

    def inverse_transform(self,s):
        # The original state is [x, xdot, cos, sin, thetadot]
        # We want to transform it to [x, xdot, theta, thetadot]
        cos = np.cos(s[2])
        sin = np.sin(s[2])
        return np.array([s[0],s[1],cos,sin,s[3]])
    
    def get_true_bounds(self):
        return NotImplementedError
    
    def get_bounds(self):
        return NotImplementedError