import numpy as np 
from AEMG.systems.system import BaseSystem

class Pendulum(BaseSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "pendulum"

        self.state_bounds = np.array([[-np.pi, np.pi], [-2*np.pi, 2*np.pi]])
        
        self.l = 0.5
    
    def transform(self,s):
        x = self.l * np.sin(s[0])
        y = self.l * np.cos(s[0])
        xdot = self.l * np.cos(s[0]) * s[1]
        ydot = -self.l * np.sin(s[0]) * s[1]
        return np.array([x,y,xdot,ydot])
    
    def inverse_transform(self,s):
        theta = np.arctan2(s[0],s[1])
        thetadot = -(-s[1] * s[2] + s[0] * s[3]) / (s[0]*s[0] + s[1]*s[1])
        return np.array([theta,thetadot])
    
    def get_true_bounds(self):
        return self.state_bounds
    
    def get_bounds(self):
        return np.array([[-self.l, self.l], [-self.l, self.l], [-2*self.l*np.pi, 2*self.l*np.pi], [-2*self.l*np.pi, 2*self.l*np.pi]])
