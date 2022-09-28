import numpy as np

class PendulumNoCtrl:
    def __init__(self):
        self.state_bounds = np.array([[-np.pi, np.pi], [-2*np.pi, 2*np.pi]])
        
        self.dt = 0.01

        # Some constants
        self.m = 0.15
        self.l = 0.5
        self.g = 9.81
        self.friction = 0.1

        self.inertia = self.m * self.l**2

    def _enforce(self, s):
        while s[0] < -np.pi:
            s[0] += 2*np.pi
        while s[0] > np.pi:
            s[0] -= 2*np.pi
        s[0] = np.clip(s[0], -np.pi, np.pi)
        s[1] = np.clip(s[1], -2*np.pi, 2*np.pi)
        return s
    
    def sample_state(self):
        return np.random.uniform(self.state_bounds[:,0], self.state_bounds[:,1])
    
    def step(self, s):
        u = 0.

        thetadot  = np.copy(s[1])
        thetaddot = self.g / self.l * np.sin(s[0])
        thetaddot -= self.friction / self.inertia * s[1]

        s += np.array([thetadot, thetaddot]) * self.dt

        s = self._enforce(s)

        return np.copy(s)

    def transform(self,s):
        # Transform from theta, thetadot to x,y,xdot,ydot
        x = self.l * np.sin(s[0])
        y = -self.l * np.cos(s[0])
        xdot = self.l * np.cos(s[0]) * s[1]
        ydot = self.l * np.sin(s[0]) * s[1]
        return np.array([x,y,xdot,ydot])
    
    def inverse_transform(self,s):
        # Transform from x,y,xdot,ydot to theta, thetadot
        theta = np.arcsin(s[0] / self.l)
        thetadot = s[2] / self.l * np.cos(theta)
        return np.array([theta, thetadot])

    def get_transformed_state_bounds(self):
        return np.array([[-self.l, self.l], [-self.l, self.l], [-2*self.l*self.g, 2*self.l*self.g], [-2*self.l*self.g, 2*self.l*self.g]])