import numpy as np 
from AEMG.systems.system import BaseSystem

class NdPendulum(BaseSystem):
    def __init__(self, dims=9, **kwargs):
        super().__init__(**kwargs)
        self.name = "ndpendulum"

        self.state_bounds = np.array([[-np.pi, np.pi], [-2*np.pi, 2*np.pi]])
        
        np.random.seed(dims)
        # Find largest N such that dims < N*N
        N = int(np.ceil(np.sqrt(dims)))
        # Get a grid of N*N points
        th = np.linspace(-np.pi, np.pi, N)
        thdot = np.linspace(-2*np.pi, 2*np.pi, N)

        self.centers = np.random.choice(np.array(np.meshgrid(th, thdot)).T.reshape(-1, 2), size=dims, replace=False)

        self.l = 0.5
    
    def transform(self,s):
        pt = np.zeros((self.centers.shape[0],))
        for i in range(self.centers.shape[0]):
            pt[i] = np.exp(-np.linalg.norm(s-self.centers[i])**2)
        return pt
    
    def inverse_transform(self,s):
        return s
    
    def get_true_bounds(self):
        return NotImplementedError
    
    def get_bounds(self):
        return NotImplementedError