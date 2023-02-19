import numpy as np

class BaseSystem:
    def __init__(self, **kwargs):
        self.name = "base_system"

        self.state_bounds = np.array([[-1, 1], [-1, 1]])

    def sample_state(self):
        return np.random.uniform(self.state_bounds[:,0], self.state_bounds[:,1])

    def get_bounds(self):
        return self.state_bounds
    
    def transform(self, s):
        return s
    
    def inverse_transform(self, s):
        return s