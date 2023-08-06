import numpy as np

class BaseSystem:
    def __init__(self, **kwargs):
        self.name = "base_system"

        self.state_bounds = np.array([[-1, 1], [-1, 1]])
    
    def f(self,s):
        return s

    # def sample_state(self):
    #     return np.random.uniform(self.state_bounds[:,0], self.state_bounds[:,1])

    def sample_state(self, num_pts):
        sample_ = np.random.uniform(self.get_true_bounds()[:,0], self.get_true_bounds()[:,1], size=(num_pts, self.dimension()))
        return self.transform(sample_)
    
    def get_bounds(self):
        return self.state_bounds
    
    def get_true_bounds(self):
        return self.state_bounds
    
    def dimension(self):
        return self.get_true_bounds().shape[0]
    
    def transform(self, s):
        return s
    
    def inverse_transform(self, s):
        return s