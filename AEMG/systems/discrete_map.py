import numpy as np 
from AEMG.systems.system import BaseSystem

class DiscreteMap(BaseSystem):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.name = "discrete_map"

    def f(self, X):
        Y0 = np.array([np.arctan(2*X[0])])
        Y1 = X[1::]/2
        return np.concatenate((Y0, Y1))

    def transform(self, s):
        return s
    
    def get_true_bounds(self):
        return [-2,-2,2,2]
    
    def get_bounds(self):
        return NotImplementedError