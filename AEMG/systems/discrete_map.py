import numpy as np 
from AEMG.systems.system import BaseSystem

class Discrete_map(BaseSystem):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.name = "discrete_map"
    
    def get_true_bounds(self):
        return NotImplementedError
    
    def get_bounds(self):
        return NotImplementedError