import numpy as np 
from AEMG.systems.system import BaseSystem

class Hopper(BaseSystem):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.name = "hopper"

    def transform(self,s):
        return s

    def inverse_transform(self,s):
        return s
    
    def get_true_bounds(self):
        return NotImplementedError
    
    def get_bounds(self):
        return NotImplementedError