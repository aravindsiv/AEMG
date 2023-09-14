import numpy as np 
from AEMG.systems.system import BaseSystem

class Satellite(BaseSystem):
    def __init__(self,**kwargs):
        self.name = "satellite"
        self.state_bounds = NotImplementedError
    
    # def get_true_bounds(self):
    #     return NotImplementedError
    
    # def get_bounds(self):
    #     return NotImplementedError