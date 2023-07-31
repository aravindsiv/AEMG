import numpy as np 
from AEMG.systems.system import BaseSystem

class Humanoid(BaseSystem):
    def __init__(self,**kwargs):
        self.name = "humanoid"
    
    def get_true_bounds(self):
        return NotImplementedError
    
    def get_bounds(self):
        return NotImplementedError