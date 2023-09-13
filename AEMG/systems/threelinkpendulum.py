from AEMG.systems.system import BaseSystem

class ThreeLinkPendulum(BaseSystem):
    def __init__(self,**kwargs):
        self.name = "threelinkpendulum"
        self.state_bounds = NotImplementedError
    
    # def get_true_bounds(self):
    #     return NotImplementedError
    
    # def get_bounds(self):
    #     return NotImplementedError