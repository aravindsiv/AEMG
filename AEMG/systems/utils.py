from AEMG.systems.pendulum import Pendulum
from AEMG.systems.ndpendulum import NdPendulum
from AEMG.systems.cartpole import Cartpole
from AEMG.systems.bistable import Bistable
from AEMG.systems.N_CML import N_CML
from AEMG.systems.leslie_map import Leslie_map

def get_system(name, dims=10, **kwargs):
    if name == "pendulum":
        system = Pendulum(**kwargs)
    elif name == "ndpendulum" and dims is not None:
        system = NdPendulum(dims, **kwargs)
    elif name == "hopper":
        system = Hopper(**kwargs)
    elif name == "cartpole":
        system = Cartpole(**kwargs)
    elif name == "bistable":
        system = Bistable(**kwargs)
    elif name == "N_CML":
        system = N_CML(**kwargs)
    elif name == "leslie_map":
        system = Leslie_map(**kwargs)
    else:
        print("That system does not exist!")
        raise NotImplementedError
    return system
