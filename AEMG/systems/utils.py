from AEMG.systems.pendulum import Pendulum
from AEMG.systems.ndpendulum import NdPendulum
from AEMG.systems.hopper import Hopper
from AEMG.systems.cartpole import Cartpole
from AEMG.systems.bistable import Bistable

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
    else:
        print("That system does not exist!")
        raise NotImplementedError
    return system
