from AEMG.systems.pendulum import Pendulum
from AEMG.systems.ndpendulum import NdPendulum
from AEMG.systems.physics_pendulum import PhysicsPendulum
from AEMG.systems.hopper import Hopper
from AEMG.systems.cartpole import Cartpole
from AEMG.systems.discrete_map import DiscreteMap

def get_system(name, dims=None, **kwargs):
    if name == "pendulum":
        system = Pendulum(**kwargs)
    elif name == "ndpendulum" and dims is not None:
        system = NdPendulum(dims, **kwargs)
    elif name == "physics_pendulum":
        system = PhysicsPendulum(**kwargs)
    elif name == "hopper":
        system = Hopper(**kwargs)
    elif name == "cartpole":
        system = Cartpole(**kwargs)
    elif name == "discrete_map":
        system = DiscreteMap(**kwargs)
    else:
        print("That system does not exist!")
        raise NotImplementedError
    return system