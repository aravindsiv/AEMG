import torch

from AEMG.systems.pendulum import Pendulum
from AEMG.systems.ndpendulum import NdPendulum
from AEMG.systems.cartpole import Cartpole
from AEMG.systems.bistable import Bistable
from AEMG.systems.N_CML import N_CML
from AEMG.systems.leslie_map import Leslie_map
from AEMG.systems.humanoid import Humanoid
from AEMG.systems.trifinger import Trifinger

def get_system(name, dims=10, **kwargs):
    if name == "pendulum":
        system = Pendulum(**kwargs)
    elif name == "ndpendulum" and dims is not None:
        system = NdPendulum(dims, **kwargs)
    elif name == "cartpole":
        system = Cartpole(**kwargs)
    elif name == "bistable":
        system = Bistable(**kwargs)
    elif name == "N_CML":
        system = N_CML(**kwargs)
    elif name == "leslie_map":
        system = Leslie_map(**kwargs)
    elif name == "humanoid":
        system = Humanoid(**kwargs)
    elif name == "trifinger":
        system = Trifinger(**kwargs)
    else:
        print("That system does not exist!")
        raise NotImplementedError
    return system

def multi_dim_tensor_cartesian(a, b):
    a_ = torch.reshape(torch.tile(a, [1, b.shape[0]]), (a.shape[0] * b.shape[0], a.shape[1]))
    b_ = torch.tile(b, [a.shape[0], 1])

    return torch.concat((a_, b_), dim=1)