from AEMG.systems.pendulum import Pendulum
from AEMG.systems.ndpendulum import NdPendulum
from AEMG.systems.cartpole import Cartpole
from AEMG.systems.bistable import Bistable
from AEMG.systems.N_CML import N_CML
from AEMG.systems.leslie_map import Leslie_map
from AEMG.systems.humanoid import Humanoid
from AEMG.systems.trifinger import Trifinger
from AEMG.systems.bistable_rot import Bistable_Rot
from AEMG.systems.unifinger import Unifinger
from AEMG.systems.pendulum3links import Pendulum3links
from AEMG.systems.satellite import Satellite

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
    elif name == "bistable_rot":
        system = Bistable_Rot(**kwargs)
    elif name == "unifinger":
        system = Unifinger(**kwargs)
    elif name == "threelinkpendulum":
        system = Pendulum3links(**kwargs)
    elif name == "satellite":
        system = Satellite(**kwargs)
    else:
        print("That system does not exist!")
        raise NotImplementedError
    return system
