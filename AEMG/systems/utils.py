from AEMG.systems.pendulum import Pendulum
from AEMG.systems.physics_pendulum import PhysicsPendulum
from AEMG.systems.hopper import Hopper

def get_system(name, **kwargs):
    if name == "pendulum":
        system = Pendulum(**kwargs)
    elif name == "physics_pendulum":
        system = PhysicsPendulum(**kwargs)
    elif name == "hopper":
        system = Hopper(**kwargs)
    else:
        print("That system does not exist!")
        raise NotImplementedError
    return system