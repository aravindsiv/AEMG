from AEMG.systems.pendulum import Pendulum

def get_system(name, **kwargs):
    if name == "pendulum":
        system = Pendulum(**kwargs)
    else:
        print("That system does not exist!")
        raise NotImplementedError
    return system