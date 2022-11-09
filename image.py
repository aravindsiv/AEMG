import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from pendulum import PendulumNoCtrl
from tqdm import tqdm
import pickle as pkl

from models import *
from config import *

env = PendulumNoCtrl()
tf_bounds = env.get_transformed_state_bounds()
gt_bounds = env.state_bounds

MODEL_PATH = f'{ROOT_PATH}/models'
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

encoder = Encoder(high_dims,low_dims)
dynamics = LatentDynamics(low_dims)
decoder = Decoder(low_dims,high_dims)

encoder = torch.load(f"{MODEL_PATH}/encoder_{mode}_0.1_20steps_1M{'_warmup' if warmup > 0 else ''}_64.pt",map_location=torch.device('cpu'))
decoder = torch.load(f"{MODEL_PATH}/decoder_{mode}_0.1_20steps_1M{'_warmup' if warmup > 0 else ''}_64.pt",map_location=torch.device('cpu'))
dynamics = torch.load(f"{MODEL_PATH}/dynamics_{mode}_0.1_20steps_1M{'_warmup' if warmup > 0 else ''}_64.pt",map_location=torch.device('cpu'))

def f(Z):
    # Get the decoding of this state.
    x = decoder(Z)
    # Bring x to R^4
    x = x * (tf_bounds[:,1] - tf_bounds[:,0]) + tf_bounds[:,0]
    # if not env.valid_state(x):
    #    return None
    assert(env.valid_state(x))
    return dynamics(x)

'''
def f(g, X):
    # Now x is in [-1,1]^2
    x = torch.tensor([0.1,0.1],dtype=torch.float32)
    # This brings x to [0,1]^4
    x = model.decoder(x)
    x = x.detach().numpy()
    # This brings x to R^4
    x = x * (tf_bounds[:,1] - tf_bounds[:,0]) + tf_bounds[:,0]
    # This brings x to [-pi,pi] x [-2pi,2pi]
    x = env.inverse_transform(x)
    # This computes the image (using TimeMap)
    x = g(X)
    # Bring x to R^4
    x = env.transform(x)
    # Bring x to [0,1]^4
    x = (x - tf_bounds[:,0]) / (tf_bounds[:,1] - tf_bounds[:,0])
    x = torch.from_numpy(x).float()
    # Bring x to [-1,1]^2
    x = model.encoder(x).detach().numpy()
    return x
'''