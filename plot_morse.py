
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import sys
import os

from pendulum import PendulumNoCtrl
from tqdm import tqdm
from collections import defaultdict
from matplotlib.patches import Rectangle
import matplotlib

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

roa_file = "RoA_"+mode+".csv"

box_dims = []
box_data = defaultdict(list)
with open(roa_file, 'r') as f:
    line_counter = 0
    for l in f:
        if line_counter == 1:
            box_dims = [float(e) for e in l.split(",")]
        elif line_counter > 2 and not l.startswith("Tile"):
            raw = [float(e) for e in l.split(",")]
            box_data[int(raw[1])].append(raw[2:])
        line_counter += 1

num_morse_sets = len(box_data.keys())

plt.figure(figsize=(8,8))
plt.xlim(-1,1)
plt.ylim(-1,1)

cmap = matplotlib.cm.get_cmap('viridis', 256)
cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_morse_sets-1)


for k in box_data.keys():
    if k < 0: continue
    clr = matplotlib.colors.to_hex(cmap(cmap_norm(k)))
    for elt in box_data[k]:
        rect = Rectangle((elt[0], elt[1]), box_dims[0], box_dims[1], edgecolor='black', facecolor=clr, alpha=0.4)
        plt.gca().add_patch(rect)

plt.show()

plt.figure(figsize=(8,8))
plt.xlim(-np.pi, np.pi)
plt.ylim(-2*np.pi, 2*np.pi)

for k in box_data.keys():
    if k < 0: continue
    clr = matplotlib.colors.to_hex(cmap(cmap_norm(k)))
    for elt in box_data[k]:
        rect_center = 0.5 * np.array([elt[0] + elt[2], elt[1] + elt[3]])
        x = torch.tensor(rect_center, dtype=torch.float32)
        x = decoder(x)
        x = x.detach().numpy()
        x = x * (tf_bounds[:,1] - tf_bounds[:,0]) + tf_bounds[:,0]
        x = env.inverse_transform(x)
        plt.scatter(x[0], x[1], color=clr, alpha=0.4)

plt.show()
