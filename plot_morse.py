
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

class BoxCenter:
    def __init__(self,corner_low,corner_high,morse_set):
        self.corner_low = corner_low
        self.corner_high = corner_high
        self.morse_set = morse_set

box_dims = []
box_corners = []
box_data = defaultdict(list)
with open(roa_file, 'r') as f:
    line_counter = 0
    for l in f:
        if line_counter == 1:
            box_dims = [float(e) for e in l.split(",")]
        elif line_counter > 2 and not l.startswith("Tile"):
            raw = [float(e) for e in l.split(",")]
            box_data[int(raw[1])].append(raw[2:])
            corner_low = np.array([raw[2],raw[3]])
            corner_high = np.array([raw[4],raw[5]])
            box_corners.append(BoxCenter(corner_low, corner_high, int(raw[1])))
        line_counter += 1

print("Number of boxes: ", len(box_corners))
num_morse_sets = len(box_data.keys())
cmap = matplotlib.cm.get_cmap('viridis', 256)
cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_morse_sets-1)

def find_box(x, y):
    # Find the box that contains the point (x,y)
    for box in box_corners:
        if x >= box.corner_low[0] and x <= box.corner_high[0] and y >= box.corner_low[1] and y <= box.corner_high[1]:
            return box.morse_set
    return -1


xs = np.linspace(-np.pi, np.pi, 41)
ys = np.linspace(-2*np.pi, 2*np.pi, 41)

plt.figure(figsize=(8,8))
plt.xlim(-np.pi, np.pi)
plt.ylim(-2*np.pi, 2*np.pi)

for i in tqdm(range(xs.shape[0])):
    for j in range(ys.shape[0]):
        state = [xs[i], ys[j]]
        x = env.transform(state)
        x = (x - tf_bounds[:,0]) / (tf_bounds[:,1] - tf_bounds[:,0])
        x = torch.tensor(x, dtype=torch.float32)
        z = encoder(x)
        z = z.detach().numpy()
        morse_set = find_box(z[0], z[1])
        color = cmap(cmap_norm(morse_set))
        if morse_set != -1:
            plt.scatter(xs[i], ys[j], color=color,alpha=0.4,marker="s")
        else:
            plt.scatter(xs[i], ys[j], color="black",alpha=0.4,marker="s")

plt.xlabel("theta")
plt.ylabel("thetadot")
plt.title("Morse sets")
plt.show()

