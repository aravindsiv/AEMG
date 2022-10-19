import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

batch_size = 32
num_epochs = int(1e2)

from pendulum import PendulumNoCtrl

env = PendulumNoCtrl()

# Generate data
dat = []
for i in tqdm(range(10000)):
    s = env.sample_state()
    dat.append(env.transform(s))

dat = np.array(dat,dtype=np.float32)
print(dat.shape)

bounds = env.get_transformed_state_bounds()

for i in range(bounds.shape[0]):
    dat[:,i] = (dat[:,i] - bounds[i,0]) / (bounds[i,1] - bounds[i,0])

'''
fname = "10k_trajs.txt"

dat = []

with open(fname) as f:
    for l in f:
        if l != "\n":
            line = np.array([float(e) for e in l.split("\t")])
            if line[-1] < 3:
                dat.append(line[:-1])

dat = np.vstack(dat)
print(dat.shape)
'''
high_dims = dat.shape[1]
low_dims  = 2

class DynamicsDataset(torch.utils.data.Dataset):
  def __init__(self,X):
    if not torch.is_tensor(X):
      self.X = torch.from_numpy(X).float()
  
  def __len__(self):
    return len(self.X)
  
  def __getitem__(self,i):
    return self.X[i]

class AutoEncoder(nn.Module):
    def __init__(self,input_shape,lower_shape):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.ReLU(True), 
            nn.Linear(32, 16), 
            nn.ReLU(True), 
            nn.Linear(16, lower_shape),
            nn.Tanh())
        self.decoder = nn.Sequential(
            nn.Linear(lower_shape, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True), 
            nn.Linear(32, input_shape), 
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

dat_tr = dat[0:8000]
dat_te = dat[8000:]

trainloader = torch.utils.data.DataLoader(dat_tr, batch_size=batch_size, shuffle=True)
valloader = torch.utils.data.DataLoader(dat_te, batch_size=batch_size, shuffle=True)

model = AutoEncoder(high_dims,low_dims)
loss_function = nn.MSELoss(reduce='mean')
optimizer = torch.optim.Adam(model.parameters())

train_losses = []
val_losses = []
for epoch in tqdm(range(0,num_epochs)):
    print("Epoch: ",epoch)

    current_train_loss = 0.0
    epoch_train_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs = data

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs,inputs)
        loss.backward()
        optimizer.step()

        current_train_loss += loss.item()
        epoch_train_loss += loss.item()
        if (i+1) % 100 == 0:
            print("Loss after mini-batch ",i+1,": ", current_train_loss)
            current_train_loss = 0.0
    
    epoch_train_loss = epoch_train_loss / len(trainloader)
    train_losses.append(epoch_train_loss)

    epoch_val_loss = 0.0
    model.eval()
    for i, data in enumerate(valloader, 0):
        inputs = data

        outputs = model(inputs)
        loss = loss_function(outputs,inputs)
        epoch_val_loss += loss.item()
    epoch_val_loss = epoch_val_loss / len(valloader)
    val_losses.append(epoch_val_loss)

plt.figure(figsize=(8,8))
plt.grid()
plt.plot(train_losses,label='Train Loss')
plt.plot(val_losses,label='Val Loss')
plt.legend(loc='best')
plt.ylim(0,0.1)
plt.show()
