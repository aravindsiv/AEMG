import numpy as np 
import torch
from torch import nn 
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

from pendulum import PendulumNoCtrl

from torch.utils.data import DataLoader

class DynamicsDataset(torch.utils.data.Dataset):
    def __init__(self,Xt, Xnext):
        if not torch.is_tensor(Xt):
            self.Xt = torch.from_numpy(Xt).float()
            self.Xnext = torch.from_numpy(Xnext).float()
    
    def __len__(self):
        return len(self.Xt)
    
    def __getitem__(self,i):
        return self.Xt[i], self.Xnext[i]

class Encoder(nn.Module):
    def __init__(self,input_shape,lower_shape):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.ReLU(True), 
            nn.Linear(32, 16), 
            nn.ReLU(True), 
            nn.Linear(16, lower_shape),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder(nn.Module):
    def __init__(self,lower_shape,input_shape):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(lower_shape, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True), 
            nn.Linear(32, input_shape),
            nn.Sigmoid() 
            )

    def forward(self, x):
        x = self.decoder(x)
        return x

class LatentDynamics(nn.Module):
    # Takes as input an encoding and returns a latent dynamics
    # vector which is just another encoding
    def __init__(self,lower_shape):
        super(LatentDynamics, self).__init__()
        self.dynamics = nn.Sequential(
            nn.Linear(lower_shape, 16),
            nn.ReLU(True),
            nn.Linear(16, 16),
            nn.ReLU(True), 
            nn.Linear(16, lower_shape), 
            nn.Tanh()
            )
    
    def forward(self, x):
        x = self.dynamics(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self,input_shape,lower_shape):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_shape,lower_shape)
        self.decoder = Decoder(lower_shape,input_shape)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DynamicsAE(nn.Module):
    def __init__(self,encoder_model, lower_shape):
        super(DynamicsAE, self).__init__()
        self.encoder = encoder_model
        self.dynamics = LatentDynamics(lower_shape)

    def forward(self, x):
        x = self.encoder(x)
        x = self.dynamics(x)
        return x

env = PendulumNoCtrl()
bounds = env.get_transformed_state_bounds()

raw_dat = np.loadtxt("pendulum_data.txt",delimiter=",")
print(raw_dat.shape)

dat = np.zeros((raw_dat.shape[0],8))
dataset_size = raw_dat.shape[0]

high_dims = 4
low_dims = 2

batch_size = 512
epochs = int(1e3)

for i in range(dat.shape[0]):
    # Transform the state
    dat[i,:4] = env.transform(raw_dat[i,:2])
    dat[i,:4] = (dat[i,:4] - bounds[:,0])/(bounds[:,1] - bounds[:,0])
    # Transform the next state
    dat[i,4:] = env.transform(raw_dat[i,2:])
    dat[i,4:] = (dat[i,4:] - bounds[:,0])/(bounds[:,1] - bounds[:,0])


dat_tr = dat[:int(0.8*dataset_size),:]
dat_te = dat[int(0.8*dataset_size):,:]
raw_dat_te = raw_dat[int(0.8*dataset_size):,:]

train_dataset = DynamicsDataset(dat_tr[:,:high_dims],dat_tr[:,high_dims:])
val_dataset = DynamicsDataset(dat_te[:,:high_dims],dat_te[:,high_dims:])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

model = AutoEncoder(high_dims,low_dims)
dynamics_model = DynamicsAE(model.encoder,low_dims)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

train_losses = []
val_losses = []

for epoch in tqdm(range(epochs)):
    current_train_loss = 0.0
    epoch_train_loss = 0.0

    model.train()
    dynamics_model.train()
    for i, data in enumerate(train_loader, 0):
        currents, nexts  = data

        optimizer.zero_grad()

        outputs_current = model(currents)
        loss_ae = criterion(outputs_current, currents)
        outputs_next = model(nexts)
        loss_ae += criterion(outputs_next, nexts)

        latents = model.encoder(nexts)
        dynamics = dynamics_model(currents)
        loss_dyn = criterion(dynamics,latents)

        loss = loss_ae + loss_dyn
        # loss = loss_ae 
        loss.backward()

        optimizer.step()

        current_train_loss += loss_ae.item() + loss_dyn.item()
        epoch_train_loss += loss_ae.item() + loss_dyn.item()
        if (i+1) % 100 == 0:
            print("Loss after mini-batch ",i+1,": ", current_train_loss)
            current_train_loss = 0.0
    
    epoch_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    epoch_val_loss = 0.0
    model.eval()
    dynamics_model.eval()
    for i, data in enumerate(val_loader, 0):
        currents, nexts  = data

        outputs_current = model(currents)
        loss_ae = criterion(outputs_current, currents)
        outputs_next = model(nexts)
        loss_ae += criterion(outputs_next, nexts)

        latents = model.encoder(nexts)
        dynamics = dynamics_model(currents)
        loss_dyn = criterion(dynamics,latents)

        epoch_val_loss += loss_ae.item() + loss_dyn.item()
        # epoch_val_loss += loss_ae.item() 
    
    epoch_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(epoch_val_loss)

plt.figure(figsize=(8,8))
plt.grid()
plt.plot(train_losses,label='Train Loss')
plt.plot(val_losses,label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Losses")
plt.legend(loc='best')
# plt.ylim(0,0.1)
plt.show()

plt.figure(figsize=(8,8))
plt.scatter(raw_dat_te[:,0],raw_dat_te[:,1],c='r')
plt.scatter(raw_dat_te[:,2],raw_dat_te[:,3],c='b')
plt.xlim(-np.pi,np.pi)
plt.ylim(-2*np.pi,2*np.pi)
plt.xlabel("theta")
plt.ylabel("thetadot")
plt.title("State Space")
plt.show()

with torch.no_grad():
    model.eval()
    dynamics_model.eval()

    # Check the reconstruction
    plt.figure(figsize=(8,8))
    plt.xlim(-np.pi,np.pi)
    plt.ylim(-2*np.pi,2*np.pi)
    z_t = model.encoder(torch.from_numpy(dat_te[:,:high_dims]).float())
    x_t_pred = model.decoder(z_t)
    x_t_pred = x_t_pred.numpy()
    # Unnormalize
    x_t_pred = x_t_pred*(bounds[:,1] - bounds[:,0]) + bounds[:,0]
    pred_plot = np.zeros((x_t_pred.shape[0],2))
    for i in range(x_t_pred.shape[0]):
        pred_plot[i] = env.inverse_transform(x_t_pred[i,:])
    plt.scatter(pred_plot[:,0],pred_plot[:,1],c='r')
    z_t_next = model.encoder(torch.from_numpy(dat_te[:,high_dims:]).float())
    x_t_next_pred = model.decoder(z_t_next)
    x_t_next_pred = x_t_next_pred.numpy()
    # Unnormalize
    x_t_next_pred = x_t_next_pred*(bounds[:,1] - bounds[:,0]) + bounds[:,0]
    pred_plot = np.zeros((x_t_next_pred.shape[0],2))
    for i in range(x_t_next_pred.shape[0]):
        pred_plot[i] = env.inverse_transform(x_t_next_pred[i,:])
    plt.scatter(pred_plot[:,0],pred_plot[:,1],c='b')
    plt.xlabel("theta")
    plt.ylabel("thetadot")
    plt.title("Reconstruction")
    plt.show()

    plt.figure(figsize=(8,8))
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    z_t = model.encoder(torch.from_numpy(dat_te[:,:high_dims]).float())
    z_t = z_t.detach().numpy()
    plt.scatter(z_t[:,0],z_t[:,1],c='r')
    z_tau = model.encoder(torch.from_numpy(dat_te[:,high_dims:]).float())
    z_tau = z_tau.detach().numpy()
    plt.scatter(z_tau[:,0],z_tau[:,1],c='b')
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("Latent space - z_t and z_tau (true)")
    plt.show()

    plt.figure(figsize=(8,8))
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    z_t = model.encoder(torch.from_numpy(dat_te[:,:high_dims]).float())
    z_t = z_t.detach().numpy()
    plt.scatter(z_t[:,0],z_t[:,1],c='r')
    z_tau_pred = dynamics_model(torch.from_numpy(dat_te[:,:high_dims]).float())
    z_tau_pred = z_tau_pred.detach().numpy()
    plt.scatter(z_tau_pred[:,0],z_tau_pred[:,1],c='b')
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("Latent space - z_t and z_tau (predicted)")
    plt.show()