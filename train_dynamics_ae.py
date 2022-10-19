import numpy as np 
import torch
from torch import nn 
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

from pendulum import PendulumNoCtrl

from torch.utils.data import DataLoader

# Set some seeds
np.random.seed(0)
torch.manual_seed(0)

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
            nn.Linear(input_shape, 16),
            nn.ReLU(True), 
            nn.Linear(16, 16), 
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
            nn.Linear(16, 16),
            nn.ReLU(True), 
            nn.Linear(16, input_shape),
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

env = PendulumNoCtrl()
bounds = env.get_transformed_state_bounds()

mode = "lqr"

if mode == "lqr":
    raw_data = np.loadtxt("pendulum_lqr.txt",delimiter=",")
    print(raw_data.shape)

if mode == "none":
    raw_data = np.loadtxt("pendulum_noctrl.txt",delimiter=",")
    print(raw_data.shape)

raw_dat = np.array(raw_data)
print(raw_dat.shape)

dat = np.zeros((raw_dat.shape[0],8))
dataset_size = raw_dat.shape[0]

high_dims = 4
low_dims = 2

batch_size = 1024
epochs = int(1e2)

x = env.sample_state()
print(x)
print(env.inverse_transform(env.transform(x)))

for i in tqdm(range(dat.shape[0])):
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

encoder = Encoder(high_dims,low_dims)
dynamics = LatentDynamics(low_dims)
decoder = Decoder(low_dims,high_dims)

criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(set(list(encoder.parameters()) + list(dynamics.parameters()) + 
list(decoder.parameters())), lr=1e-3)

train_losses = []
val_losses = []

for epoch in tqdm(range(epochs)):
    current_train_loss = 0.0
    epoch_train_loss = 0.0

    encoder.train()
    dynamics.train()
    decoder.train()
    for i, data in enumerate(train_loader, 0):
        x_t, x_tau  = data

        optimizer.zero_grad()

        z_t = encoder(x_t)
        x_t_pred = decoder(z_t)
        z_tau_pred = dynamics(z_t)
        x_tau_pred = decoder(z_tau_pred)
        z_tau = encoder(x_tau)

        loss_ae1 = criterion(x_t_pred, x_t)
        loss_ae2 = criterion(x_tau_pred, x_tau)
        loss_dyn = criterion(z_tau_pred, z_tau)

        loss = loss_ae1 + loss_ae2 + loss_dyn
        loss.backward()

        optimizer.step()

        current_train_loss += loss.item()
        epoch_train_loss += loss.item()
        
        if (i+1) % 100 == 0:
            print("Loss after mini-batch ",i+1,": ", current_train_loss)
            current_train_loss = 0.0
    
    epoch_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    epoch_val_loss = 0.0
    encoder.eval()
    dynamics.eval()
    decoder.eval()
    for i, data in enumerate(val_loader, 0):
        x_t, x_tau  = data

        z_t = encoder(x_t)
        x_t_pred = decoder(z_t)
        z_tau_pred = dynamics(z_t)
        x_tau_pred = decoder(z_tau_pred)
        z_tau = encoder(x_tau)

        loss_ae1 = criterion(x_t_pred, x_t)
        loss_ae2 = criterion(x_tau_pred, x_tau)
        loss_dyn = criterion(z_tau_pred, z_tau)

        epoch_val_loss += loss_ae1.item() + loss_ae2.item() + loss_dyn.item()
    
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
plt.ylim(0,0.1)
plt.show()

with torch.no_grad():
    encoder.eval()
    dynamics.eval()
    decoder.eval()

    # Check the reconstruction for x_t
    num_points = int(1e4)

    plt.figure(figsize=(8,8))
    plt.scatter(raw_dat_te[:num_points,0],raw_dat_te[:num_points,1],c='r',label='Actual')
    z_t_pred = encoder(torch.from_numpy(dat_te[:num_points,:high_dims]).float())
    x_t_pred = decoder(z_t_pred).numpy()
    x_t_pred = x_t_pred*(bounds[:,1] - bounds[:,0]) + bounds[:,0]
    pred_plot = np.zeros((x_t_pred.shape[0],2))
    for i in range(x_t_pred.shape[0]):
        pred_plot[i] = env.inverse_transform(x_t_pred[i,:])
    plt.scatter(pred_plot[:,0],pred_plot[:,1],c='b',label='Predicted')
    # Plot line between actual and predicted
    for i in range(num_points):
        plt.plot([raw_dat_te[i,0],pred_plot[i,0]],[raw_dat_te[i,1],pred_plot[i,1]],c='k')
    plt.legend(loc='best')
    plt.xlim(-np.pi,np.pi)
    plt.ylim(-2*np.pi,2*np.pi)
    plt.xlabel("theta")
    plt.ylabel("thetadot")
    plt.title("State Space")
    plt.show()

    plt.figure(figsize=(8,8))
    plt.scatter(raw_dat_te[:num_points,2],raw_dat_te[:num_points,3],c='r',label='Actual')
    z_t_pred = encoder(torch.from_numpy(dat_te[:num_points,high_dims:]).float())
    x_t_pred = decoder(z_t_pred).numpy()
    x_t_pred = x_t_pred*(bounds[:,1] - bounds[:,0]) + bounds[:,0]
    pred_plot = np.zeros((x_t_pred.shape[0],2))
    for i in range(x_t_pred.shape[0]):
        pred_plot[i] = env.inverse_transform(x_t_pred[i,:])
    plt.scatter(pred_plot[:,0],pred_plot[:,1],c='b',label='Predicted')
    # Plot line between actual and predicted
    for i in range(num_points):
        plt.plot([raw_dat_te[i,2],pred_plot[i,0]],[raw_dat_te[i,3],pred_plot[i,1]],c='k')
    plt.legend(loc='best')
    plt.xlim(-np.pi,np.pi)
    plt.ylim(-2*np.pi,2*np.pi)
    plt.xlabel("theta")
    plt.ylabel("thetadot")
    plt.title("State Space")
    plt.show()

    plt.figure(figsize=(8,8))
    z_t = encoder(torch.from_numpy(dat_te[:num_points,:high_dims]).float())
    z_tau = encoder(torch.from_numpy(dat_te[:num_points,high_dims:]).float())
    z_tau = z_tau.detach().numpy()
    plt.scatter(z_tau[:,0],z_tau[:,1],c='r',label='Actual')
    z_tau_pred = dynamics(z_t).float()
    z_tau_pred = z_tau_pred.detach().numpy()
    plt.scatter(z_tau_pred[:,0],z_tau_pred[:,1],c='b',label='Predicted')
    for i in range(num_points):
        plt.plot([z_tau[i,0],z_tau_pred[i,0]],[z_tau[i,1],z_tau_pred[i,1]],c='k')
    plt.legend(loc='best')
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("Latent Space")
    plt.show()

    plt.figure(figsize=(8,8))
    z_t = encoder(torch.from_numpy(dat_te[:num_points,high_dims:]).float())
    z_tau = encoder(torch.from_numpy(dat_te[:num_points,high_dims:]).float())
    z_tau_pred = dynamics(z_t).float()

    x_tau = decoder(z_tau).numpy()
    x_tau = x_tau*(bounds[:,1] - bounds[:,0]) + bounds[:,0]
    x_tau_pred = decoder(z_tau_pred).numpy()
    x_tau_pred = x_tau_pred*(bounds[:,1] - bounds[:,0]) + bounds[:,0]
    pred_plot_pred = np.zeros((x_tau_pred.shape[0],2))
    for i in range(x_tau_pred.shape[0]):
        pred_plot_pred[i] = env.inverse_transform(x_tau_pred[i,:])
    pred_plot = np.zeros((x_tau.shape[0],2))
    for i in range(x_tau.shape[0]):
        pred_plot[i] = env.inverse_transform(x_tau[i,:])
    plt.scatter(pred_plot[:,0],pred_plot[:,1],c='r',label='Actual')
    plt.scatter(pred_plot_pred[:,0],pred_plot_pred[:,1],c='b',label='Predicted')
    for i in range(num_points):
        plt.plot([pred_plot[i,0],pred_plot_pred[i,0]],[pred_plot[i,1],pred_plot_pred[i,1]],c='k')
    plt.legend(loc='best')
    plt.xlim(-np.pi,np.pi)
    plt.ylim(-2*np.pi,2*np.pi)
    plt.xlabel("theta")
    plt.ylabel("thetadot")
    plt.title("State Space")
    plt.show()
    '''

    plt.figure(figsize=(8,8))
    # plt.xlim(-1,1)
    # plt.ylim(-1,1)
    z_t = encoder(torch.from_numpy(dat_te[:,:high_dims]).float())
    z_t = z_t.detach().numpy()
    plt.scatter(z_t[:,0],z_t[:,1],c='r',label='Initial')
    z_tau = encoder(torch.from_numpy(dat_te[:,high_dims:]).float())
    z_tau = z_tau.detach().numpy()
    plt.scatter(z_tau[:,0],z_tau[:,1],c='b',label='Final')
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("Latent space - z_t and z_tau (true)")
    plt.legend(loc='best')
    plt.show()

    plt.figure(figsize=(8,8))
    # plt.xlim(-1,1)
    # plt.ylim(-1,1)
    z_t = encoder(torch.from_numpy(dat_te[:,:high_dims]).float())
    z_t = z_t.detach().numpy()
    plt.scatter(z_t[:,0],z_t[:,1],c='r',label='Initial')
    z_tau_pred = dynamics(torch.from_numpy(z_t).float())
    z_tau_pred = z_tau_pred.detach().numpy()
    plt.scatter(z_tau_pred[:,0],z_tau_pred[:,1],c='b',label='Final')
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("Latent space - z_t and z_tau (predicted)")
    plt.legend(loc='best')
    plt.show()
    '''