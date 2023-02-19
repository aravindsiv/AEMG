import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

from pendulum import PendulumNoCtrl

from torch.utils.data import DataLoader
import wandb
from models import *
import os
from config import *

# ROOT_PATH = 'root_lqr'
MODEL_PATH = f'{ROOT_PATH}/models'
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)\

RESULTS_PATH = f'{ROOT_PATH}/results'
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

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

env = PendulumNoCtrl()
bounds = env.get_transformed_state_bounds()

if mode == "lqr":
    raw_data = np.loadtxt(data_file, delimiter=",")
    print(raw_data.shape)

if mode == "noctrl":
    raw_data = np.loadtxt(data_file, delimiter=",")
    print(raw_data.shape)

raw_dat = np.array(raw_data)
print(raw_dat.shape)

dat = np.zeros((raw_dat.shape[0],8))
dataset_size = raw_dat.shape[0]

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

# encoder = encoder.to('cuda:0')
# dynamics = dynamics.to('cuda:0')
# decoder = decoder.to('cuda:0')

encoder = encoder.to(str("cuda:0") if torch.cuda.is_available() else "cpu")
dynamics = dynamics.to(str("cuda:0") if torch.cuda.is_available() else "cpu")
decoder = decoder.to(str("cuda:0") if torch.cuda.is_available() else "cpu")

criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(set(list(encoder.parameters()) + list(dynamics.parameters()) +
list(decoder.parameters())), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.001, patience=5, verbose=True)

train_losses = []
val_losses = []

# latent_coeff = 1.0

# train = True
# warmup = 20

if train:
    wandb.init('dynamics')

    train_losses = {'loss_ae1': [], "loss_ae2": [], "loss_dyn": [], 'loss_total': []}
    val_losses = {'loss_ae1': [], "loss_ae2": [], "loss_dyn": [], 'loss_total': []}
    for epoch in tqdm(range(epochs)):
        current_train_loss = 0.0
        epoch_train_loss = 0.0
        latent_coeff = 1.0 # epoch/epochs

        loss_ae1_train = 0
        loss_ae2_train = 0
        loss_dyn_train = 0

        encoder.train()
        dynamics.train()
        decoder.train()
        ctr = 0
        for i, data in enumerate(train_loader, 0):
            optimizer.zero_grad()
            x_t, x_tau  = data
            x_t = x_t.to(str("cuda:0") if torch.cuda.is_available() else "cpu")
            x_tau = x_tau.to(str("cuda:0") if torch.cuda.is_available() else "cpu")
            # print(torch.cat([x_t, x_tau], dim=-1).shape)
            z_t = encoder(x_t)
            x_t_pred = decoder(z_t)
            z_tau_pred = dynamics(z_t)
            x_tau_pred = decoder(z_tau_pred)
            z_tau = encoder(x_tau)

            loss_ae1 = criterion(x_t_pred, x_t)
            loss_ae2 = criterion(x_tau_pred, x_tau)
            loss_dyn = criterion(z_tau_pred, z_tau.detach())

            if epoch >= warmup:
                loss = loss_ae1 + loss_ae2 + loss_dyn
            else:
                loss = loss_ae1 + loss_ae2
            loss.backward()
            optimizer.step()

            current_train_loss += loss.item()
            epoch_train_loss += loss.item()

            loss_ae1_train += loss_ae1.item()
            loss_ae2_train += loss_ae2.item()
            loss_dyn_train += loss_dyn.item() if epoch>=warmup else 0.0
            ctr += 1

            if (i+1) % 100 == 0:
                print("Loss after mini-batch ",i+1,": ", current_train_loss)
                current_train_loss = 0.0
        wandb.log({
            "train/loss_ae1": loss_ae1_train/ctr,
            "train/loss_ae2": loss_ae2_train/ctr,
            "train/loss_dyn": loss_dyn_train/ctr,
            "train/total_loss": epoch_train_loss/ctr,
            # 'learning_rate': scheduler.get_lr()
        })

        train_losses['loss_ae1'].append(loss_ae1_train/ctr)
        train_losses['loss_ae2'].append(loss_ae2_train/ctr)
        train_losses['loss_dyn'].append(loss_dyn_train/ctr)
        train_losses['loss_total'].append(epoch_train_loss/ctr)

        # epoch_train_loss = epoch_train_loss / len(train_loader)
        # train_losses.append(epoch_train_loss)

        with torch.no_grad():
            epoch_val_loss = 0.0
            loss_ae1_val = 0.
            loss_ae2_val = 0.
            loss_dyn_val = 0.
            encoder.eval()
            dynamics.eval()
            decoder.eval()
            ctr = 0
            for i, data in enumerate(val_loader, 0):
                x_t, x_tau  = data
                x_t = x_t.to(str("cuda:0") if torch.cuda.is_available() else "cpu")
                x_tau = x_tau.to(str("cuda:0") if torch.cuda.is_available() else "cpu")

                z_t = encoder(x_t)
                x_t_pred = decoder(z_t)
                z_tau_pred = dynamics(z_t)
                x_tau_pred = decoder(z_tau_pred)
                z_tau = encoder(x_tau)

                loss_ae1 = criterion(x_t_pred, x_t)
                loss_ae2 = criterion(x_tau_pred, x_tau)
                loss_dyn = criterion(z_tau_pred, z_tau)


                loss_ae1_val += loss_ae1.item()
                loss_ae2_val += loss_ae2.item()
                loss_dyn_val += loss_dyn.item() if epoch>=warmup else 0.0

                ctr += 1

                if epoch >= warmup:
                    epoch_val_loss += loss_ae1.item() + loss_ae2.item() + loss_dyn.item()
                else:
                    epoch_val_loss += loss_ae1.item() + loss_ae2.item()

            wandb.log({
                "val/loss_ae1": loss_ae1_val/ctr,
                "val/loss_ae2": loss_ae2_val/ctr,
                "val/loss_dyn": loss_dyn_val/ctr,
                "val/total_loss": epoch_val_loss/ctr,
            })
            val_losses['loss_ae1'].append(loss_ae1_val/ctr)
            val_losses['loss_ae2'].append(loss_ae2_val/ctr)
            val_losses['loss_dyn'].append(loss_dyn_val/ctr)
            val_losses['loss_total'].append(epoch_val_loss/ctr)

            if epoch >= warmup:
                scheduler.step(epoch_val_loss/ctr)

        import pickle
        with open(f'{RESULTS_PATH}/losses{"_warmup" if warmup > 0 else ""}_{mode}.pkl', 'wb') as f:
            pickle.dump({'train': train_losses, 'val': val_losses}, f)

        print('train loss:', epoch_train_loss, 'test loss:', epoch_val_loss)

    torch.save(encoder, f"{MODEL_PATH}/encoder_{mode}_0.1_20steps_1M{'_warmup' if warmup > 0 else ''}_64.pt")
    torch.save(decoder, f"{MODEL_PATH}/decoder_{mode}_0.1_20steps_1M{'_warmup' if warmup > 0 else ''}_64.pt")
    torch.save(dynamics, f"{MODEL_PATH}/dynamics_{mode}_0.1_20steps_1M{'_warmup' if warmup > 0 else ''}_64.pt")

encoder = torch.load(f"{MODEL_PATH}/encoder_{mode}_0.1_20steps_1M{'_warmup' if warmup > 0 else ''}_64.pt").to("cpu")
decoder = torch.load(f"{MODEL_PATH}/decoder_{mode}_0.1_20steps_1M{'_warmup' if warmup > 0 else ''}_64.pt").to("cpu")
dynamics = torch.load(f"{MODEL_PATH}/dynamics_{mode}_0.1_20steps_1M{'_warmup' if warmup > 0 else ''}_64.pt").to("cpu")

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
    plt.savefig(f'{RESULTS_PATH}/L1_{mode}.png')
    # plt.show()

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
    plt.savefig(f'{RESULTS_PATH}/L2_{mode}.png')
    # plt.show()

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
    plt.savefig(f'{RESULTS_PATH}/L3_{mode}.png')
    # plt.show()

    num_points = int(1e3)

    plt.figure(figsize=(8,8))
    z_t = encoder(torch.from_numpy(dat_te[:num_points,:high_dims]).float())
    z_tau = encoder(torch.from_numpy(dat_te[:num_points,high_dims:]).float())
    z_tau = z_tau.detach().numpy()
    plt.scatter(z_t[:,0],z_t[:,1],c='r',label='z_t')
    z_tau_pred = dynamics(z_t).float()
    z_tau_pred = z_tau_pred.detach().numpy()
    plt.scatter(z_tau_pred[:,0],z_tau_pred[:,1],c='b',label='z_tau')
    for i in range(num_points):
        plt.plot([z_t[i,0],z_tau_pred[i,0]],[z_t[i,1],z_tau_pred[i,1]],c='k')
    plt.legend(loc='best')
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("Latent Space Dyn")
    plt.savefig(f'{RESULTS_PATH}/latent_dyn_{mode}.png')
    # plt.show()

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
    plt.scatter(pred_plot[:,0],pred_plot[:,1],c='r',label='z_t')
    plt.scatter(pred_plot_pred[:,0],pred_plot_pred[:,1],c='b',label='z_tau')
    for i in range(num_points):
        plt.plot([pred_plot[i,0],pred_plot_pred[i,0]],[pred_plot[i,1],pred_plot_pred[i,1]],c='k')
    plt.legend(loc='best')
    plt.xlim(-np.pi,np.pi)
    plt.ylim(-2*np.pi,2*np.pi)

    plt.xlabel("theta")
    plt.ylabel("thetadot")
    plt.title("State Space")
    # plt.show()

    fig, axes = plt.subplots(1, 2,  figsize=(16,8), sharey=True)
    ax = axes.flatten()

    x = torch.from_numpy(dat_te[:num_points,:high_dims]).float()
    x = x*(bounds[:,1] - bounds[:,0]) + bounds[:,0]
    x_tau = torch.from_numpy(dat_te[:num_points,high_dims:]).float()
    x_tau = x_tau*(bounds[:,1] - bounds[:,0]) + bounds[:,0]
    pred_plot_pred = np.zeros((x_tau.shape[0],2))
    for i in range(x_tau.shape[0]):
        pred_plot_pred[i] = env.inverse_transform(x_tau[i,:])
    pred_plot = np.zeros((x.shape[0],2))
    for i in range(x.shape[0]):
        pred_plot[i] = env.inverse_transform(x[i,:])
    ax[0].scatter(pred_plot[:,0],pred_plot[:,1],c='r',label='x_t')
    ax[0].scatter(pred_plot_pred[:,0],pred_plot_pred[:,1],c='b',label='x_tau')
    for i in range(num_points):
        ax[0].plot([pred_plot[i,0],pred_plot_pred[i,0]],[pred_plot[i,1],pred_plot_pred[i,1]],c='k')

    ax[0].set_xlim(-np.pi,np.pi)
    ax[0].set_ylim(-2*np.pi,2*np.pi)
    ax[0].legend(loc='best')

    z_t = encoder(torch.from_numpy(dat_te[:num_points,:high_dims]).float())
    z_tau = encoder(torch.from_numpy(dat_te[:num_points,high_dims:]).float())
    z_tau_pred = dynamics(z_t).float()

    x_tau = decoder(z_t).numpy()
    x_tau = x_tau*(bounds[:,1] - bounds[:,0]) + bounds[:,0]
    x_tau_pred = decoder(z_tau_pred).numpy()
    x_tau_pred = x_tau_pred*(bounds[:,1] - bounds[:,0]) + bounds[:,0]
    pred_plot_pred = np.zeros((x_tau_pred.shape[0],2))
    for i in range(x_tau_pred.shape[0]):
        pred_plot_pred[i] = env.inverse_transform(x_tau_pred[i,:])
    pred_plot = np.zeros((x_tau.shape[0],2))
    for i in range(x_tau.shape[0]):
        pred_plot[i] = env.inverse_transform(x_tau[i,:])
    ax[1].scatter(pred_plot[:,0],pred_plot[:,1],c='r',label='x_t')
    ax[1].scatter(pred_plot_pred[:,0],pred_plot_pred[:,1],c='b',label='x_tau')
    for i in range(num_points):
        ax[1].plot([pred_plot[i,0],pred_plot_pred[i,0]],[pred_plot[i,1],pred_plot_pred[i,1]],c='k')

    ax[1].set_xlim(-np.pi,np.pi)
    ax[1].set_ylim(-2*np.pi,2*np.pi)
    ax[1].legend(loc='best')
    plt.savefig(f'{RESULTS_PATH}/pred_dyn_{mode}.png')
