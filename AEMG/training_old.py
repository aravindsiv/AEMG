import torch
import os
import pickle
import numpy as np
from torch import nn
from tqdm import tqdm
from AEMG.models import *

class Training:
    def __init__(self, config, loaders, verbose):
        self.encoder = Encoder(config)
        self.dynamics = LatentDynamics(config)
        self.decoder = Decoder(config)

        self.verbose = bool(verbose)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.device)

        self.encoder.to(self.device)
        self.dynamics.to(self.device)
        self.decoder.to(self.device)

        self.train_loader = loaders['train']
        self.test_loader = loaders['test']

        self.reset_losses()

        self.criterion = nn.MSELoss(reduction='mean')
        self.lr = config["learning_rate"]

        self.model_dir = config["model_dir"]
        self.log_dir = config["log_dir"]

    def save_models(self):
        torch.save(self.encoder, os.path.join(self.model_dir, 'encoder.pt'))
        torch.save(self.dynamics, os.path.join(self.model_dir, 'dynamics.pt'))
        torch.save(self.decoder, os.path.join(self.model_dir, 'decoder.pt'))
    
    def save_logs(self, suffix):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        with open(os.path.join(self.log_dir, 'train_losses_' + suffix + '.pkl'), 'wb') as f:
            pickle.dump(self.train_losses, f)
        
        with open(os.path.join(self.log_dir, 'test_losses_' + suffix + '.pkl'), 'wb') as f:
            pickle.dump(self.test_losses, f)
    
    def reset_losses(self):
        self.train_losses = {'loss_ae1': [], 'loss_ae2': [], 'loss_dyn': [], 'loss_total': []}
        self.test_losses = {'loss_ae1': [], 'loss_ae2': [], 'loss_dyn': [], 'loss_total': []}
    
    def train_encoder_decoder(self, epochs=1000, patience=50, loss='ae1'):
        '''
        Function that trains only the encoder and decoder models.
        It will stop if the test loss does not improve for "patience" epochs.
        '''
        assert loss in ['ae1', 'ae2', 'both'], "Loss must be either 'ae1' or 'ae2' or 'both'"
        optimizer = torch.optim.Adam(set(list(self.encoder.parameters()) + list(self.decoder.parameters())), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.001, patience=10, verbose=True)
        for epoch in tqdm(range(epochs)):
            loss_ae1_train = 0
            loss_ae2_train = 0

            epoch_train_loss = 0
            epoch_test_loss  = 0

            self.encoder.train()

            for i, (x_t, x_tau) in enumerate(self.train_loader):
                optimizer.zero_grad()

                # Forward pass
                x_t = x_t.to(self.device)
                x_tau = x_tau.to(self.device)

                z_t = self.encoder(x_t)
                x_t_pred = self.decoder(z_t)

                z_tau = self.encoder(x_tau)
                x_tau_pred = self.decoder(z_tau)

                if loss == 'ae2':
                    z_tau_pred = self.dynamics(z_t)
                    x_tau_pred_dyn = self.decoder(z_tau_pred)
                    loss_ae2 = self.criterion(x_tau, x_tau_pred_dyn)
                    loss_ae1 = torch.zeros(1).to(self.device)
                
                elif loss == 'ae1':
                    loss_ae1 = self.criterion(x_t, x_t_pred) + self.criterion(x_tau, x_tau_pred)
                    loss_ae2 = torch.zeros(1).to(self.device)
                
                elif loss == 'both':
                    z_tau_pred = self.dynamics(z_t)
                    x_tau_pred_dyn = self.decoder(z_tau_pred)
                    loss_ae2 = self.criterion(x_tau, x_tau_pred_dyn)
                    loss_ae1 = self.criterion(x_t, x_t_pred) + self.criterion(x_tau, x_tau_pred)
                
                loss_total = loss_ae1 + loss_ae2

                # Backward pass
                loss_total.backward()
                optimizer.step()

                loss_ae1_train += loss_ae1.item()
                loss_ae2_train += loss_ae2.item()
                epoch_train_loss += loss_total.item()

            self.train_losses['loss_ae1'].append(loss_ae1_train / len(self.train_loader))
            self.train_losses['loss_ae2'].append(loss_ae2_train / len(self.train_loader))
            self.train_losses['loss_total'].append(epoch_train_loss / len(self.train_loader))

            with torch.no_grad():
                loss_ae1_test = 0
                loss_ae2_test = 0

                self.encoder.eval()

                for i, (x_t, x_tau) in enumerate(self.test_loader):
                    x_t = x_t.to(self.device)
                    x_tau = x_tau.to(self.device)

                    z_t = self.encoder(x_t)
                    x_t_pred = self.decoder(z_t)

                    z_tau = self.encoder(x_tau)
                    x_tau_pred = self.decoder(z_tau)

                    if loss == 'ae2':
                        z_tau_pred = self.dynamics(z_t)
                        x_tau_pred_dyn = self.decoder(z_tau_pred)
                        loss_ae2 = self.criterion(x_tau, x_tau_pred_dyn)
                        loss_ae1 = torch.zeros(1).to(self.device)
                    
                    elif loss == 'ae1':
                        loss_ae1 = self.criterion(x_t, x_t_pred) + self.criterion(x_tau, x_tau_pred)
                        loss_ae2 = torch.zeros(1).to(self.device)

                    elif loss == 'both':
                        z_tau_pred = self.dynamics(z_t)
                        x_tau_pred_dyn = self.decoder(z_tau_pred)
                        loss_ae2 = self.criterion(x_tau, x_tau_pred_dyn)
                        loss_ae1 = self.criterion(x_t, x_t_pred) + self.criterion(x_tau, x_tau_pred)

                    loss_total = loss_ae1 + loss_ae2

                    loss_ae1_test += loss_ae1.item()
                    loss_ae2_test += loss_ae2.item()
                    epoch_test_loss += loss_total.item()

                self.test_losses['loss_ae1'].append(loss_ae1_test / len(self.test_loader))
                self.test_losses['loss_ae2'].append(loss_ae2_test / len(self.test_loader))
                self.test_losses['loss_total'].append(epoch_test_loss / len(self.test_loader))

            scheduler.step(loss_ae1_test / len(self.test_loader))

            if epoch >= patience:
                if np.mean(self.test_losses['loss_total'][-patience:]) > np.mean(self.test_losses['loss_total'][-patience-1:-1]):
                    print("Early stopping")
                    break
            
            print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch + 1, epochs, epoch_train_loss / len(self.train_loader), epoch_test_loss / len(self.test_loader)))

    def train_dynamics(self, epochs=1000, patience=50, use_l2=False):
        '''
        Function that trains only the dynamics model with the "L3" loss.
        It will stop if the test loss does not improve for "patience" epochs.
        '''
        optimizer = torch.optim.Adam(self.dynamics.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.001, patience=10, verbose=True)
        for epoch in tqdm(range(epochs)):
            loss_dyn_train = 0
            loss_ae2_train = 0

            epoch_train_loss = 0
            epoch_test_loss  = 0

            self.dynamics.train()

            for i, (x_t, x_tau) in enumerate(self.train_loader):
                optimizer.zero_grad()

                # Forward pass
                x_t = x_t.to(self.device)
                x_tau = x_tau.to(self.device)

                z_t = self.encoder(x_t)
                z_tau = self.encoder(x_tau)

                z_tau_pred = self.dynamics(z_t)
                x_tau_pred_dyn = self.decoder(z_tau_pred)

                # Compute losses
                loss_dyn = self.criterion(z_tau_pred, z_tau)
                loss_total = loss_dyn

                if use_l2:
                    loss_ae2 = self.criterion(x_tau, x_tau_pred_dyn)
                    loss_total += loss_ae2
                else:
                    loss_ae2 = torch.zeros(1).to(self.device)

                # Backward pass
                loss_total.backward()
                optimizer.step()

                loss_dyn_train += loss_dyn.item()
                loss_ae2_train += loss_ae2.item()
                epoch_train_loss += loss_total.item()

            self.train_losses['loss_dyn'].append(loss_dyn_train / len(self.train_loader))
            self.train_losses['loss_ae2'].append(loss_ae2_train / len(self.train_loader))
            self.train_losses['loss_total'].append(epoch_train_loss / len(self.train_loader))

            with torch.no_grad():
                loss_dyn_test = 0
                loss_ae2_test = 0

                self.dynamics.eval()

                for i, (x_t, x_tau) in enumerate(self.test_loader):
                    x_t = x_t.to(self.device)
                    x_tau = x_tau.to(self.device)

                    z_t = self.encoder(x_t)
                    z_tau = self.encoder(x_tau)

                    z_tau_pred = self.dynamics(z_t)
                    x_tau_pred_dyn = self.decoder(z_tau_pred)

                    loss_dyn = self.criterion(z_tau_pred, z_tau)
                    loss_total = loss_dyn

                    if use_l2:
                        loss_ae2 = self.criterion(x_tau, x_tau_pred_dyn)
                        loss_total += loss_ae2
                    else:
                        loss_ae2 = torch.zeros(1).to(self.device)

                    loss_dyn_test += loss_dyn.item()
                    loss_ae2_test += loss_ae2.item()
                    epoch_test_loss += loss_total.item()

                self.test_losses['loss_dyn'].append(loss_dyn_test / len(self.test_loader))
                self.test_losses['loss_ae2'].append(loss_ae2_test / len(self.test_loader))
                self.test_losses['loss_total'].append(epoch_test_loss / len(self.test_loader))

            scheduler.step(loss_dyn_test / len(self.test_loader))
            
            if epoch >= patience:
                if np.mean(self.test_losses['loss_total'][-patience:]) > np.mean(self.test_losses['loss_total'][-patience-1:-1]):
                    print("Early stopping")
                    break
            
            print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch + 1, epochs, epoch_train_loss / len(self.train_loader), epoch_test_loss / len(self.test_loader)))
    
    def train_all(self, epochs=1000, patience=50):
        '''
        Function that trains all the models with all the losses.
        It will stop if the test loss does not improve for "patience" epochs.
        '''
        optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.dynamics.parameters()), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.001, patience=10, verbose=True)
        for epoch in tqdm(range(epochs)):
            loss_ae1_train = 0
            loss_ae2_train = 0
            loss_dyn_train = 0

            epoch_train_loss = 0
            epoch_test_loss  = 0

            self.encoder.train()
            self.decoder.train()
            self.dynamics.train()

            for i, (x_t, x_tau) in enumerate(self.train_loader):
                optimizer.zero_grad()

                # Forward pass
                x_t = x_t.to(self.device)
                x_tau = x_tau.to(self.device)

                z_t = self.encoder(x_t)
                x_t_pred = self.decoder(z_t)

                z_tau = self.encoder(x_tau)
                x_tau_pred = self.decoder(z_tau)

                z_tau_pred = self.dynamics(z_t)

                # Compute losses
                loss_ae1 = self.criterion(x_t, x_t_pred)
                loss_ae2 = self.criterion(x_tau, x_tau_pred)
                loss_dyn = self.criterion(z_tau_pred, z_tau)

                loss_total = loss_ae1 + loss_ae2 + loss_dyn

                # Backward pass
                loss_total.backward()
                optimizer.step()

                loss_ae1_train += loss_ae1.item()
                loss_ae2_train += loss_ae2.item()
                loss_dyn_train += loss_dyn.item()
                epoch_train_loss += loss_total.item()

            self.train_losses['loss_ae1'].append(loss_ae1_train / len(self.train_loader))
            self.train_losses['loss_ae2'].append(loss_ae2_train / len(self.train_loader))
            self.train_losses['loss_dyn'].append(loss_dyn_train / len(self.train_loader))
            self.train_losses['loss_total'].append(epoch_train_loss / len(self.train_loader))

            with torch.no_grad():
                loss_ae1_test = 0
                loss_ae2_test = 0
                loss_dyn_test = 0

                self.encoder.eval()
                self.decoder.eval()
                self.dynamics.eval()

                for i, (x_t, x_tau) in enumerate(self.test_loader):
                    x_t = x_t.to(self.device)
                    x_tau = x_tau.to(self.device)

                    z_t = self.encoder(x_t)
                    x_t_pred = self.decoder(z_t)

                    z_tau = self.encoder(x_tau)
                    x_tau_pred = self.decoder(z_tau)

                    z_tau_pred = self.dynamics(z_t)

                    loss_ae1 = self.criterion(x_t, x_t_pred)
                    loss_ae2 = self.criterion(x_tau, x_tau_pred)
                    loss_dyn = self.criterion(z_tau_pred, z_tau)

                    loss_total = loss_ae1 + loss_ae2 + loss_dyn

                    loss_ae1_test += loss_ae1.item()
                    loss_ae2_test += loss_ae2.item()
                    loss_dyn_test += loss_dyn.item()
                    epoch_test_loss += loss_total.item()

                self.test_losses['loss_ae1'].append(loss_ae1_test / len(self.test_loader))
                self.test_losses['loss_ae2'].append(loss_ae2_test / len(self.test_loader))
                self.test_losses['loss_dyn'].append(loss_dyn_test / len(self.test_loader))
                self.test_losses['loss_total'].append(epoch_test_loss / len(self.test_loader))

            scheduler.step(epoch_test_loss / len(self.test_loader))
            
            if epoch >= patience:
                if np.mean(self.test_losses['loss_total'][-patience:]) > np.mean(self.test_losses['loss_total'][-patience-1:-1]):
                    print("Early stopping")
                    break
            
            print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch + 1, epochs, epoch_train_loss / len(self.train_loader), epoch_test_loss / len(self.test_loader)))
    
    def forward(self, x_t, x_tau):
        x_t = x_t.to(self.device)
        x_tau = x_tau.to(self.device)

        z_t = self.encoder(x_t)
        x_t_pred = self.decoder(z_t)

        z_tau = self.encoder(x_tau)
        x_tau_pred = self.decoder(z_tau)

        z_tau_pred = self.dynamics(z_t)
        x_tau_pred_dyn = self.decoder(z_tau_pred)

        return (x_t, x_tau, x_t_pred, z_tau, z_tau_pred, x_tau_pred_dyn)

    def losses(self, foward_pass, weight):
        x_t, x_tau, x_t_pred, z_tau, z_tau_pred, x_tau_pred_dyn = foward_pass

        loss_ae1 = self.criterion(x_t, x_t_pred)
        loss_ae2 = self.criterion(x_tau, x_tau_pred_dyn)
        loss_dyn = self.criterion(z_tau_pred, z_tau)
        loss_total = loss_ae1 * weight[0] + loss_ae2 * weight[1] + loss_dyn * weight[2]
        return loss_ae1, loss_ae2, loss_dyn, loss_total

    def train(self, epochs=1000, patience=50, weight=[1,1,1]):
        '''
        Function that trains all the models with all the losses and weight.
        It will stop if the test loss does not improve for "patience" epochs.
        '''
        weight_bool = [bool(i) for i in weight]
        list_parameters = (weight_bool[0] or weight_bool[1]) * (list(self.encoder.parameters()) + list(self.decoder.parameters()))
        list_parameters += weight_bool[2] * list(self.dynamics.parameters())
        optimizer = torch.optim.Adam(list_parameters, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.001, patience=patience, verbose=True)
        for epoch in tqdm(range(epochs)):
            loss_ae1_train = 0
            loss_ae2_train = 0
            loss_dyn_train = 0

            epoch_train_loss = 0
            epoch_test_loss  = 0


            if weight_bool[0] or weight_bool[1]: 
                self.encoder.train() 
                self.decoder.train() 
            if weight_bool[2]: 
                self.dynamics.train()

            for i, (x_t, x_tau) in enumerate(self.train_loader):
                optimizer.zero_grad()

                # Forward pass
                forward_pass = self.forward(x_t, x_tau)
                # Compute losses
                loss_ae1, loss_ae2, loss_dyn, loss_total = self.losses(forward_pass, weight)
                # Backward pass
                loss_total.backward()
                optimizer.step()

                loss_ae1_train += loss_ae1.item() * weight[0]
                loss_ae2_train += loss_ae2.item() * weight[1]
                loss_dyn_train += loss_dyn.item() * weight[2]
                epoch_train_loss += loss_total.item()

            self.train_losses['loss_ae1'].append(loss_ae1_train / len(self.train_loader))
            self.train_losses['loss_ae2'].append(loss_ae2_train / len(self.train_loader))
            self.train_losses['loss_dyn'].append(loss_dyn_train / len(self.train_loader))
            self.train_losses['loss_total'].append(epoch_train_loss / len(self.train_loader))

            with torch.no_grad():
                loss_ae1_test = 0
                loss_ae2_test = 0
                loss_dyn_test = 0

                    
                    self.encoder.eval() 
                    self.decoder.eval() 
                if weight_bool[2]: 
                    self.dynamics.eval()

                for i, (x_t, x_tau) in enumerate(self.test_loader):
                    # Forward pass
                    forward_pass = self.forward(x_t, x_tau)
                    # Compute losses
                    loss_ae1, loss_ae2, loss_dyn, loss_total = self.losses(forward_pass, weight)

                    loss_ae1_test += loss_ae1.item() * weight[0]
                    loss_ae2_test += loss_ae2.item() * weight[1]
                    loss_dyn_test += loss_dyn.item() * weight[2]
                    epoch_test_loss += loss_total.item()

                self.test_losses['loss_ae1'].append(loss_ae1_test / len(self.test_loader))
                self.test_losses['loss_ae2'].append(loss_ae2_test / len(self.test_loader))
                self.test_losses['loss_dyn'].append(loss_dyn_test / len(self.test_loader))
                self.test_losses['loss_total'].append(epoch_test_loss / len(self.test_loader))

            scheduler.step(epoch_test_loss / len(self.test_loader))
            
            if epoch >= patience:
                if np.mean(self.test_losses['loss_total'][-patience:]) > np.mean(self.test_losses['loss_total'][-patience-1:-1]):
                    if self.verbose:
                        print("Early stopping")
                    break
            
            if self.verbose:
                print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch + 1, epochs, epoch_train_loss / len(self.train_loader), epoch_test_loss / len(self.test_loader)))