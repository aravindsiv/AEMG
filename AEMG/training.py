import torch
import os
import pickle
import numpy as np
from torch import nn
from tqdm import tqdm
from AEMG.models import *

class TrainingConfig:
    def __init__(self, weights_str):
        self.weights_str = weights_str
        self.parse_config()
    
    def parse_config(self):
        ids = self.weights_str.split('_')
        self.weights = []
        for _, id in enumerate(ids):
            self.weights.append([float(e) for e in id.split('x')[:-1]])
            if len(self.weights[-1]) != 4:
                print("Expected 4 values per training config, got ", len(self.weights[-1]))
                raise ValueError
    
    def __getitem__(self, key):
        return self.weights[key]
    
    def __len__(self):
        return len(self.weights)

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

        self.dynamics_train_loader = loaders['train_dynamics']
        self.dynamics_test_loader = loaders['test_dynamics']
        self.labels_loader = loaders['labels']

        self.reset_losses()

        self.criterion = nn.MSELoss(reduction='mean')
        self.lr = config["learning_rate"]

        self.model_dir = config["model_dir"]
        self.log_dir = config["log_dir"]

        self.contrastive_loss_margin = config["contrastive_loss_margin"]

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
        self.train_losses = {'loss_ae1': [], 'loss_ae2': [], 'loss_dyn': [], 'loss_contrastive': [], 'loss_total': []}
        self.test_losses = {'loss_ae1': [], 'loss_ae2': [], 'loss_dyn': [], 'loss_contrastive': [], 'loss_total': []}
    
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

    def dynamics_losses(self, forward_pass, weight):
        x_t, x_tau, x_t_pred, z_tau, z_tau_pred, x_tau_pred_dyn = forward_pass

        loss_ae1 = self.criterion(x_t, x_t_pred)
        loss_ae2 = self.criterion(x_tau, x_tau_pred_dyn)
        loss_dyn = self.criterion(z_tau_pred, z_tau)
        loss_total = loss_ae1 * weight[0] + loss_ae2 * weight[1] + loss_dyn * weight[2]
        return loss_ae1, loss_ae2, loss_dyn, loss_total

    def labels_losses(self, encodings, labels, weight):
        # Gathering the indices of the success and failure labels
        success_indices = torch.where(labels == 1)[0]
        failure_indices = torch.where(labels == 0)[0]

        # Creating all possible pairs of positive pairs which have the same label and negative pairs which have different labels
        positive_pairs = torch.cat([
            torch.stack(torch.meshgrid(success_indices, success_indices, indexing='ij'), dim=-1).reshape(-1, 2),
            torch.stack(torch.meshgrid(failure_indices, failure_indices, indexing='ij'), dim=-1).reshape(-1, 2),
        ])
        negative_pairs = torch.stack(torch.meshgrid(success_indices, failure_indices, indexing='ij'), dim=-1).reshape(-1, 2)

        # Calculating the pairwise cosine distance between all the encodings
        pairwise_cosine_distance = 1 - nn.functional.cosine_similarity(encodings[None, :, :], encodings[:, None, :], dim=-1)

        # Calculating the mean positive and negative cosine distance
        positive_distance = torch.mean(pairwise_cosine_distance[positive_pairs[:, 0], positive_pairs[:, 1]])
        negative_distance = torch.mean(pairwise_cosine_distance[negative_pairs[:, 0], negative_pairs[:, 1]])

        # Replicating Triplet Loss' formula
        return torch.mean(torch.clamp(positive_distance - negative_distance + self.contrastive_loss_margin, min=0))

    def train(self, epochs=1000, patience=50, weight=[1,1,1,0]):
        '''
        Function that trains all the models with all the losses and weight.
        It will stop if the test loss does not improve for "patience" epochs.
        '''
        weight_bool = [bool(i) for i in weight]
        list_parameters = (weight_bool[0] or weight_bool[1] or weight_bool[2]) * (list(self.encoder.parameters()) + list(self.decoder.parameters()))
        list_parameters += (weight_bool[1] or weight_bool[2]) * list(self.dynamics.parameters())
        optimizer = torch.optim.Adam(list_parameters, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.001, patience=patience, verbose=True)
        for epoch in tqdm(range(epochs)):
            loss_ae1_train = 0
            loss_ae2_train = 0
            loss_dyn_train = 0
            loss_contrastive = 0

            epoch_train_loss = 0
            epoch_test_loss  = 0


            if weight_bool[0] or weight_bool[1]: 
                self.encoder.train() 
                self.decoder.train() 
            if weight_bool[2]: 
                self.dynamics.train()

            for i, (x_t, x_tau) in enumerate(self.dynamics_train_loader):
                optimizer.zero_grad()

                # Forward pass
                forward_pass = self.forward(x_t, x_tau)
                # Compute losses
                loss_ae1, loss_ae2, loss_dyn, loss_total = self.dynamics_losses(forward_pass, weight)
                # Backward pass
                loss_total.backward()
                optimizer.step()

                loss_ae1_train += loss_ae1.item() * weight[0]
                loss_ae2_train += loss_ae2.item() * weight[1]
                loss_dyn_train += loss_dyn.item() * weight[2]
                epoch_train_loss += loss_total.item()

            if weight[3] != 0:
                for i, (x_final, label) in enumerate(self.labels_loader):
                    x_final = x_final.to(self.device)
                    label = label.to(self.device)
                    z_final = self.encoder(x_final)
                    loss_con = self.labels_losses(z_final, label, weight)
                    loss_contrastive += loss_con.item() * weight[3]
                    loss_con.backward()
                    optimizer.step()

            self.train_losses['loss_ae1'].append(loss_ae1_train / len(self.dynamics_train_loader))
            self.train_losses['loss_ae2'].append(loss_ae2_train / len(self.dynamics_train_loader))
            self.train_losses['loss_dyn'].append(loss_dyn_train / len(self.dynamics_train_loader))
            self.train_losses['loss_contrastive'].append(loss_contrastive / len(self.labels_loader))
            self.train_losses['loss_total'].append(epoch_train_loss / len(self.dynamics_train_loader))

            with torch.no_grad():
                loss_ae1_test = 0
                loss_ae2_test = 0
                loss_dyn_test = 0

                if weight_bool[0] or weight_bool[1]:  
                    self.encoder.eval() 
                    self.decoder.eval() 
                if weight_bool[2]: 
                    self.dynamics.eval()

                for i, (x_t, x_tau) in enumerate(self.dynamics_test_loader):
                    # Forward pass
                    forward_pass = self.forward(x_t, x_tau)
                    # Compute losses
                    loss_ae1, loss_ae2, loss_dyn, loss_total = self.dynamics_losses(forward_pass, weight)

                    loss_ae1_test += loss_ae1.item() * weight[0]
                    loss_ae2_test += loss_ae2.item() * weight[1]
                    loss_dyn_test += loss_dyn.item() * weight[2]
                    epoch_test_loss += loss_total.item()

                self.test_losses['loss_ae1'].append(loss_ae1_test / len(self.dynamics_test_loader))
                self.test_losses['loss_ae2'].append(loss_ae2_test / len(self.dynamics_test_loader))
                self.test_losses['loss_dyn'].append(loss_dyn_test / len(self.dynamics_test_loader))
                self.test_losses['loss_contrastive'].append(loss_contrastive / len(self.labels_loader))
                self.test_losses['loss_total'].append(epoch_test_loss / len(self.dynamics_test_loader))

            scheduler.step(epoch_test_loss / len(self.dynamics_test_loader))
            
            if epoch >= patience:
                if np.mean(self.test_losses['loss_total'][-patience:]) > np.mean(self.test_losses['loss_total'][-patience-1:-1]):
                    if self.verbose:
                        print("Early stopping")
                    break
            
            if self.verbose:
                print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch + 1, epochs, epoch_train_loss / len(self.dynamics_train_loader), epoch_test_loss / len(self.dynamics_test_loader)))