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

# class FullModel(nn.Module):
#     def __init__(encoder, dynamics, dynamics):
#         self.encoder = encoder
#         self.dynamics = encoder
#         self.dynamics = encoder
