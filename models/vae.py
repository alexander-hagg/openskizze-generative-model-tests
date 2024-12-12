import torch
import torch.nn as nn

class VoxelVAE(nn.Module):
    def __init__(self, latent_dim=200):
        super(VoxelVAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv3d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv3d(64, 128, 4, 2, 1),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128 * 4 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4 * 4, latent_dim)
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 128 * 4 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, 128, 4, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar