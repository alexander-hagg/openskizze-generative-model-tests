import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from models.vae import VoxelVAE

class VoxelDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        voxel = np.load(self.files[idx])
        voxel = torch.tensor(voxel, dtype=torch.float32).unsqueeze(0)
        return voxel

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = VoxelDataset('data/processed/')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = VoxelVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        total_loss = 0
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        print(f'Epoch [{epoch+1}/50], Loss: {total_loss/len(dataloader.dataset):.4f}')

        # Save the model checkpoint
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f'models/vae_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    train()