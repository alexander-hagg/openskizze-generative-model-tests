import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from models.gan import Generator, Discriminator

class VoxelDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        voxel = np.load(self.files[idx])
        voxel = torch.tensor(voxel, dtype=torch.float32).unsqueeze(0)
        return voxel

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = VoxelDataset('data/processed/')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

    for epoch in range(50):
        for i, data in enumerate(dataloader):
            real_data = data.to(device)
            batch_size = real_data.size(0)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            discriminator.zero_grad()
            outputs = discriminator(real_data)
            d_loss_real = criterion(outputs, real_labels)
            z = torch.randn(batch_size, 100).to(device)
            fake_data = generator(z)
            outputs = discriminator(fake_data.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            generator.zero_grad()
            outputs = discriminator(fake_data)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizer_G.step()

        print(f'Epoch [{epoch+1}/50], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

        # Save the model checkpoints
        if (epoch+1) % 10 == 0:
            torch.save(generator.state_dict(), f'models/gan_generator_epoch_{epoch+1}.pth')
            torch.save(discriminator.state_dict(), f'models/gan_discriminator_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    train()