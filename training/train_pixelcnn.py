import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from models.pixelcnn import PixelCNN

class VoxelSequenceDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        voxel = np.load(self.files[idx]).flatten()
        voxel = torch.tensor(voxel, dtype=torch.float32).unsqueeze(-1)
        return voxel[:-1], voxel[1:]  # Input sequence and target sequence

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = VoxelSequenceDataset('data/processed/')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = VoxelRNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        total_loss = 0
        for input_seq, target_seq in dataloader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            optimizer.zero_grad()
            output = model(input_seq)
            loss = criterion(output.view(-1), target_seq.view(-1))
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        print(f'Epoch [{epoch+1}/20], Loss: {total_loss/len(dataloader):.4f}')

        # Save the model checkpoint
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f'models/autoregressive_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    train()