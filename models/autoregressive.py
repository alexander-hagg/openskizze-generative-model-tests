import torch
import torch.nn as nn

class VoxelRNN(nn.Module):
    def __init__(self, voxel_size=32):
        super(VoxelRNN, self).__init__()
        self.rnn = nn.LSTM(input_size=1, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, 1)
        self.voxel_size = voxel_size

    def forward(self, x):
        x = x.view(x.size(0), -1, 1)  # Flatten the voxel grid
        output, _ = self.rnn(x)
        output = self.fc(output)
        output = torch.sigmoid(output)
        return output.view(-1, 1, self.voxel_size, self.voxel_size, self.voxel_size)