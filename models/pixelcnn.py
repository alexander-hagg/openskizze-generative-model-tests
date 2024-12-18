import torch
import torch.nn as nn

class PixelCNN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, kernel_size=7):
        super(PixelCNN, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, input_dim, kernel_size, padding=kernel_size//2)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(x)