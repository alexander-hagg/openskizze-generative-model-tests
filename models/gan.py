import torch
import torch.nn as nn
import math

def calculate_upsampling_steps(initial_size, target_size):
    """
    Calculates the number of upsampling steps required to reach or exceed the target size.
    
    Args:
        initial_size (int): The starting spatial size (e.g., 4 for 4x4).
        target_size (int): The desired spatial size (e.g., 30 for 30x30).
    
    Returns:
        int: Number of upsampling steps needed.
    """
    steps = 0
    size = initial_size
    while size < target_size:
        size = size * 2
        steps += 1
    return steps

class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_size=30, initial_size=4, base_channels=128):
        """
        Initializes the Generator.
        
        Args:
            latent_dim (int): Dimension of the latent vector.
            output_size (int): Desired spatial size of the output height map (e.g., 30 for 30x30).
            initial_size (int): Starting spatial size after the first linear layer (default: 4).
            base_channels (int): Number of feature maps in the first ConvTranspose2d layer.
        """
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.initial_size = initial_size
        self.base_channels = base_channels
        
        # Calculate the number of upsampling steps needed
        self.num_upsamples = calculate_upsampling_steps(self.initial_size, self.output_size)
        
        # Project and reshape the latent vector
        self.project = nn.Sequential(
            nn.Linear(latent_dim, base_channels * initial_size * initial_size),
            nn.BatchNorm1d(base_channels * initial_size * initial_size),
            nn.ReLU(True)
        )
        
        # Build the upsampling blocks dynamically
        self.upsampling_blocks = nn.ModuleList()
        current_channels = base_channels
        current_size = initial_size
        
        for step in range(self.num_upsamples):
            next_size = current_size * 2
            # If doubling overshoots the target size, adjust the scale factor
            if next_size > self.output_size:
                scale_factor = self.output_size / current_size
                current_size = self.output_size
            else:
                scale_factor = 2
                current_size = next_size
            
            # Reduce the number of channels as we upsample
            next_channels = max(current_channels // 2, 64)
            
            self.upsampling_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                    nn.Conv2d(current_channels, next_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(next_channels),
                    nn.ReLU(True)
                )
            )
            current_channels = next_channels
        
        # Final convolution to get desired output channels
        self.final_layer = nn.Sequential(
            nn.Conv2d(current_channels, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Ensures output is in [-1, 1]
        )
        
    def forward(self, z):
        """
        Forward pass of the Generator.
        
        Args:
            z (torch.Tensor): Latent vector of shape [batch_size, latent_dim].
        
        Returns:
            torch.Tensor: Generated height maps of shape [batch_size, 1, output_size, output_size].
        """
        batch_size = z.size(0)
        x = self.project(z)  # Shape: [batch_size, base_channels * initial_size * initial_size]
        x = x.view(batch_size, self.base_channels, self.initial_size, self.initial_size)  # [batch_size, base_channels, initial_size, initial_size]
        
        for block in self.upsampling_blocks:
            x = block(x)
        
        x = self.final_layer(x)
        return x

def calculate_downsampling_steps(initial_size, target_size):
    """
    Calculates the number of downsampling steps required to reach or below the target size.
    
    Args:
        initial_size (int): The starting spatial size (e.g., 30 for 30x30).
        target_size (int): The desired minimum spatial size (e.g., 4 for 4x4).
    
    Returns:
        int: Number of downsampling steps needed.
    """
    steps = 0
    size = initial_size
    while size > target_size:
        size = size // 2
        steps += 1
    return steps

class Discriminator(nn.Module):
    def __init__(self, input_size=30, base_channels=64, final_pool_size=(4, 4)):
        """
        Initializes the Discriminator (Critic) with a dynamic architecture based on input size.
        
        Args:
            input_size (int): Spatial size of the input height map (e.g., 30 for 30x30).
            base_channels (int): Number of feature maps in the first Conv2d layer.
            final_pool_size (tuple): Size to which the feature map is adaptively pooled before the linear layer.
        """
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.base_channels = base_channels
        self.final_pool_size = final_pool_size  # e.g., (4,4)
        
        # Calculate the number of downsampling steps needed
        self.num_downsamples = calculate_downsampling_steps(initial_size=input_size, target_size=final_pool_size[0])
        
        # Start with in_channels=1 (since height maps are single-channel)
        current_channels = 1
        
        # Build the downsampling blocks dynamically
        self.downsampling_blocks = nn.ModuleList()
        current_size = input_size
        for step in range(self.num_downsamples):
            # Determine if the next downsampling step will overshoot the final pool size
            next_size = current_size // 2
            if next_size < self.final_pool_size[0]:
                kernel_size = current_size - self.final_pool_size[0]
                stride = 1
                padding = 0
            else:
                kernel_size = 4
                stride = 2
                padding = 1
            
            out_channels = self.base_channels * (2 ** step)
            
            self.downsampling_blocks.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.InstanceNorm2d(out_channels, affine=True),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            
            current_channels = out_channels
            current_size = next_size
        
        # Adaptive pooling to reach a fixed size before the linear layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d(self.final_pool_size)
        
        # Final classification layer
        self.classifier = nn.Linear(current_channels * self.final_pool_size[0] * self.final_pool_size[1], 1)
        
    def forward(self, x):
        """
        Forward pass of the Discriminator (Critic).
        
        Args:
            x (torch.Tensor): Input height maps of shape [batch_size, 1, input_size, input_size].
        
        Returns:
            torch.Tensor: Scalar scores of shape [batch_size, 1].
        """
        for block in self.downsampling_blocks:
            x = block(x)
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)  # Raw scores, no sigmoid
        return x
