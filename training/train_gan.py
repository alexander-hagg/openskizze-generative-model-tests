import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from models.gan import Generator, Discriminator  # Ensure Discriminator does not have a sigmoid at the end

import torch
from torch.utils.data import Dataset
import os
import numpy as np

import torch
from torch.utils.data import Dataset
import os
import numpy as np

class VoxelDataset(Dataset):
    def __init__(self, data_file):
        """
        Initializes the dataset by loading a single .npy voxel file containing multiple samples.

        Args:
            data_file (str): Path to the .npy file containing voxel data.
        """
        # Load the entire .npy file into memory
        try:
            self.data = np.load(data_file)
            if self.data.ndim != 4:
                raise ValueError(f"Expected voxel data to have 4 dimensions [num_samples, x, y, z], but got shape {self.data.shape}")
            print(f"Loaded voxel data from {data_file} with shape {self.data.shape}")
        except Exception as e:
            raise IOError(f"Failed to load data from {data_file}: {e}")
    
    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        """
        Retrieves the height map for the voxel data at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            torch.Tensor: A tensor of shape [1, x, z] representing the normalized height map.
        """
        # Retrieve the voxel sample; shape: [x, y, z]
        voxel = self.data[idx]
    
        # Check voxel dimensions
        if voxel.ndim != 3:
            raise ValueError(f"Expected voxel data to be 3D, but got shape {voxel.shape} for sample index {idx}")
    
        # Convert voxel data to binary if it's not already
        if not np.issubdtype(voxel.dtype, np.bool_) and voxel.dtype != np.bool:
            voxel_binary = voxel > 0  # Assuming non-zero indicates occupancy
        else:
            voxel_binary = voxel
    
        # Extract the height map by finding the highest occupied voxel along the z-axis
        # For each (x, y), find the maximum z where voxel[x, y, z] is True
        # If no voxel is occupied in a column, set height to 0
        # Using `np.argmax` on reversed z-axis to find the first occurrence from the top
        # If no occupancy, `np.argmax` returns 0, which is correct since height should be 0
        height_map = voxel_binary.argmax(axis=2)  # Shape: [x, y]
        # However, if no True in a column, argmax returns 0, but we need to ensure it reflects no occupancy
        # To handle this, set height to 0 where there's no occupancy
        no_occupancy = ~voxel_binary.any(axis=2)
        height_map = height_map.astype(np.float32)
        height_map[no_occupancy] = 0.0  # Explicitly set height to 0 where there's no occupancy
        
        # Normalize the height map to [-1, 1]
        max_height = np.max(height_map) if np.max(height_map) > 0 else 1.0
        height_map_normalized = (height_map / max_height) * 2 - 1  # Scale to [-1, 1]
    
        # Convert the height map to a float tensor and add a channel dimension
        height_map_tensor = torch.tensor(height_map_normalized, dtype=torch.float32).unsqueeze(0)  # Shape: [1, x, z]
    
        return height_map_tensor


def gradient_penalty(critic, real_samples, fake_samples, device, lambda_gp=10):
    """
    Computes the gradient penalty for WGAN-GP.

    Args:
        critic (nn.Module): The critic network.
        real_samples (torch.Tensor): Real data samples.
        fake_samples (torch.Tensor): Generated data samples.
        device (torch.device): The device to perform computations on.
        lambda_gp (float): Weight for the gradient penalty term.

    Returns:
        torch.Tensor: The gradient penalty.
    """
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    # For height maps, adjust the dimensions accordingly
    alpha = alpha.expand_as(real_samples)

    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    interpolates.requires_grad_(True)

    critic_interpolates = critic(interpolates)
    # For WGAN, the output is a scalar per sample
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    return penalty

def train(progress_callback=None, output_size=30, latent_dim=10, initial_size=4, base_channels_G=128, base_channels_D=64, final_pool_size=(4,4), dataset_path='data/processed/all_samples.npy', model_path='models/'):
    """
    Trains the WGAN-GP model on the provided voxel dataset.

    Args:
        progress_callback (function, optional): Function to call with progress updates.
        output_size (int): Spatial size of the output height maps.
        latent_dim (int): Dimension of the latent vector.
        initial_size (int): Starting spatial size after the first linear layer.
        base_channels_G (int): Base number of channels for the Generator.
        base_channels_D (int): Base number of channels for the Critic.
        final_pool_size (tuple): Final pooling size before classification in the Critic.
        dataset_path (str): Path to the voxel dataset (.npy file).
        model_path (str): Directory to save the trained models.
    """
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = VoxelDataset(dataset_path)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
        print(f"Dataset length: {len(dataloader.dataset)}")  # Should print the number of samples
        
        # Initialize Generator and Critic (Discriminator) with dynamic architectures
        generator = Generator(latent_dim=latent_dim, output_size=output_size, initial_size=initial_size, base_channels=base_channels_G).to(device)
        critic = Discriminator(input_size=output_size, base_channels=base_channels_D, final_pool_size=final_pool_size).to(device)

        def weights_init_normal(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

        # Apply the weights_init_normal to the generator and critic
        generator.apply(weights_init_normal)
        critic.apply(weights_init_normal)
        
        # If using multiple GPUs, wrap with DataParallel
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            generator = nn.DataParallel(generator)
            critic = nn.DataParallel(critic)
        
        # Optimizers
        # WGAN-GP recommends using Adam with betas=(0.0, 0.9)
        optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
        optimizer_C = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))

        scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=50, gamma=0.5)
        scheduler_C = optim.lr_scheduler.StepLR(optimizer_C, step_size=50, gamma=0.5)
        
        # Training parameters
        total_epochs = 10  # Increase epochs for better learning
        n_critic = 5        # Number of critic updates per generator update
        lambda_gp = 10      # Gradient penalty coefficient

        for epoch in range(total_epochs):
            generator.train()
            critic.train()
            d_loss_epoch = 0.0
            g_loss_epoch = 0.0

            for i, data in enumerate(dataloader):
                real_samples = data.to(device)  # Shape: [batch_size, 1, x, z]
                batch_size = real_samples.size(0)
                
                # ---------------------
                #  Train Critic
                # ---------------------
                for _ in range(n_critic):
                    optimizer_C.zero_grad()
                    
                    # Sample noise as generator input
                    z = torch.randn(batch_size, latent_dim, device=device)
                    
                    # Generate fake samples
                    fake_samples = generator(z)
                    
                    # Critic outputs
                    critic_real = critic(real_samples)
                    critic_fake = critic(fake_samples.detach())
                    
                    # Wasserstein loss
                    loss_C = -(torch.mean(critic_real) - torch.mean(critic_fake))
                    
                    # Gradient penalty
                    gp = gradient_penalty(critic, real_samples, fake_samples.detach(), device, lambda_gp)
                    
                    # Total loss
                    total_loss_C = loss_C + gp
                    total_loss_C.backward()
                    optimizer_C.step()
                    
                    d_loss_epoch += total_loss_C.item()
                
                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()
                
                # Generate fake samples
                z = torch.randn(batch_size, latent_dim, device=device)
                fake_samples = generator(z)
                
                # Critic evaluates the fake samples
                critic_fake = critic(fake_samples)
                
                # Generator loss: minimize -E[critic(fake)]
                loss_G = -torch.mean(critic_fake)
                loss_G.backward()
                optimizer_G.step()
                
                g_loss_epoch += loss_G.item()
            
            avg_d_loss = d_loss_epoch / (len(dataloader) * n_critic)
            avg_g_loss = g_loss_epoch / len(dataloader)
            print(f'Epoch [{epoch+1}/{total_epochs}], Critic Loss: {avg_d_loss:.4f}, Generator Loss: {avg_g_loss:.4f}')

            scheduler_G.step()
            scheduler_C.step()

            # Update progress
            if progress_callback:
                progress = int(((epoch + 1) / total_epochs) * 100)
                progress_callback(progress)

            # Save the model checkpoints
            if (epoch+1) % 10 == 0:
                os.makedirs(model_path, exist_ok=True)
                torch.save(generator.state_dict(), os.path.join(model_path, f'gan_generator_epoch_{epoch+1}.pth'))
                torch.save(critic.state_dict(), os.path.join(model_path, f'gan_critic_epoch_{epoch+1}.pth'))

        # Save final model
        os.makedirs(model_path, exist_ok=True)
        torch.save(generator.state_dict(), os.path.join(model_path, 'gan_generator_final.pth'))
        torch.save(critic.state_dict(), os.path.join(model_path, 'gan_critic_final.pth'))

        # Final progress update
        if progress_callback:
            progress_callback(100)
    except Exception as e:
        if progress_callback:
            progress_callback(-1)  # Indicate failure
        raise e

if __name__ == '__main__':
    train()