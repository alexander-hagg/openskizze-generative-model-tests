import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from models.gan import Generator, Discriminator

import torch
from torch.utils.data import Dataset
import os
import numpy as np

import torch
from torch.utils.data import Dataset
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
            torch.Tensor: A tensor of shape [1, x, z] representing the height map.
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
        height_map = voxel_binary.argmax(axis=2)  # Shape: [x, z]
        # However, if no True in a column, argmax returns 0, but we need to ensure it reflects no occupancy
        # To handle this, set height to 0 where there's no occupancy
        no_occupancy = ~voxel_binary.any(axis=2)
        height_map = height_map.astype(np.float32)
        height_map[no_occupancy] = 0.0  # Explicitly set height to 0 where there's no occupancy
        
        # Convert the height map to a float tensor and add a channel dimension
        height_map_tensor = torch.tensor(height_map, dtype=torch.float32).unsqueeze(0)  # Shape: [1, x, z]
    
        return height_map_tensor



def train(progress_callback=None, output_size=30, latent_dim=100, initial_size=4, base_channels_G=128, base_channels_D=64, final_pool_size=(4,4), dataset_path='data/processed/all_samples.npy', model_path='models/'):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = VoxelDataset(dataset_path)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        print(f"Dataset length: {len(dataloader.dataset)}")  # Should print 1000
        
        # Initialize Generator and Discriminator with dynamic architectures
        generator = Generator(latent_dim=latent_dim, output_size=output_size, initial_size=initial_size, base_channels=base_channels_G).to(device)
        discriminator = Discriminator(input_size=output_size, base_channels=base_channels_D, final_pool_size=final_pool_size).to(device)
        
        
        # If using multiple GPUs, wrap with DataParallel
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            generator = nn.DataParallel(generator)
            discriminator = nn.DataParallel(discriminator)
        
        criterion = nn.BCELoss()
        optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
        optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

        total_epochs = 25
        for epoch in range(total_epochs):
            d_loss_epoch = 0.0
            g_loss_epoch = 0.0
            for i, data in enumerate(dataloader):
                real_data = data.to(device)  # Shape: [batch_size, 1, output_size, output_size]
                batch_size = real_data.size(0)
                # print(f'Batch {i+1}/{len(dataloader)}: batch_size={batch_size}, real_data shape={real_data.shape}')

                # ---------------------
                #  Train Discriminator
                # ---------------------
                discriminator.zero_grad()

                # Real labels: 1
                real_labels = torch.ones(batch_size, 1).to(device)
                outputs_real = discriminator(real_data)
                d_loss_real = criterion(outputs_real, real_labels)

                # Fake labels: 0
                z = torch.randn(batch_size, latent_dim).to(device)
                fake_data = generator(z)
                outputs_fake = discriminator(fake_data.detach())
                fake_labels = torch.zeros(batch_size, 1).to(device)
                d_loss_fake = criterion(outputs_fake, fake_labels)

                # Total Discriminator Loss
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_D.step()

                # -----------------
                #  Train Generator
                # -----------------
                generator.zero_grad()
                # Generator wants the discriminator to believe the fake data is real
                generator_labels = torch.ones(batch_size, 1).to(device)
                outputs_gen = discriminator(fake_data)
                g_loss = criterion(outputs_gen, generator_labels)
                g_loss.backward()
                optimizer_G.step()

                d_loss_epoch += d_loss.item()
                g_loss_epoch += g_loss.item()

            avg_d_loss = d_loss_epoch / len(dataloader)
            avg_g_loss = g_loss_epoch / len(dataloader)
            print(f'Epoch [{epoch+1}/{total_epochs}], d_loss: {avg_d_loss:.4f}, g_loss: {avg_g_loss:.4f}')

            # Update progress
            if progress_callback:
                progress = int(((epoch + 1) / total_epochs) * 100)
                progress_callback(progress)

            # Save the model checkpoints
            if (epoch+1) % 10 == 0:
                os.makedirs('models', exist_ok=True)
                torch.save(generator.state_dict(), f'models/gan_generator_epoch_{epoch+1}.pth')
                torch.save(discriminator.state_dict(), f'models/gan_discriminator_epoch_{epoch+1}.pth')


        # Save final model
        os.makedirs('models', exist_ok=True)
        torch.save(generator.state_dict(), f'models/gan_generator_final.pth')
        torch.save(discriminator.state_dict(), f'models/gan_discriminator_final.pth')

        # Final progress update
        if progress_callback:
            progress_callback(100)
    except Exception as e:
        if progress_callback:
            progress_callback(-1)  # Indicate failure
        raise e

if __name__ == '__main__':
    train()
