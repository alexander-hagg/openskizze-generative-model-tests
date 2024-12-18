import torch
import numpy as np
from models.gan import Generator
import sys

def infer(progress_callback=None, model_path=None, output_path=None, num_samples=10, latent_dim=10):
    # Debug: Check the type of generator_path
    generator_path = model_path + 'gan_generator_final.pth'
    if not isinstance(generator_path, str):
        print("Error: generator_path must be a string representing the file path.")
        sys.exit(1)
    
    if not callable(torch.load):
        print("Error: torch.load has been overridden and is not callable.")
        sys.exit(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator(latent_dim=latent_dim, output_size=30, initial_size=4, base_channels=128).to(device)
    
    try:
        # Load the state_dict with weights_only=True to suppress the FutureWarning
        # Ensure you're using a PyTorch version that supports weights_only
        # If not, you might need to update PyTorch or ignore the warning for now
        generator_state = torch.load(generator_path, map_location=device, weights_only=True)
    except TypeError:
        # If weights_only is not a valid argument (older PyTorch version), omit it
        print("Warning: 'weights_only' argument is not supported in your PyTorch version. Proceeding without it.")
        generator_state = torch.load(generator_path, map_location=device)
    except Exception as e:
        print(f"Error loading generator state: {e}")
        sys.exit(1)
    
    try:
        generator.load_state_dict(generator_state)
    except Exception as e:
        print(f"Error loading state_dict into Generator: {e}")
        sys.exit(1)
    
    generator.eval()
    
    with torch.no_grad():
        z = torch.randn(num_samples, 10).to(device)
        generated_voxel = generator(z)
        print(generated_voxel.shape)
    
    voxel = generated_voxel.cpu().numpy().squeeze()
    
    # Debug: Check the shape of the generated voxel
    print(f"Generated voxel shape: {voxel.shape}")
    
    try:
        print(output_path)
        np.save(output_path, voxel)
        print(f"Generated voxel saved to {output_path}")
        # Final progress update
        if progress_callback:
            progress_callback(100)        
    except Exception as e:
        print(f"Error saving generated voxel: {e}")
        sys.exit(1)

if __name__ == '__main__':
    # Ensure the correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python infer_gan.py <generator_path> <output_path>")
        sys.exit(1)
    
    generator_path = sys.argv[1]
    output_path = sys.argv[2]
    infer(generator_path, output_path)
