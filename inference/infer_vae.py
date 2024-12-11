import torch
import numpy as np
from models.vae import VoxelVAE

def generate_voxel(model_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VoxelVAE().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        z = torch.randn(1, 200).to(device)
        generated_voxel, _, _ = model.decode(z)
    voxel = generated_voxel.cpu().numpy().squeeze()
    np.save(output_path, voxel)
    print(f"Generated voxel saved to {output_path}")

if __name__ == '__main__':
    generate_voxel('models/vae_epoch_50.pth', 'outputs/generated_voxel_vae.npy')