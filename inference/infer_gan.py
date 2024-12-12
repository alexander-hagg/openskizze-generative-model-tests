import torch
import numpy as np
from models.gan import Generator

def infer(generator_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()
    with torch.no_grad():
        z = torch.randn(1, 100).to(device)
        generated_voxel = generator(z)
    voxel = generated_voxel.cpu().numpy().squeeze()
    np.save(output_path, voxel)
    print(f"Generated voxel saved to {output_path}")

if __name__ == '__main__':
    infer('models/gan_generator_epoch_50.pth', 'outputs/generated_voxel_gan.npy')