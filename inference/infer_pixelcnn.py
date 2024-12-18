import torch
import numpy as np
from models.pixelcnn import PixelCNN

def infer(model_path, output_path, voxel_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PixelCNN(voxel_size=voxel_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        input_seq = torch.zeros(1, voxel_size ** 3, 1).to(device)
        generated_voxel = model(input_seq)
    voxel = generated_voxel.cpu().numpy().squeeze()
    np.save(output_path, voxel)
    print(f"Generated voxel saved to {output_path}")

if __name__ == '__main__':
    infer('models/pixelcnn_epoch_20.pth', 'outputs/generated_voxel_pixelcnn.npy')