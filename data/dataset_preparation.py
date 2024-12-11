import numpy as np
import os

def generate_synthetic_data(num_samples=1000, voxel_size=32):
    """Generates synthetic 2.5D voxel height maps for urban structures."""
    data = []
    for _ in range(num_samples):
        building = np.zeros((voxel_size, voxel_size, voxel_size))
        num_buildings = np.random.randint(1, 5)
        for _ in range(num_buildings):
            x, y = np.random.randint(0, voxel_size, size=2)
            width, depth = np.random.randint(4, voxel_size // 4, size=2)
            height = np.random.randint(1, voxel_size // 2)
            building[x:x+width, y:y+depth, :height] = 1
        data.append(building)
    return np.array(data)

def save_data(data, save_dir):
    """Saves the generated data to the specified directory."""
    os.makedirs(save_dir, exist_ok=True)
    for i, sample in enumerate(data):
        np.save(os.path.join(save_dir, f'sample_{i}.npy'), sample)

if __name__ == '__main__':
    data = generate_synthetic_data()
    save_data(data, 'processed/')
    print("Dataset generated and saved to 'processed/' directory.")