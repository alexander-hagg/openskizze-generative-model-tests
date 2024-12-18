import numpy as np
import os
import time

def generate_synthetic_data(progress_callback=None, num_samples=1000, max_width=30, max_height=9, save_dir='data/processed/'):
    """Generates synthetic 2.5D voxel height maps for urban structures.
    
    Each voxel represents a 3x3x3 meter cube.
    Buildings have a maximum height of max_height meters."""
    
    voxel_resolution = 3  # Each voxel is 3x3x3 meters
    max_voxel_width = max_width // voxel_resolution
    max_voxel_height = max_height // voxel_resolution

    data = []
    for i in range(num_samples):
        building = np.zeros((max_voxel_width, max_voxel_width, max_voxel_height))
        num_buildings = np.random.randint(1, 5)
        for j in range(num_buildings):
            x, y = np.random.randint(0, max_voxel_width, size=2)
            width = np.random.randint(1, max_voxel_width - x + 1)
            depth = np.random.randint(1, max_voxel_width - y + 1)
            height = np.random.randint(1, max_voxel_height + 1)
            # Ensure that x + width and y + depth do not exceed max_voxel_width
            x_end = min(x + width, max_voxel_width)
            y_end = min(y + depth, max_voxel_width)
            building[x:x_end, y:y_end, :height] = 1
        data.append(building)
        # Update progress
        if progress_callback:
            progress = int(((i + 1) / num_samples) * 100)
            progress_callback(progress)
    save_data(data, save_dir)
    return np.array(data)

def save_data(data, save_dir):
    """Saves all generated data samples into a single file in the specified directory."""
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'all_samples.npy'), data)

if __name__ == '__main__':
    data = generate_synthetic_data()
    print(f'Synthetic data generated: {data.shape}')
    save_data(data, 'data/processed/')
    print("Dataset generated and saved to 'data/processed/' directory.")