import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import random

# Directory setup
# input_dir = './train_samples/'  # Root directory to process all subdirectories
input_dir = './validation_samples5/'  # Root directory to process all subdirectories

save_dir = './Output_val5/'  # Root directory for outputs

def load_and_process_file(file_path):
    data = np.load(file_path)
    
    # Check if the file is "current" and needs expanding dimensions
    if 'current' in file_path:
        data = np.expand_dims(data, 0)
    
    return data

def plot_and_save(data, file_path, epoch):
    if data.ndim < 2 or data.shape[1] < 3:
        print(f"Skipping {file_path} due to unexpected data dimensions.")
        return  # Skip files that do not meet the expected dimensions

    # Extract x, y, z coordinates assuming they are the first three 
    if data.ndim == 2:
        x_, y_, z_ = data[:, 0], data[:, 1], data[:, 2]
    else:
        x_, y_, z_ = data[0, :, 0], data[0, :, 1], data[0, :, 2]

    print(f"Plotting {file_path}...")
    # Plotting the 3D trajectory
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot(x_, y_, z_, 'g-')  # Green line
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Determine file type for title (current or reconstructed)
    file_type = 'Reconstructed' if 'reconstructed' in file_path else 'Current'
    ax.set_title(f'{file_type} Sample - Epoch {epoch+1}')

    # Define the output path by replacing input_dir with save_dir in the file's full path
    output_path = file_path.replace(input_dir, save_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure the directory exists

    # Save the plot
    output_filename = output_path.replace('.npy', '.png')
    fig.savefig(output_filename)
    plt.close(fig)

def main():
    num_epochs = 1
    triplets = []

    for root, dirs, files in os.walk(input_dir):
        grouped_files = {}
        for file in files:
            if file.endswith('.npy'):
                base_name = file.rsplit('_', 2)[0]  # Split off the suffix like _current, _next
                if base_name not in grouped_files:
                    grouped_files[base_name] = []
                grouped_files[base_name].append(os.path.join(root, file))

        # Select 5 random groups of triplets
        selected_triplets = random.sample(list(grouped_files.values()), min(10, len(grouped_files)))
        triplets.extend(selected_triplets)

    for epoch in range(num_epochs):
        for group in triplets:
            for file_path in group:
                data = load_and_process_file(file_path)
                plot_and_save(data, file_path, epoch)

    print("Finished processing selected samples.")

if __name__ == "__main__":
    main()
