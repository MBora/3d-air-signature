import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import random

# Directory setup
# input_dir = './train_samples/'  # Root directory to process all subdirectories
input_dir = './validation_samples_two_stream/'  # Root directory to process all subdirectories

save_dir = './Output_val_two_stream/'  # Root directory for outputs

def load_and_process_file(file_path):
    data = np.load(file_path)
    
    # Check if the file is "current" and needs expanding dimensions
    if 'current' in file_path:
        data = np.expand_dims(data, 0)
    
    return data

def plot_and_save(data, file_path, epoch):
    if data.ndim < 2 or data.shape[-1] < 3:
        print(f"Skipping {file_path} due to unexpected data dimensions.")
        return  # Skip files that do not meet the expected dimensions

    if 'reconstructed' in file_path:
        # Extract x, y, z coordinates for pen tip and pen tail from reconstructed data
        x_tip, y_tip, z_tip = data[0, :, 0], data[0, :, 1], data[0, :, 2]
        x_tail, y_tail, z_tail = data[1, :, 0], data[1, :, 1], data[1, :, 2]
    else:
        # Extract x, y, z coordinates for pen tip (first three columns)
        if data.ndim == 2:
            x_tip, y_tip, z_tip = data[:, 0], data[:, 1], data[:, 2]
            if data.shape[1] >= 6:
                x_tail, y_tail, z_tail = data[:, 3], data[:, 4], data[:, 5]
            else:
                x_tail, y_tail, z_tail = None, None, None
        else:
            x_tip, y_tip, z_tip = data[0, :, 0], data[0, :, 1], data[0, :, 2]
            if data.shape[2] >= 6:
                x_tail, y_tail, z_tail = data[0, :, 3], data[0, :, 4], data[0, :, 5]
            else:
                x_tail, y_tail, z_tail = None, None, None

    print(f"Plotting {file_path}...")
    # Plotting the 3D trajectory
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot(x_tip, y_tip, z_tip, 'g-', label='Pen Tip')  # Green line for pen tip
    
    if x_tail is not None and y_tail is not None and z_tail is not None:
        ax.plot(x_tail, y_tail, z_tail, 'r-', label='Pen Tail')  # Red line for pen tail
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

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
