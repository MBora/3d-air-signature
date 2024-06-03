import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import random

# Directory setup
# input_dir = './train_samples/'  # Root directory to process all subdirectories
# save_dir = './output_train/'  # Root directory for outputs

input_dir = './validation_samples2/'  # Root directory to process all subdirectories
save_dir = './output_val2/'  # Root directory for outputs

def load_and_process_file(file_path):
    data = np.load(file_path)
    if 'current' in file_path:
        data = np.expand_dims(data, 0)
    return data

def blend_data(data1, data2, alpha=0.5):
    blended_data = alpha * data1 + (1 - alpha) * data2
    return blended_data

def plot_and_save(data, file_path, epoch, blend=False):
    if data.ndim < 2 or data.shape[1] < 3:
        print(f"Skipping {file_path} due to unexpected data dimensions.")
        return

    if data.ndim == 2:
        x_, y_, z_ = data[:, 0], data[:, 1], data[:, 2]
    else:
        x_, y_, z_ = data[0, :, 0], data[0, :, 1], data[0, :, 2]

    print(f"Plotting {file_path}...")
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot(x_, y_, z_, 'g-')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    file_type = 'Blended' if blend else ('Reconstructed' if 'reconstructed' in file_path else 'Current')
    ax.set_title(f'{file_type} Sample - Epoch {epoch+1}')

    output_path = file_path.replace(input_dir, save_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_filename = output_path.replace('.npy', '.png')
    fig.savefig(output_filename)
    plt.close(fig)

def main():
    alpha=0.5
    for root, dirs, files in os.walk(input_dir):
        label_files = {}
        for file in files:
            if file.endswith('.npy'):
                base_name = '_'.join(file.split('_')[:-1])  # Get the base name (e.g., sample_2_13)
                suffix = file.split('_')[-1].replace('.npy', '')  # Get the suffix (e.g., current, next)
                if base_name not in label_files:
                    label_files[base_name] = {}
                label_files[base_name][suffix] = os.path.join(root, file)

        for base_name, file_dict in label_files.items():
            if 'current' in file_dict and 'next' in file_dict:
                data_current = load_and_process_file(file_dict['current'])
                data_next = load_and_process_file(file_dict['next'])
                blended_data = blend_data(data_current, data_next, alpha=alpha)
                blend_file_path = os.path.join(root, f'{base_name}_blend.npy')
                np.save(blend_file_path, blended_data)  # Save the blended data
                plot_and_save(blended_data, blend_file_path, 0, blend=True)

    print("Finished processing selected samples.")

if __name__ == "__main__":
    main()
