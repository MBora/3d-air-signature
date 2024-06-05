import os
import shutil

def combine_folders(base_path, source_folders, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    for source_folder in source_folders:
        full_source_path = os.path.join(base_path, source_folder)
        for root, dirs, files in os.walk(full_source_path):
            # Determine the relative path from the source folder
            relative_path = os.path.relpath(root, full_source_path)
            # Create corresponding directory in the destination folder
            dest_dir = os.path.join(destination_folder, relative_path)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            
            # Copy all files to the destination directory
            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dest_dir, file)
                shutil.copy2(src_file, dst_file)
                print(f"Copied {src_file} to {dst_file}")

# Define your base path, source folders, and destination folder
base_path = '/media/valgrant/Windows-SSD/Users/mahes/Desktop/Projects/JournalMain/AirSigns/AirSignsBothBallsAug'  # Change this to your actual base path
source_folders = ['Train', 'Test', 'Validation']
destination_folder = os.path.join(base_path, 'Combined')

combine_folders(base_path, source_folders, destination_folder)
