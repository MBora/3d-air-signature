import sys
import cv2
from signal import signal, SIGINT
import numpy as np
import time
import os, os.path
from glob import iglob
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from numpy import genfromtxt

saveDir = './output/'
frameWidth = 672
frameHeight = 376
pngdim = 256
density = 3
margin = 10
f_cam = 100
b_cam = 0.12

def main():
    num_epochs = 1  # Replace with the actual number of epochs

    for epoch in range(num_epochs):
        sample_filename = f"reconstructed_sample_Tip_Gaussian_-0.5stddev.npy"
        reconstructed_sample = np.load(sample_filename)
        # sample_filename = f"./input_sample_Tip_Gaussian_-0.5stddev.npy"
        # reconstructed_sample = np.load(sample_filename)
        # reconstructed_sample = np.expand_dims(reconstructed_sample, 0)

        # Extract the x, y, and z coordinates from the reconstructed sample
        x_ = reconstructed_sample[0, :, 0]
        y_ = reconstructed_sample[0, :, 1]
        z_ = reconstructed_sample[0, :, 2]  # Assuming z-coordinate is stored in the third element

        # Graph:
        fig2 = plt.figure(2)
        ax3d = fig2.add_subplot(111, projection='3d')
        l = len(x_)

        # Check if the number of data points is greater than the degree of the spline
        if l > 3:
            tck, u = interpolate.splprep([x_, y_, z_], s=0, k=3)  # Use k=3 instead of k=4
            u_fine = np.linspace(0, 1, 3*l)
            x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
            ax3d.plot(x_fine, y_fine, z_fine, 'g')
        else:
            ax3d.plot(x_, y_, z_, 'g')

        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Z')
        ax3d.set_title(f'Reconstructed Sample - Epoch {epoch+1}')

        fig2.show()
        plt.show()

    print("Finished processing reconstructed samples.")

if __name__ == "__main__":
    main()