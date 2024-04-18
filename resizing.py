# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 21:17:16 2024

@author: Fkhan
"""

import os
import numpy as np

# Directory containing the .npy files
directory = 'C:/Users/Fkhan/OneDrive - Georgia Institute of Technology/Coursework/Spring 2024/CS 6476/Group_project/MU3D_processed/filtered_image_stack/'
save_directory = 'C:/Users/Fkhan/OneDrive - Georgia Institute of Technology/Coursework/Spring 2024/CS 6476/Group_project/MU3D_processed/resized_image_stack/'
# List all .npy files in the directory
npy_files = [file for file in os.listdir(directory) if file.endswith('.npy')]

# Initialize variables to store maximum height and width
max_height = 0
max_width = 0
all_height = []
all_width = []
# Determine maximum height and width among all frames
for npy_file in npy_files:
    # Load the frame from the .npy file
    frame = np.load(os.path.join(directory, npy_file))
    height, width = frame.shape[1:3]
    max_height = max(max_height, height)
    max_width = max(max_width, width)
    all_height = all_height+[height]
    all_width = all_width + [width]
max_width = 880
# Pad frames to make them all the same size
for npy_file in npy_files:
    # Load the frames from the .npy file
    frames = np.load(os.path.join(directory, npy_file))
    num_frames, height, width, channels = frames.shape
    
    # Determine padding sizes
    pad_height_total = max_height - height
    pad_width_total = max_width - width
    
    # Calculate padding on both sides
    pad_height_left = pad_height_total // 2
    pad_height_right = pad_height_total - pad_height_left
    pad_width_left = pad_width_total // 2
    pad_width_right = pad_width_total - pad_width_left
    
    # Add padding to each frame
    padded_frames = np.pad(frames, ((0, 0), (pad_height_left, pad_height_right), (pad_width_left, pad_width_right), (0, 0)), mode='constant', constant_values=0)
    
    np.save(os.path.join(save_directory, npy_file), padded_frames)

print("Padding completed for all frames.")

