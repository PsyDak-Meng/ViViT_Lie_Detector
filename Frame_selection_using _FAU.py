
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 17:12:50 2024

@author: Fkhan
"""

import os
import dlib
import cv2
import json
import csv
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained facial landmark detector from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define Facial Action Units based on landmark indices
AU_definitions = {
    'AU1' : [(17,21),(18, 22), (19, 23), (20, 24)],  # Inner Brow Raiser
    'AU2' : [(1, 5), (2, 4)],  # Outer Brow Raiser
    'AU4' : [(1, 2), (4, 5)],  # Drow Lowerer
    #'AU5' :  [(37, 38, 39, 40, 41), (43, 44, 45, 46, 47)],  # Upper Lid Raiser
    'AU6' : [(49, 54), (53, 52), (52, 55)],  # Cheek Raiser
    'AU7' :  [(38, 40), (44, 46)], # Lid Tightener
    'AU9' : [(33, 36), (34, 35)],  # Nose Wrinkler
    #'AU10' : [(49, 50, 51), (53, 54, 55)],  # Upper lip Raiser
    'AU12' : [(48, 54), (51, 57), (57, 55)], # Lip corner puller
    'AU15' :  [(48, 58), (54, 58), (51, 58), (57, 58)],  # Lip Corner Depressor
    'AU17' : [(7, 9)],  # Chin Raiser
    'AU25' : [(50, 52), (58, 56), (51, 53), (57, 55)]  # Lips part
}
# Add more AU definitions as needed

# Functions
def smooth_sequence(seq, window_len):
    filtered_sequence = scipy.signal.savgol_filter(seq, window_length=window_len, polyorder=3)
    return filtered_sequence

def peak_cwt(seq):
    filtered_sequence = smooth_sequence(seq, 60)
    cwt_peaks = scipy.signal.find_peaks_cwt(filtered_sequence, widths=np.arange(5, 15))
    return cwt_peaks, filtered_sequence

def peak_def(seq):
    # Apply Savitzky-Golay filter for smoothing (optional)
    filtered_sequence = smooth_sequence(seq, 30)
    # Differentiate the smoothed signal to find peaks
    differentiated_y = np.gradient(filtered_sequence)
    # Find peaks in the differentiated signal
    peaks, _ = scipy.signal.find_peaks(differentiated_y, prominence=0.2)
    return peaks, filtered_sequence

def remove_duplicates_within_range(nums, threshold):
    # Sort the list to ensure ascending order
    #nums.sort()

    # Initialize a list to store the filtered numbers
    filtered_nums = []

    # Initialize a variable to store the previously seen number
    prev_num = None

    # Iterate through the list of numbers
    for num in nums:
        # If prev_num is None or num is not within +-3 range of prev_num, add it to filtered_nums
        if prev_num is None or abs(num - prev_num) > threshold:
            filtered_nums.append(num)
            prev_num = num  # Update prev_num to the current number

    return filtered_nums

# video file directory
directory_path = f'C:/Users\Fkhan\OneDrive - Georgia Institute of Technology\Coursework\Spring 2024\CS 6476\Group_project\MU3D_dataset\Videos\Videos/'
# get all the video file names in the directory
video_file_names = os.listdir(directory_path)

#iterate through all the files 
for file_name in video_file_names[1:2]:
    #video_file_name = "BF015_1PT.wmv"
    video_path = f'{directory_path}/{file_name}'
    cap = cv2.VideoCapture(video_path)
    
    # Initialize an empty list to store the facial landmarks sequence
    landmarks_sequence = []
    
    # Initialize an empty list to store Facial Action Units sequence
    AU_sequence = {au: [] for au in AU_definitions}
    frames = []
    # Loop through frames in the video
    while cap.isOpened():
        ret, frame = cap.read()
    
        if not ret:
            break
    
        #append each frame to a list
        frames.append(frame.copy())
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Detect faces in the frame
        faces = detector(gray)
    
        
        # Loop over detected faces
        for face in faces:
            # Predict facial landmarks
            landmarks = predictor(gray, face)
    
            # Convert landmarks to a list of (x, y) coordinates
            landmarks_list = [(p.x, p.y) for p in landmarks.parts()]
    
            # Append the landmarks to the sequence
            landmarks_sequence.append(landmarks_list)
            
            # Draw landmarks on the frame
            for (x, y) in landmarks_list:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
            # Extract Facial Action Units based on landmark movements
    
            for au, landmark_pairs in AU_definitions.items():
                AU_frame = [
                    np.linalg.norm(np.array(landmarks_list[j]) - np.array(landmarks_list[i]))
                    for (i, j) in landmark_pairs
                ]
    
                # Append the AU_frame to the sequence
                AU_sequence[au].append(AU_frame)
    
    """
        # Display the frame
        cv2.imshow("Facial Action Units", frame)
    
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    """
    # Release the video capture object
    cap.release()
    
    # Convert the list of frames to a NumPy array stack
    stacked_frames = np.stack(frames)
    
    cv2.destroyAllWindows()
    """
    # Visualize the sequence of facial landmarks (optional)
    for landmarks_frame in landmarks_sequence:
        for (x, y) in landmarks_frame:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
    """        
    # Initialize an empty list to store FAU peaks sequence
    FAU_peaks = {au: [] for au in AU_definitions}
    # Inititalize a multiplot figure that shows the FAU sequence and the selected frames
    fig, axes = plt.subplots(nrows=len(AU_sequence), ncols=1, figsize=(10, 5 * len(AU_sequence)))
    
    #iterate through each AU
    for i, (au, sequence) in enumerate(AU_sequence.items()):
        sequence = np.array(sequence)
        combined_peaks = []
        
        # Iterate through each points of the AU 
        for j in range (np.shape(sequence)[1]):
            seq = [au_seq[j] for au_seq in sequence]
            peaks, filtered_sequence = peak_cwt(seq)
            combined_peaks = combined_peaks + [peaks]
        #flatten the peaks
        flattened_peaks = [item for sublist in combined_peaks for item in sublist]
        
        #get unique peaks
        unique_peaks = sorted(set(flattened_peaks))
        
        #Remove frames that are closer to each other. Threshold set to 5
        filtered_peaks = remove_duplicates_within_range(unique_peaks,5)
        
        #convert peak values to int datatype
        filtered_peaks = [int(x) for x in filtered_peaks]
        # Append the AU_frame to the sequence
        FAU_peaks[au].append(filtered_peaks)
        
        #plot AU sequences and peak frame location (red vertical line)
        axes[i].plot(sequence)
        #axes[i].plot(filtered_sequence)
        axes[i].set_title(f'{au} Sequence')
        axes[i].set_xlabel('Frame')
        axes[i].set_ylabel('AU Intensity')
        for x in peaks:
            axes[i].axvline(x, color='red', linestyle='--', linewidth=1)

    #plot significant frames for specific AU    
    Significant_frames = FAU_peaks['AU1'][0]
    for i in Significant_frames:
        frame_num = i
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2RGB))  # Accessing the ith frame (index starts from 0)
        plt.title(f' {frame_num}th Frame from the Video')
        plt.axis('off')
        plt.show()

    #Merge peaks based on different AUs
    flattened_all_peaks = [item for sublist in FAU_peaks for lists in FAU_peaks[sublist] for item in lists]
    unique_all_peaks = remove_duplicates_within_range(sorted(set(flattened_all_peaks)),5)
    save_path = f'C:/Users\Fkhan\OneDrive - Georgia Institute of Technology\Coursework\Spring 2024\CS 6476\Group_project\MU3D_processed/{file_name}'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Iterate through all the peaks, plot corresponding frames.
    for i in unique_all_peaks:
        frame_num = i
        img_height, img_width, _ = frames[frame_num].shape
        aspect_ratio = img_width / img_height
        fig_width = 10
        fig_height = fig_width / aspect_ratio
        plt.figure(figsize=(fig_width, fig_height))
        plt.imshow(cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2RGB))  # Accessing the ith frame (index starts from 0)
        #plt.title(f' {frame_num}th Frame from the Video')
        plt.axis('off')
        
        plt.savefig (f'{save_path}/{i}.png')
        plt.close('all')
    
    #Save peaks for each AU as a dictionary
    # Specify the file path
    dict_path = f'{save_path}/peaks_dictionary.json'
    
    # Save the dictionary to a JSON file
    with open(dict_path, 'w') as json_file:
        json.dump(FAU_peaks, json_file)
    
    # Save the list of significant frames
    # Specify the file path
    list_path = f'{save_path}/unique_peaks.json'
    # Save the list to a CSV file
    with open(list_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows([unique_all_peaks])
