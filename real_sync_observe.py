'''
Script for creating real-sync video for observe mode

@author: Abhijan Wasti
'''

import numpy as np
import json
import os
import sys
import cv2

from tqdm import tqdm

# Function definitions
def readJsonParameters(json_path, parameter):
    '''
    Read a json for parameters 
    '''
    with open(json_path) as json_file:
        json_str = json_file.read()
        parameter = json.loads(json_str)
    return parameter

def read_blender_lookup():
    filename = os.path.join(os.getcwd(), output_folder, lookup_folder, str(person_idx), str(trial_idx), "blender_lookup.npy")
    lookup = np.load(filename)
    return lookup

# Static variable Initializations

filespath_json_path = "FilePaths.json"

output_folder = "renderings"
observation_output_folder = "observation"
lookup_folder = "lookup"
lookup_filename = "blender_lookup.npy"

person_idx = 3
trial_idx = 1

start_frame= 25000
end_frame= 26200

# Variables for putText
font = cv2.FONT_HERSHEY_SIMPLEX
org = (1920-250, 1080-50)
fontScale = 1
color = 127
thickness = 5

#################
# Script starts #
#################

file_paths = []

file_paths = readJsonParameters(filespath_json_path, file_paths)

data_path = file_paths["DATA_PATH"]

# Open lookup file and get the frame index and world index
lookup_filename = os.path.join(os.getcwd(), output_folder, lookup_folder, str(person_idx), str(trial_idx), lookup_filename)
if not os.path.exists(lookup_filename):
    print('Failed to create real sync video. Blender lookup file does not exist at', lookup_filename, '! Re-run the RIT-Net pipeline to generate the lookup file.')
    sys.exit()

lookup_table = np.load(lookup_filename)
frame_indices = lookup_table[:, 0].astype(int)
world_indices = lookup_table[:, 4].astype(int)

# Read the world video
data_directory = os.path.join(data_path, str(person_idx), str(trial_idx), 'exports')
# data_directory = os.path.join(data_path, str(person_idx), str(trial_idx))
world_filename = os.path.join(data_directory, 'world.mp4')

cap = cv2.VideoCapture(world_filename)

# Read the rendered images and write the composite video out
renders_directory = os.path.join(os.getcwd(), output_folder, observation_output_folder, str(person_idx), str(trial_idx))

ret, world_img = cap.read()
if not ret:
    print('Failed to read world video. World.mp4 file does not exist at', lookup_filename, '! Check to see if the path is set correctly.')
    sys.exit()

frame_height, frame_width = world_img.shape[:2]
print('Setting frame size to ', frame_width, frame_height*2)
output = cv2.VideoWriter('Composite_3_1_w_gaze_mapped_2.mp4', cv2.VideoWriter_fourcc('m','p','4','v'), 120, (frame_width,frame_height*2))
# output = cv2.VideoWriter('Composite_1_1_w_raw_world_2.mp4', cv2.VideoWriter_fourcc('m','p','4','v'), 120, (frame_width,frame_height*2))

for idx in tqdm(range(start_frame, end_frame)):
    render_filename = os.path.join(renders_directory, str(idx).zfill(4) + ".tif")
    
    rendered_img = cv2.imread(render_filename)
    rendered_img = cv2.putText(rendered_img, '120 fps', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    
    # cap.set(cv2.CAP_PROP_POS_FRAMES, world_indices[np.where(frame_indices==idx)[0][0]])
    cap.set(cv2.CAP_PROP_POS_FRAMES, world_indices[np.where(frame_indices==idx)[0][0]] + 6)

    ret, frame = cap.read()
    if ret:
        world_img = cv2.putText(frame, '30 fps', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)

    composite = np.vstack((world_img, rendered_img))
    
    output.write(composite)

cap.release()
output.release()
print('Done!')