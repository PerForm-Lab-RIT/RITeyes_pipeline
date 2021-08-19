#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:25:45 2021

@author: aayush
"""

import msgpack
import json
import file_methods
import os
from os import walk
import subprocess
import pandas as pd
import argparse
import csv
import pickle
import numpy as np
import math


try:
    default_blender_path_file = open('blender_path.txt', 'r')
    default_blender_path = default_blender_path_file.read()
except Exception:
    print('Not blender_path.txt file!')
    default_blender_path = './blender-2.82a-linux64/blender'

    
try:
    default_data_path_file = open('data_path.txt', 'r')
    default_data_path = default_data_path_file.read()
except Exception:
    print('Not data_path.txt file!')
    default_data_path = 'D:/Raw Eye Data/'


# Read the excel sheet
# data = pd.read_excel('GIWdata.xlsx')

parser = argparse.ArgumentParser() 
parser.add_argument('--condition', type=str,help='Select run type: test, complete, blink',default= 'blink')
parser.add_argument('--folder_name', type=str,help='Set folder name',default= 'test_trial')
parser.add_argument('--path_to_blender', type=str,help='Set blender path',default= default_blender_path)
parser.add_argument('--frame_cap', type=int,help='Number of frames to render for blink-data. -1 means there is no limit',default=-1)
parser.add_argument('--delta', type=int,help='Uses the Nth frame of a 120 fps video. A value of 4 would make it 30 fps.',default=4)
parser.add_argument('--start_frame', type=int,help='The frame to start data collection',default=0)
parser.add_argument('--type', type=str,help='Example imgage generation',default= 'seq')

parser.add_argument('--person_idx', type=str, help='Person index in the GIW Data folder', default=0)
parser.add_argument('--trial_idx', type=str, help='Trial index in the GIW Data folder', default=0)
parser.add_argument('--head_model', type=str, help='Head model used for the data matching', default=0)

parser.add_argument('--cornea', help='Cornea to use in the dimulation', default='0')
parser.add_argument('--iris', help='Iris to use in the dimulation', default='1')
parser.add_argument('--sclera', help='Sclera to use in the dimulation', default='0')
parser.add_argument('--focal_length', help='Focal length to use in the dimulation', default='35')


args = parser.parse_args()
filename=args.folder_name

print(args)

if args.type == 'seq':

    print('Person: '+ str(args.person_idx))
    print('Trial: '+ str(args.trial_idx))

    # camera should never rotate
    camera_elevation =  '0'
    camera_roll =       '0'
    camera_azimuthal =  '0'

    # reads the files we need to build the data pickle file for the blender script
    raw_pupil_data = pd.read_csv(default_data_path +str(args.person_idx)+'/'+str(args.trial_idx)+'/exports/pupil_positions.csv')
    timestamps = np.load(default_data_path +str(args.person_idx)+'/'+str(args.trial_idx)+'/eye0_timestamps.npy')
    # print(raw_pupil_data.keys())
    
    # create the data we need
    print('Gathering Data')
    frame_number = 0

    # data we are saving
    frame = []
    pupil_normal = []
    pupil = []
    eye_loc = []
    time_ms = []

    # how many times we are doing this
    frame_cap = len(timestamps)
    if (args.frame_cap != -1):
        frame_cap = min(frame_cap, args.frame_cap)
    
    # loop through the files an gather data
    i = 0
    skip = 0
    for f in range(frame_cap):
        # skips over the data we don't want
        while True:
            timestamp = timestamps[(f+skip)*args.delta]
            while float(raw_pupil_data['timestamp'][i]) < timestamp or raw_pupil_data['id'][i] != 0:
                i += 1
            if float(raw_pupil_data['confidence'][i]) < 0.8 or math.isnan(float(raw_pupil_data['sphere_center_z'][i])) or float(raw_pupil_data['sphere_center_z'][i]) < 30 or float(raw_pupil_data['sphere_center_z'][i]) > 100:
                skip += 1
            else:
                break

        # frame number
        frame.append((f+skip)*args.delta)
        time_ms.append(float(raw_pupil_data['timestamp'][i])*1000)

        # pupil position
        norm = []
        norm.append(float(raw_pupil_data['circle_3d_normal_x'][i])) # make this negative for binocular left eye
        norm.append(float(raw_pupil_data['circle_3d_normal_z'][i]))
        norm.append(float(raw_pupil_data['circle_3d_normal_y'][i]))
        pupil_normal.append(norm)

        # eye location 
        loc = []
        loc.append(float(raw_pupil_data['sphere_center_x'][i])/10) # make this negative for binocular left eye
        loc.append(float(raw_pupil_data['sphere_center_z'][i])/10)
        loc.append(float(raw_pupil_data['sphere_center_y'][i])/10)
        eye_loc.append(loc)

        # pupil dialation
        pupil.append(float(raw_pupil_data['diameter_3d'][i]))
        print('Frame: '+str(len(frame)) + '   \tTimestamp: '+str(raw_pupil_data['timestamp'][i]))

    # gets the final position of the eyes
    final_eye0_pos = []
    final_eye1_pos = []
    i = len(raw_pupil_data['id'])-1
    while len(final_eye0_pos) == 0 or len(final_eye1_pos) == 0:
        if raw_pupil_data['id'][i] == 0 and len(final_eye0_pos) == 0 and math.isnan(float(raw_pupil_data['sphere_center_z'][i])) == False:
            final_eye0_pos.append(raw_pupil_data['sphere_center_x'][i]/10)
            final_eye0_pos.append(raw_pupil_data['sphere_center_y'][i]/10)
            final_eye0_pos.append(raw_pupil_data['sphere_center_z'][i]/10)
        elif raw_pupil_data['id'][i] == 1 and len(final_eye1_pos) == 0 and math.isnan(float(raw_pupil_data['sphere_center_z'][i])) == False:
            final_eye1_pos.append(raw_pupil_data['sphere_center_x'][i]/10)
            final_eye1_pos.append(raw_pupil_data['sphere_center_y'][i]/10)
            final_eye1_pos.append(raw_pupil_data['sphere_center_z'][i]/10)
        i -= 1


    # gest the calibration data
    filepath = default_data_path +str(args.person_idx)+'/'+str(args.trial_idx)+'/calibrations/'
    f = []
    for (dirpath, dirnames, filenames) in walk(filepath):
        f.extend(filenames)
        break
    calib_filename = f.pop()
    while calib_filename[:3] == 'Def' and len(f) > 0:
        calib_filename = f.pop()
    with open(filepath+calib_filename, "rb") as data_file:
        byte_data = data_file.read()

    calibration_data = msgpack.unpackb(byte_data, use_list=False, strict_map_key=False)
    # print (calibration_data)

    lm_mat = calibration_data['data']['calib_params']['left_model']['eye_camera_to_world_matrix']
    rm_mat = calibration_data['data']['calib_params']['right_model']['eye_camera_to_world_matrix']
    # bm_mat0 = calibration_data['data']['calib_params']['binocular_model']['eye_camera_to_world_matrix0']
    # bm_mat1 = calibration_data['data']['calib_params']['binocular_model']['eye_camera_to_world_matrix1']

    # get eye 1 position in world space
    eye1_world = []
    eye1_world.append(final_eye1_pos[0]*rm_mat[0][0] + final_eye1_pos[1]*rm_mat[1][0] + final_eye1_pos[2]*rm_mat[2][0] + rm_mat[0][3]/100)
    eye1_world.append(-final_eye1_pos[0]*rm_mat[0][1] + -final_eye1_pos[1]*rm_mat[1][1] + -final_eye1_pos[2]*rm_mat[2][1] + rm_mat[1][3]/100)
    eye1_world.append(final_eye1_pos[0]*rm_mat[0][2] + final_eye1_pos[1]*rm_mat[1][2] + final_eye1_pos[2]*rm_mat[2][2] + rm_mat[2][3]/100)

    # subtracts camera 0 position in world space giving us a vector pointing from the left camera to the right eye in scene space
    eye1_world[0] -= lm_mat[0][3]/100
    eye1_world[1] -= lm_mat[1][3]/100
    eye1_world[2] -= lm_mat[2][3]/100

    # apply the left camera's rotation matrix to the vector to get the right eye's location in the left camera space
    eye1_loc = []
    eye1_loc.append(eye1_world[0]*lm_mat[0][0] + eye1_world[1]*lm_mat[0][1] + eye1_world[2]*lm_mat[0][2])
    eye1_loc.append(eye1_world[0]*lm_mat[2][0] + eye1_world[1]*lm_mat[2][1] + eye1_world[2]*lm_mat[2][2])
    eye1_loc.append(eye1_world[0]*lm_mat[1][0] + eye1_world[1]*lm_mat[1][1] + eye1_world[2]*lm_mat[1][2])

    # get the vector from the left eye to the right eye
    eye_delta = [eye1_loc[0]-final_eye0_pos[0], eye1_loc[1]-final_eye0_pos[2], eye1_loc[2]-final_eye0_pos[1]]

    print()
    print('lm_mat: ' + str(lm_mat))
    print('rm_mat: ' + str(rm_mat))
    print()
    print('Eye1 local:      '+str(final_eye1_pos))
    print('Eye1 world:      '+str(eye1_world))
    print('Eye0 local:      '+str(final_eye0_pos))
    print('Other Eye loc:   '+str(eye1_loc))
    print('Delta eye:       '+str(eye_delta))
    print()


    # print ()
    # print('lm_mat location: ' + str(lm_mat[0][3]) + ', ' + str(lm_mat[1][3]) + ', ' + str(lm_mat[2][3]))
    # print('rm_mat location: ' + str(rm_mat[0][3]) + ', ' + str(rm_mat[1][3]) + ', ' + str(rm_mat[2][3]))
    # print('bm_mat0 location: ' + str(bm_mat0[0][3]) + ', ' + str(bm_mat0[1][3]) + ', ' + str(bm_mat0[2][3]))
    # print('bm_mat1 location: ' + str(bm_mat1[0][3]) + ', ' + str(bm_mat1[1][3]) + ', ' + str(bm_mat1[2][3]))
    # print ()

    # print('lm_mat')
    # matrix = lm_mat
    # print('['+ 
    #     '[' + str(matrix[0][0]) + ',' + str(matrix[0][2]) + ',' + str(matrix[0][1]) + ',' + str(matrix[0][3]/100) + '],'
    #     '[' + str(matrix[2][0]) + ',' + str(matrix[2][2]) + ',' + str(matrix[2][1]) + ',' + str(matrix[2][3]/100) + '],'
    #     '[' + str(matrix[1][0]) + ',' + str(matrix[1][2]) + ',' + str(matrix[1][1]) + ',' + str(matrix[1][3]/100) + '],'
    #     '[' + str(matrix[3][0]) + ',' + str(matrix[3][2]) + ',' + str(matrix[3][1]) + ',' + str(matrix[3][3]) + ']' 
    #     +']')
    # print()

    # print('rm_mat')
    # matrix = rm_mat
    # print('['+ 
    #     '[' + str(matrix[0][0]) + ',' + str(matrix[0][1]) + ',' + str(matrix[0][2]) + ',' + str(matrix[0][3]) + '],' 
    #     '[' + str(matrix[1][0]) + ',' + str(matrix[1][1]) + ',' + str(matrix[1][2]) + ',' + str(matrix[1][3]) + '],' 
    #     '[' + str(matrix[2][0]) + ',' + str(matrix[2][1]) + ',' + str(matrix[2][2]) + ',' + str(matrix[2][3]) + '],' 
    #     '[' + str(matrix[3][0]) + ',' + str(matrix[3][1]) + ',' + str(matrix[3][2]) + ',' + str(matrix[3][3]) + ']' 
    #     +']')
    # print()

    # lr_dist = math.sqrt((lm_mat[0][3]-rm_mat[0][3])**2 + (lm_mat[1][3]-rm_mat[1][3])**2 + (lm_mat[2][3]-rm_mat[2][3])**2)
    # bm_dist = math.sqrt((bm_mat0[0][3]-bm_mat1[0][3])**2 + (bm_mat0[1][3]-bm_mat1[1][3])**2 + (bm_mat0[2][3]-bm_mat1[2][3])**2)

    # print('eye distance lr model: ' + str(lr_dist))
    # print('eye distance bm model: ' + str(bm_dist))
    # print('ratio: ' + str(lr_dist/bm_dist))
    # print ()

    eye_delta_norm = eye_delta / np.linalg.norm(eye_delta)

    print('Delta eye norm:       '+str(eye_delta_norm))

    head_elevation =    '-30'
    head_roll =         str(math.degrees(math.asin(eye_delta_norm[2])))
    head_azimuthal =    str(math.degrees(-math.asin(eye_delta_norm[1])))
    print ('head roll: '+head_roll)
    print ('head azimuthal: '+head_azimuthal)


    print('Done gathering data')

    print('Saving to a pickle file')
    pickle_data = {
        'frame'         : frame,
        'time_ms'       : time_ms,
        'pupil_normal'  : pupil_normal,
        'pupil'         : pupil,
        'eye_loc'       : eye_loc,
        'eye_delta'     : eye_delta
    }

    with open(default_data_path+str(args.person_idx)+'/'+str(args.trial_idx)+'/export_data.pickle', 'wb') as handle:
        pickle.dump(pickle_data, handle, protocol=pickle.DEFAULT_PROTOCOL)


    distance = str(eye_loc[0][0]) + ',' + str(eye_loc[0][1]) + ',' + str(eye_loc[0][2])
    print(distance)

    inputfile="./GIW_Data/giw_"+args.condition 
    blend_file=args.head_model+'_v9-pupil.blend'
    
    if True:
        subprocess.call([args.path_to_blender,'-b','static_model/'+args.head_model+'/'+blend_file,
                    '-P','RIT-Eyes_sequential_render.py','--',
                    '--model_id',args.head_model,
                    '--inp',inputfile, 
                    '--data_source','seq',
                    '--person_idx',str(args.person_idx),
                    '--trial_idx',str(args.trial_idx),
                    '--camera_focal_length', args.focal_length,
                    '--light_1_loc',',-1,-1,0',
                    '--light_2_loc',',-1,-0.5,0',
                    '--light_1_energy','100',
                    '--light_2_energy','100',
                    '--camera_elevation',','+camera_elevation, 
                    '--head_elevation',','+head_elevation,
                    '--camera_roll',','+camera_roll, 
                    '--head_roll',','+head_roll,
                    '--camera_azimuthal',','+camera_azimuthal, 
                    '--head_azimuthal',','+head_azimuthal,
                    '--camera_distance',','+distance,
                    '--camera_sensor_size','35',
                    '--pupil', '1,3',
                    '--cornea', args.cornea, 
                    '--iris_textures', args.iris,
                    '--sclera_textures', args.sclera, 
                    '--glass','0',
                    '--no_cornea','1',
                    '--no_reflection','1',
                    '--env','0',
                    '--output_file',filename,
					'--picklefile', default_data_path,
                    '--frame_cap',str(args.frame_cap),
                    '--index','0'])

if args.type == 'example':
    for i in range(1,3):
        head_model = str(i)
        blend_file=str(i)+'_v9-pupil.blend'
        subprocess.call([args.path_to_blender,'-b','static_model/'+head_model+'/'+blend_file,
                    '-P','RIT-Eyes_sequential_render.py','--','--model_id',head_model+'_v9',
                    '--data_source','example','--light_1_loc',',-1,1,0','--light_2_loc',',-1,0.5,0',
                    '--light_1_energy','100','--light_2_energy','100'])

        
        # blend_file=str(i)+'-pupil.blend'
        # subprocess.call([args.path_to_blender,'-b','static_model/'+head_model+'/'+blend_file,
        #             '-P','RIT-Eyes_sequential_render.py','--','--model_id',head_model+'_v1',
        #             '--data_source','example','--light_1_loc',',-1,1,0','--light_2_loc',',-1,0.5,0',
        #             '--light_1_energy','100','--light_2_energy','100'])




# eye_elevation = '-30,30'
# eye_azimuthal = '-30,30'
# cornea = '0,1,2'
# number = 100

# subprocess.call([args.path_to_blender,'-b','static_model/'+head_model+'/'+blend_file,
#                  '-P','RIT-Eyes_sequential_render.py','--','--model_id',head_model,'--inp',
#                  inputfile, '--data_source','random','--data_source_path',trial_name+'.p',
#                  '--camera_focal_length', focal_length,'--light_1_loc',',-1,1,0',
#                  '--light_2_loc',',-1,0.5,0','--light_1_energy','100','--light_2_energy','100',
#                  '--camera_elevation',','+camera_elevation, '--head_elevation',','+head_elevation,
#                  '--camera_roll',','+camera_roll, '--head_roll',','+head_roll,
#                  '--camera_azimuthal',','+camera_azimuthal, '--head_azimuthal',','+head_azimuthal,
#                  '--camera_distance',','+distance,'--camera_sensor_size','35','--pupil', ',1,3',
#                  '--cornea', cornea, '--iris_textures',iris,'--sclera_textures',sclera, 
#                  '--glass','0','--no_cornea','1','--no_reflection','1','--env','0',
#                  '--output_file',filename,'--frame_cap',str(args.frame_cap),
#                  '--eye_elevation',','+eye_elevation,'--eye_azimuthal',','+eye_azimuthal,
#                  '--cornea',','+cornea,'--number',str(number)])
