#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:25:45 2021

@author: aayush
"""

import os
import subprocess
import pandas as pd
import argparse

# Read the excel sheet
data = pd.read_excel('GIWdata.xlsx')

parser = argparse.ArgumentParser() 
parser.add_argument('--index', type=int, help='Select index from the excel sheet',default=35)
parser.add_argument('--condition', type=str,help='Select run type: test, complete, blink',default= 'blink')
parser.add_argument('--folder_name', type=str,help='Set folder name',default= 'test_trial')
parser.add_argument('--path_to_blender', type=str,help='Set blender path',default= 'C:\Program Files\Blender Foundation\Blender 2.90\\blender.exe')
parser.add_argument('--frame_cap', type=int,help='Number of frames to render for blink-data. -1 means there is no limit',default=-1)


args = parser.parse_args()
filename=args.folder_name

print(args)

head_model=str(data['Model'][args.index])
focal_length=str(data['focal_length'][args.index])
cornea=str(data['cornea'][args.index])
iris=str(data['iris'][args.index])
sclera=str(data['sclera'][args.index])
trial_name=data['Trial'][args.index]

camera_elevation = data['Camera_Orientation'][args.index].split(',')[0]
camera_roll = data['Camera_Orientation'][args.index].split(',')[1]
camera_azimuthal = data['Camera_Orientation'][args.index].split(',')[2]
head_elevation = data['Head_Orientation'][args.index].split(',')[0]
head_roll = data['Head_Orientation'][args.index].split(',')[1]
head_azimuthal = data['Head_Orientation'][args.index].split(',')[2]
distance = str(data['camera-x'][args.index]) + ',' + str(data['camera-y'][args.index]) + ',' + str(data['camera-z'][args.index])
inputfile="./GIW_Data/giw_"+args.condition 
if args.condition=='blink':
    blend_file=head_model+'_v8-pupil.blend'
else:
    blend_file=head_model+'_v6-pupil.blend'
    
  
subprocess.call([args.path_to_blender,'-b','static_model/'+head_model+'/'+blend_file,
                 '-P','RIT-Eyes_sequential_render.py','--','--model_id',head_model,'--inp',
                 inputfile, '--data_source','seq','--data_source_path',trial_name+'.p',
                 '--camera_focal_length', focal_length,'--light_1_loc',',-1,1,0',
                 '--light_2_loc',',-1,0.5,0','--light_1_energy','100','--light_2_energy','100',
                 '--camera_elevation',','+camera_elevation, '--head_elevation',','+head_elevation,
                 '--camera_roll',','+camera_roll, '--head_roll',','+head_roll,
                 '--camera_azimuthal',','+camera_azimuthal, '--head_azimuthal',','+head_azimuthal,
                 '--camera_distance',','+distance,'--camera_sensor_size','35','--pupil', ',1,3',
                 '--cornea', cornea, '--iris_textures',iris,'--sclera_textures',sclera, 
                 '--glass','0','--no_cornea','1','--no_reflection','1','--env','0',
                 '--output_file',filename,'--frame_cap',str(args.frame_cap)])

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
