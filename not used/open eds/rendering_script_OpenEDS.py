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
import numpy as np
# Read the excel sheet
#data = pd.read_excel('GIWdata.xlsx')

#parser = argparse.ArgumentParser() 
#parser.add_argument('--path_to_blender', type=str,help='Set blender path',default= './blender-2.82a-linux64/blender')
#args = parser.parse_args()

head_models=np.arange(1,25)
focal_length='46.657'
cornea=',0'
iris=',1'
sclera=',1'
camera_elevation_list=np.arange(96,105,0.5)
camera_roll_list=np.arange(-10,0.1,0.5)
camera_azimuthal_list=np.arange(-10,10,0.5)
head_elevation_list=[0]#,-5.9363,-22.7838]
head_roll_list=[0]#,-2.576,-2.260]
head_azimuthal_list=[0.]#,17.73,3.41]
camera_distance_x=np.arange(-1.05,2.67,0.51)
camera_distance_y=np.arange(-6.98,-4.11,0.51)
camera_distance_z=np.arange(-1.75,1.75,0.51)
counter=0

class Logger():
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        self.dirname = dirname
        self.log_file = open(output_name, 'a+')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)
     
    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write_silent(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print (msg)
    def write_summary(self,msg):
        self.log_file.write(msg)
        self.log_file.write('\n')
        self.log_file.flush()
        print (msg)          
    
for head_model in head_models:

  head_model=str(head_model)
  
  LOGDIR = 'logs/{}'.format(head_model)
  os.makedirs(LOGDIR,exist_ok=True)
  logger = Logger(os.path.join(LOGDIR,'logs.log'))

  blend_file=head_model+'_v7-pupil.blend'
  for camera_elevation in camera_elevation_list:
    for camera_roll in camera_roll_list:
      for camera_azimuthal in camera_azimuthal_list:
        for head_elevation in head_elevation_list:
          for head_roll in head_roll_list:
            for head_azimuthal in head_azimuthal_list:
              camera_distance_x=[np.random.randint(-105,267)/100]#np.arange(-1.05,2.67,0.51)
              camera_distance_y=[np.random.randint(-698,-411)/100]#np.arange(-6.98,-4.11,0.51)
              camera_distance_z=[np.random.randint(-175,175)/100]#np.arange(-1.75,1.75,0.51)
              for camera_x in camera_distance_x:
                for camera_y in camera_distance_y:
                  for camera_z in camera_distance_z: 
                    camera_elevation=str(camera_elevation)
                    camera_roll=str(camera_roll)
                    camera_azimuthal=str(camera_azimuthal)
                    head_elevation=str(head_elevation)
                    head_roll=str(head_roll)
                    head_azimuthal=str(head_azimuthal)
                         
                    filename=str(counter)
                    distance = str(camera_x) + ',' + str(camera_y) + ',' + str(camera_z)
                    counter=counter+1
                    if counter%1000==0:
                        print (counter)
                    logger.write(str(['./blender-2.82a-linux64/blender','-b','static_model/'+head_model+'/'+blend_file,
                 '-P','static_render_Best_version_April_25_individual.py','--',
                 '--model_id',head_model,'--data_source','random','--number','1','--camera_focal_length', focal_length,'--light_1_loc',',-1,1,0',
                 '--light_2_loc',',-1,0.5,0','--light_1_energy','100','--light_2_energy','100',
                 '--camera_elevation',','+camera_elevation, '--head_elevation',','+head_elevation,
                 '--camera_roll',','+camera_roll, '--head_roll',','+head_roll,
                 '--camera_azimuthal',','+camera_azimuthal, '--head_azimuthal',','+head_azimuthal,
                 '--camera_distance',','+distance,'--camera_sensor_size','35','--pupil', ',1,3',
                 '--cornea', cornea, '--iris_textures',iris,'--sclera_textures',sclera, 
                  '--eye_elevation', ',-20,20' ,'--eye_azimuthal' ,',-20,20', 
                 '--glass','0','--no_cornea','1','--no_reflection','1','--env','0',
                 '--output_file',str(filename)]))
                    subprocess.call(['./blender-2.82a-linux64/blender','-b','static_model/'+head_model+'/'+blend_file,
                                     '-P','static_render_Best_version_April_25_individual.py','--',
                                     '--model_id',head_model,'--data_source','random','--number','1','--camera_focal_length', focal_length,'--light_1_loc',',-1,1,0',
                                     '--light_2_loc',',-1,0.5,0','--light_1_energy','100','--light_2_energy','100',
                                     '--camera_elevation',','+camera_elevation, '--head_elevation',','+head_elevation,
                                     '--camera_roll',','+camera_roll, '--head_roll',','+head_roll,
                                     '--camera_azimuthal',','+camera_azimuthal, '--head_azimuthal',','+head_azimuthal,
                                     '--camera_distance',','+distance,'--camera_sensor_size','35','--pupil', ',1,3',
                                     '--cornea', cornea, '--iris_textures',iris,'--sclera_textures',sclera, 
                                      '--eye_elevation', ',-20,20' ,'--eye_azimuthal' ,',-20,20', 
                                     '--glass','0','--no_cornea','1','--no_reflection','1','--env','0',
                                     '--output_file',str(filename)])

  
  
