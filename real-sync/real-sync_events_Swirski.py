#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 15:47:29 2021

@author: aaa
"""





import scipy.io as sp
import cv2
import pickle
import pdb
import numpy as np
import os
import pdb
import argparse
import scipy.io as sp
from PIL import ImageFont, ImageDraw, Image
import pandas as pd

head_model=['1','1','2','3','3','4','4','5','5','6','20','7','7','8','9','9','10','10','11','11','12','12','14','14','15','16','16','17','18','19','13','21','22','23','24']
person_id=['17', '16', '15', '18', '22', '18', '8', '18', '2', '19', '3', '19', '3', '20', '20', '6', '22', '6', '2', '8', '22', '8', '12', '9', '9', '17', '10', '1', '12', '12', '12', '2', '1', '16', '17']
trial_id=['2','1','3','1','2','3','3','4','3','2','1','3','2','1','3','1','1','2','1','1','3','2','1','1','2','3','1','2','2','3','4','2','1','2','1']
filename='Blink'

system='home'

if system=='rc':
    pickle_path='/shared/rc/riteyes/RIT_Eyes_Rendering/GIW_Data/giw_blink/PrIdx_'
    rendering_path='/scratch/riteyes/Blink_27May/'
    giw_video_path='/shared/rc/riteyes/RC_sequential/FinalSet/'
if system=='home':
    pickle_path='/media/aaa/backup/FB_deliverable_6_month/RIT-Eyes_deliverable/RIT_Eyes_Rendering/GIW_Data/giw_blink/PrIdx_'
    rendering_path='/media/aaa/backup/FB_deliverable_6_month/RIT-Eyes_deliverable/RIT_Eyes_Rendering/renderings/'
    giw_video_path='/media/aaa/backup/Data_from_RC/FinalSet/'
    
for j in range(26,len(head_model)):
    print (j)
    data = pd.read_pickle(pickle_path+person_id[j]+'_TrIdx_'+trial_id[j]+'.p')
    path = rendering_path+head_model[j]+'/'+filename+'/PrIdx_'+person_id[j]+'_TrIdx_'+trial_id[j]+'/'
    #path='/shared/rc/riteyes/PrIdx_3_TrIdx_1/'
    size=(1280,480)
    out = cv2.VideoWriter(os.path.join(path,'real-syn.avi'),cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
    fontsize = 40
#    font = ImageFont.truetype("arial.ttf", fontsize)
    cap = cv2.VideoCapture(giw_video_path+person_id[j]+'/'+trial_id[j]+'/eye1.mp4')
    for i in range (len(data['frame_no'])-2):
        cap.set(cv2.CAP_PROP_POS_FRAMES, data['frame_no'][i])
        print('Position:'+ str( int(cap.get(cv2.CAP_PROP_POS_FRAMES))) +'  frame number:'+str(i))
        ret, frame = cap.read()
        if ret:
            syn=cv2.imread(os.path.join(path,'synthetic',str(i+1).zfill(4)+'.tif'))
            print (i, syn.shape, frame.shape,person_id[j],trial_id[j])
            frame=Image.fromarray(np.uint8(frame))
#            d1 = ImageDraw.Draw(frame)
#            d1.text((240, 400), "Event: "+event_type[int(classes[i])], fill= -1,font=font)
            file_to_save=np.concatenate((frame,syn),axis=1)
        
            out.write(file_to_save)
#           cv2.imwrite(os.path.join(path,'synthetic_real',str(i).zfill(4)+'.tif'),file_to_save)
    print ('Completed',j)
    out.release()

