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

head_model='22'
person_id='1'
trial_id='1'
filename='blinks_v4'

data = pd.read_pickle('/media/aaa/backup/FB_deliverable_6_month/RIT-Eyes_deliverable/giw_small/PrIdx_'+person_id+'_TrIdx_'+trial_id+'.p')
path='/media/aaa/backup/Data_from_RC/Video/'+head_model+'/'+filename+'/PrIdx_'+person_id+'_TrIdx_'+trial_id+'/'
data = pd.read_pickle('/media/aaa/backup/FB_deliverable_6_month/RIT-Eyes_deliverable/giw_small/PrIdx_'+person_id+'_TrIdx_'+trial_id+'_blinks4.p')
path='/media/aaa/backup/FB_deliverable_6_month/RIT-Eyes_deliverable/static_model/'+head_model+'/'+filename+'/PrIdx_'+person_id+'_TrIdx_'+trial_id+'_blinks4/'
i=0
x=1
size=(1280,480)
out = cv2.VideoWriter(os.path.join(path,'real-syn.avi'),cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
frame_idx=data['frame_no']
time_stamp_120=data['time_stamp_120']
time_stamp_300=data['time_stamp_300']
classes=data['classes']
fontsize = 40
#font = ImageFont.truetype("arial.ttf", fontsize)
cap = cv2.VideoCapture('/media/aaa/backup/Data_from_RC/FinalSet/'+person_id+'/'+trial_id+'/eye1.mp4')
i=0
event_type=["No event","Fixation","Pursuit", "Saccade" ,"Blink","Fixation"]
frame_number=[]
for i in range (1,len(classes)):
    idx=np.argmin(np.abs(time_stamp_300-time_stamp_120[i]))
    frame_no=frame_idx[idx]
    frame_number.append(frame_no)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    print('Position:'+ str( int(cap.get(cv2.CAP_PROP_POS_FRAMES))) +'  frame number:'+str(i))
    ret, frame = cap.read()
#    cv2.imwrite('/media/aaa/backup/Data_from_RC/Video/orig/'+str(i)+'.png',frame)
    if ret:
        syn=cv2.imread(os.path.join(path,'synthetic',str(i).zfill(4)+'.tif'))
        frame=Image.fromarray(np.uint8(frame))
#        d1 = ImageDraw.Draw(frame)
#        d1.text((240, 400), "Event: "+event_type[int(classes[i])], fill= -1,font=font)
        file_to_save=np.concatenate((frame,syn),axis=1)
        
        out.write(file_to_save)
#        cv2.imwrite(os.path.join(path,'synthetic_real',str(i).zfill(4)+'.tif'),file_to_save)
out.release()


