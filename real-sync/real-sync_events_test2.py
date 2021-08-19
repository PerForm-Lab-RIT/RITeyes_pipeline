

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

head_model=['1', '1', '16', '14', '3', '3', '4', '4', '5', '5', '6', '6', '7', '7', '8', '9', '9', '10', '10', '11', '11', '12', '12', '21', '14', '15', '16', '17', '18', '19', '20', '2', '22', '23', '24']
person_id=['17', '16', '17', '12', '18', '22', '18', '8', '18', '2', '19', '3', '19', '3', '20', '20', '6', '22', '6', '2', '8', '22', '8', '2', '9', '9', '10', '1', '12', '12', '12', '15', '1', '16', '17']
trial_id=['2', '1', '3', '1', '1', '2', '3', '3', '4', '3', '2', '1', '3', '2', '1', '3', '1', '1', '2', '1', '1', '3', '2', '2', '1', '2', '1', '2', '2', '3', '4', '3', '1', '2', '1']
filename='2021_March_23_v5'

for j in range(1,len(head_model)):
    data = pd.read_pickle('/home/akc5959/RC_sequential/RC_sequential/giw_full/PrIdx_'+person_id[j]+'_TrIdx_'+trial_id[j]+'.p')
    path='/home/akc5959/RC_sequential/RC_sequential/GIW_renderings/'+head_model[j]+'/'+filename+'/PrIdx_'+person_id[j]+'_TrIdx_'+trial_id[j]+'/'
    #path='/shared/rc/riteyes/PrIdx_3_TrIdx_1/'
    size=(1280,480)
    out = cv2.VideoWriter(os.path.join(path,'real-syn.avi'),cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
    frame_idx=data['frame_no']
    time_stamp_120=data['time_stamp_120']
    time_stamp_300=data['time_stamp_300']
    classes=data['classes']
    fontsize = 40
    font = ImageFont.truetype("arial.ttf", fontsize)
    cap = cv2.VideoCapture('/shared/rc/riteyes/RC_sequential/FinalSet/'+person_id[j]+'/'+trial_id[j]+'/eye1.mp4')
    print (path)
    event_type=["No event","Fixation","Pursuit", "Saccade" ,"Blink","Fixation"]
    for i in range (1,len(classes)):
        idx=np.argmin(np.abs(time_stamp_300-time_stamp_120[i]))
        frame_no=frame_idx[idx]
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        print('Position:'+ str( int(cap.get(cv2.CAP_PROP_POS_FRAMES))) +'  frame number:'+str(i))
        ret, frame = cap.read()
        if ret:
            syn=cv2.imread(os.path.join(path,'synthetic',str(i).zfill(4)+'.tif'))
            print (i, syn.shape, frame.shape,person_id[j],trial_id[j])
            frame=Image.fromarray(np.uint8(frame))
            d1 = ImageDraw.Draw(frame)
            d1.text((240, 400), "Event: "+event_type[int(classes[i])], fill= -1,font=font)
            file_to_save=np.concatenate((frame,syn),axis=1)
        
            out.write(file_to_save)
#           cv2.imwrite(os.path.join(path,'synthetic_real',str(i).zfill(4)+'.tif'),file_to_save)
out.release()


