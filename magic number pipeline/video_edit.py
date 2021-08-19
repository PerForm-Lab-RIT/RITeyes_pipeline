#import scipy.io as sp
import cv2
import pickle
import pdb
import numpy as np
import os
import scipy.io as sp
from PIL import ImageFont, ImageDraw, Image
import pandas as pd
import threading

frames = 500
mix = False

path= 'D:/RITeyes_pipeline/renderings/4/test_trial/PrIdx_8_TrIdx_3'
file_name = 'PrIdx_8_TrIdx_3.p'

with open('D:/RITeyes_pipeline/GIW_Data/giw_blink/'+file_name,'rb') as dataPickle:
    data = pickle.load(dataPickle)

#path='/shared/rc/riteyes/PrIdx_3_TrIdx_1/'
size=(640*2, 480*2)
out = cv2.VideoWriter(os.path.join(path,'real-syn.avi'),cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
fontsize = 40
#font = ImageFont.truetype("arial.ttf", fontsize)
cap = cv2.VideoCapture(path+'/eye1.mp4')
for i in range (1,frames):#(len(1, data['frame_no'])):
    cap.set(cv2.CAP_PROP_POS_FRAMES, data['frame_no'][i])
    ret, frame = cap.read()
    if ret:
        if (i%1 == 0):
            print('Position:'+ str( int(cap.get(cv2.CAP_PROP_POS_FRAMES))) +'  frame number:'+str(i))

        # reads the two rendered images
        syn=cv2.imread(os.path.join(path,'synthetic',str(i).zfill(4)+'.tif'))
        syn_mask=cv2.imread(os.path.join(path,'maskwith_skin',str(i).zfill(4)+'.tif'))
        mask_noskin=cv2.imread(os.path.join(path,'maskwithout_skin',str(i).zfill(4)+'.tif'))
        
        # applys the syn images to the mask
        if mix:
            for i in range(syn_mask.shape[0]):
                for j in range(syn_mask.shape[1]):
                    for k in range(syn_mask.shape[2]):
                        if (syn_mask[i,j][k] < 100):
                            syn_mask[i,j][k] = syn[i,j][k]


        # concats the images side-by-side-by-side
        row1 = np.concatenate((frame,syn),axis=1)
        row2 = np.concatenate((syn_mask,mask_noskin),axis=1)
        file_to_save=np.concatenate((row1, row2),axis=0)
        
        # if (i == 1):
        #     os.chdir(path)
        #     flat_file = np.concatenate((row1, row2),axis=1)
        #     cv2.imwrite('file_flat.png', flat_file)

        out.write(file_to_save)
        #out.write(row1)

out.release()