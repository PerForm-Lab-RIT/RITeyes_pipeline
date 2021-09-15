#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 23:30:08 2021

@author: aayush
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pylab
import glob
from PIL import Image
import random
import cv2

#params = {'legend.fontsize': 30,
#          'figure.figsize': (10,5),
#         'axes.labelsize': 30,
#         'axes.titlesize':30,
#         'xtick.labelsize':30,
#         'ytick.labelsize':30}
#pylab.rcParams.update(params)  



filename='/media/aaa/backup/GIW_right/Saccade/'
folders_train=['374saccade', 'Person_2_Trial_117_blink','384saccade','Person_8_Trial_3106_blink']
folders_test=['126saccade','Person_9_Trial_16_blink']
file_real='/media/aaa/backup/Data_from_RC/Video/orig/'

m=0
n=0      

def plotter(gs,n,m,data,title='',color_set='k'):
    ax = plt.subplot(gs[n, m])
    ax.imshow(data,cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([]) 
    ax.axis('off')
    ax.set_title(title,color=color_set)
    
def plotter_text(gs,n,m,data,title='',color_set='k'):
    ax = plt.subplot(gs[n, m])
    ax.text(0.5, 0.5, title, horizontalalignment='center',verticalalignment='center', color=color_set,transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([]) 
    ax.axis('off')

for i in range(1,207):
    print (i)
    fig = plt.figure(figsize=(13.5,9),dpi=100)#, constrained_layout=False)
    gs=gridspec.GridSpec(3, 3)
    gs.update(left=0.05, right=0.95, wspace=0.001)
    
    image_real=np.array(Image.open(file_real+str(i)+'.png'))
    img1=cv2.imread('/media/aaa/backup/Data_from_RC/Video/3/2021_April_15/PrIdx_6_TrIdx_2/synthetic/'+str(i).zfill(4)+'.tif')
    img2=cv2.imread('/media/aaa/backup/Data_from_RC/Video/10/2021_April_15/PrIdx_6_TrIdx_2/maskwith_skin/'+str(i).zfill(4)+'.tif')
    img3=cv2.imread('/media/aaa/backup/Data_from_RC/Video/8/2021_April_15/PrIdx_6_TrIdx_2/synthetic/'+str(i).zfill(4)+'.tif')
    img4=cv2.imread('/media/aaa/backup/Data_from_RC/Video/10/2021_April_15/PrIdx_6_TrIdx_2/synthetic/'+str(i).zfill(4)+'.tif')
    img5=cv2.imread('/media/aaa/backup/Data_from_RC/Video/11/2021_April_15/PrIdx_6_TrIdx_2/synthetic/'+str(i).zfill(4)+'.tif')
    img6=cv2.imread('/media/aaa/backup/Data_from_RC/Video/12/2021_April_15/PrIdx_6_TrIdx_2/synthetic/'+str(i).zfill(4)+'.tif')
    img7=cv2.imread('/media/aaa/backup/Data_from_RC/Video/16/2021_April_15/PrIdx_6_TrIdx_2/synthetic/'+str(i).zfill(4)+'.tif')
    img8=cv2.imread('/media/aaa/backup/Data_from_RC/Video/22/2021_April_15/PrIdx_6_TrIdx_2/synthetic/'+str(i).zfill(4)+'.tif')
    img9=cv2.imread('/media/aaa/backup/Data_from_RC/Video/23/2021_April_15/PrIdx_6_TrIdx_2/synthetic/'+str(i).zfill(4)+'.tif')
    img10=cv2.imread('/media/aaa/backup/Data_from_RC/Video/24/2021_April_15/PrIdx_6_TrIdx_2/synthetic/'+str(i).zfill(4)+'.tif')
    img20=cv2.imread('/media/aaa/backup/Data_from_RC/Video/10/2021_April_15/PrIdx_6_TrIdx_2/maskwithout_skin/'+str(i).zfill(4)+'.tif')

    plotter(gs,0,0,image_real,'Real sequence','r')    
    plotter(gs,0,1,cv2.cvtColor(img2, cv2.COLOR_BGR2RGB),'2D segmentation mask','r')
    plotter(gs,0,2,cv2.cvtColor(img20, cv2.COLOR_BGR2RGB),'3D segmentation mask','r')

    plotter(gs,1,0,img1)#,'Individual frames','b')
    plotter(gs,1,1,img7)#,'Randomized Sequences','g')
    plotter(gs,1,2,img3)#,'Ordered Sequences','r')
    plotter(gs,2,0,img4)#,'Individual frames','b')
    plotter(gs,2,1,img5)#,'Randomized Sequences','g')
    plotter(gs,2,2,img10)
#    plotter(gs,2,0,img10)


    
    plt.tight_layout()
    gs.tight_layout(fig)        
    plt.savefig('/media/aaa/backup/Data_from_RC/Video/plots/'+str(i)+'.png',dpi=100)
    
    img2=np.uint8(np.array(Image.open('/media/aaa/backup/Data_from_RC/Video/plots/'+str(i)+'.png')))
    value=(255,0,0)
    img2[310:312,50:1300,:3]=value
    img2[310:890,50:52,:3]=value
    img2[890:892,50:1300,:3]=value
    img2[310:890,1298:1300,:3]=value

    plt.imsave('/media/aaa/backup/Data_from_RC/Video/plots/'+str(i)+'.png',np.uint8(img2),dpi=100)
    plt.imsave('/media/aaa/backup/Data_from_RC/Video/plots/'+str(206+i)+'.png',np.uint8(img2),dpi=100)
    plt.imsave('/media/aaa/backup/Data_from_RC/Video/plots/'+str(412+i)+'.png',np.uint8(img2),dpi=100)
    plt.imsave('/media/aaa/backup/Data_from_RC/Video/plots/'+str(618+i)+'.png',np.uint8(img2),dpi=100)
                