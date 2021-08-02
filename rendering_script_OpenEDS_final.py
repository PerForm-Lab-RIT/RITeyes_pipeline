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

parser = argparse.ArgumentParser() 
parser.add_argument('--filename', type=str,help='Set parameters',default= '1')
args = parser.parse_args()

def data_convert(data,index):
    return (data.split(',')[index]).replace('\'','').strip(' ')

number_of_frames='1'
linepath='./rendering_files/'+args.filename+'.txt'
lineDescriptor = open(linepath)
line = True
while line:
  line = lineDescriptor.readline()
  if line!='\n' and line:
    
      head_model=str(int(data_convert(line,7)))
      print (line.split(',')[0].strip('[').replace('\'',''))
      subprocess.call([line.split(',')[0].strip('[').replace('\'',''),
                      data_convert(line,1),data_convert(line,2).replace('v7','v8'),data_convert(line,3),
                      data_convert(line,4),data_convert(line,5),data_convert(line,6),
                      data_convert(line,7),data_convert(line,8),data_convert(line,9),
                      data_convert(line,10),number_of_frames,data_convert(line,12),
                      data_convert(line,13),data_convert(line,14),','+data_convert(line,16)+','+data_convert(line,17)+','+data_convert(line,18),
                      data_convert(line,19),','+data_convert(line,21)+','+data_convert(line,22)+','+data_convert(line,23),
                      data_convert(line,24),data_convert(line,25),data_convert(line,26),data_convert(line,27),
                      data_convert(line,28),','+data_convert(line,30),
                      data_convert(line,31),','+data_convert(line,33),
                      data_convert(line,34),','+data_convert(line,36),
                      data_convert(line,37),','+data_convert(line,39),
                      data_convert(line,40),','+data_convert(line,42),
                      data_convert(line,43),','+data_convert(line,45),
                      data_convert(line,46),','+data_convert(line,48)+','+data_convert(line,49)+','+data_convert(line,50),
                      data_convert(line,51),data_convert(line,52),
                      data_convert(line,53),','+data_convert(line,55)+','+data_convert(line,56),
                      data_convert(line,57),','+data_convert(line,59),
                      data_convert(line,60),','+data_convert(line,62),
                      data_convert(line,63),','+data_convert(line,65),
                      data_convert(line,66),','+data_convert(line,68)+','+data_convert(line,69),
                      data_convert(line,70),','+data_convert(line,72)+','+data_convert(line,73),
                      data_convert(line,74),data_convert(line,75),
                      data_convert(line,76),data_convert(line,77),data_convert(line,78),
                      data_convert(line,79),data_convert(line,80),data_convert(line,81),
                      data_convert(line,82),line.split(',')[-1].strip(']\n').replace('\'','').strip(' ')])
      
      subprocess.call(['rm','-rf','static_model/'+head_model+'/'+str(int(line.split(',')[-1].strip(']\n').replace('\'',''))).strip(' ')])  
        
