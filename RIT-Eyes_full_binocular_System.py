'''
Script for constructing full binocular system for RIT-Eyes

@author: Chengyi Ma
'''
import os
import sys
#sys.path.append("C:\\Users\\mcy13\\anaconda3\\envs\\RITEyes\\Lib")
#os.environ['PYTHONPATH'] = "C:\Python39"
import argparse
import pandas as pd
import numpy as np
import subprocess

####    Initialize section starts   ####
## Decide on what to import on start up.
## First, need to understand how the environment works.

## Handle arguments:

# for debug
print("sys.argv before", sys.argv)

if '--' in sys.argv:
    argv = sys.argv[sys.argv.index('--') + 1:]

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type= int, help='Choose a head model (1-24)', default=1)
parser.add_argument("--person_idx", type=int, help='Person index in the GIW Data folder', default=0)
parser.add_argument('--trial_idx', type=int, help='Trial index in the GIW Data folder', default=0)

args = parser.parse_args(argv)

## Global Variables Starts  ##
default_data_path = ""      # Where all person data rest
data_directory = ""         # Where the data rest for a particualr person_idx and trial_idx
model_id = args.model_id
person_idx = args.person_idx
trial_idx = args.trial_idx

gaze_data:pd.DataFrame = None
eye_locations = [] # a list of two sub_lists [[eye0x, eye0y, eye0z], [eye1x, eye1y, eye1z]]


## Global Varaiables Ends   ##

#### Blender Initialization Starts ####
isBlenderProcess = False
#blender_path = "./blender-2.93.3-linux-x64/blender"
blender_path = "D:/Softwares/Blender/blender.exe"

try:
    import bpy
    isBlenderProcess = True
except ModuleNotFoundError:
    print("Blender not detected, starting Blender now")
    subprocess.call([
        blender_path, 
        '-b',
        '--python','RIT-Eyes_full_binocular_System.py',
        '--',
        '--model_id',
        str(args.model_id)
        ])
    sys.exit()

print("isBlenderProcess", isBlenderProcess)


#### Blender Initialization Ends ####

## Initialize Functions Start
def getDataPath():
    """
    Initialize raw data path from file "data_path.txt"

    Path save to default_data_path
    """
    try:
        with open('data_path.txt', 'r') as default_data_path_file:
            global default_data_path 
            default_data_path = default_data_path_file.read().strip()
    except Exception:
        print('No data_path.txt file!')
        default_data_path = 'D:/Raw Eye Data/'

def readGazeData(data_directory:str) -> None:
    global gaze_data
    gaze_data = pd.read_csv(os.path.join(data_directory, "exports/gaze_positions.csv"))
    #print(type(gaze_data))
    return None

def getHighestConfidenceFrame(gaze_data_dictList:dict) -> dict:
    '''
    Find the tuples with highest confidence estimation for L and R eyes
    return in a dictionary of 4 pairs of keys and values:
        "L_index" :         (int)highest_confidence_index_L,
        "L_confidence" :    (int)highest_confidence_L,
        "R_index" :         (int)highest_confidence_index_R,
        "R_confidence" :    (int)highest_confidence_R

    '''
    highest_confidence_L = 0.0
    highest_confidence_index_L = 0
    highest_confidence_R = 0.0
    highest_confidence_index_R = 0

    for i in range(0, len(gaze_data_dictList)):
        float(gaze_data_dictList[i]["confidence"])
        this_confidence = float(gaze_data_dictList[i]["confidence"])

        if this_confidence >= highest_confidence_L:
            if pd.isna(gaze_data_dictList[i]["eye_center0_3d_x"]):
                pass
            else:
                highest_confidence_L = this_confidence
                highest_confidence_index_L = i

        if this_confidence >= highest_confidence_R:
            if pd.isna(gaze_data_dictList[i]["eye_center1_3d_x"]):
                pass
            else:
                highest_confidence_R = this_confidence
                highest_confidence_index_R = i

        if highest_confidence_L == 1.0 and highest_confidence_R == 1.0:
            break

    return {
        "L_index" : highest_confidence_index_L,
        "L_confidence" : highest_confidence_L,
        "R_index" : highest_confidence_index_R,
        "R_confidence" : highest_confidence_R
    }


def getEyelocations(gaze_data:pd.DataFrame):
    '''
    From gaze data, we find the best fit eye locations for both eyes in world camera space.
    '''
    gaze_data_dictList = gaze_data.to_dict("records")
    
    # Find highest confident frame:
    highConfidenceFrameDict = getHighestConfidenceFrame(gaze_data_dictList)
    eye0_highconfidence_tuple_index = highConfidenceFrameDict["L_index"]
    eye1_highconfidence_tuple_index = highConfidenceFrameDict["R_index"]
    
    eye0_location = [
        gaze_data_dictList[eye0_highconfidence_tuple_index]["eye_center0_3d_x"],
        gaze_data_dictList[eye0_highconfidence_tuple_index]["eye_center0_3d_y"],
        gaze_data_dictList[eye0_highconfidence_tuple_index]["eye_center0_3d_z"],
    ]

    eye1_location = [
        gaze_data_dictList[eye1_highconfidence_tuple_index]["eye_center1_3d_x"],
        gaze_data_dictList[eye1_highconfidence_tuple_index]["eye_center1_3d_y"],
        gaze_data_dictList[eye1_highconfidence_tuple_index]["eye_center1_3d_z"],
    ]

    # for debug
    # print(eye0_highconfidence_tuple_index, eye0_location, eye1_highconfidence_tuple_index, eye1_location)
    return [eye0_location, eye1_location]

def printArgs() -> None:
    ''' For debug, print input arguments '''
    print(args)

## Initialize Functions Ends

getDataPath()
data_directory = os.path.join(default_data_path, str(person_idx), str(trial_idx))
readGazeData(data_directory)
eye_locations = getEyelocations(gaze_data)
print(eye_locations)

####    Initialize section ends     ####

####    Blender Scene Edit Starts   ####

## Blender Opertaion Functions:
def openBlenderFile():
    '''
    Open Blender file from Command Line Argument
    '''
    if args.model_id != 0:
        bpy.ops.wm.open_mainfile(filepath=os.path.join("static_model",str(args.model_id),str(args.model_id)+"_v9-pupil.blend"))

def selectObjectHierarchy(obj):
    '''
    Select the whole hierarchy from a give parent obj, Be aware of that it only looks downward to the children.
    '''
    obj.select_set(True)
    for child_obj in obj.children:
        selectObjectHierarchy(child_obj)

def hideObjectHierarchy(obj):
    '''
    hide an obj and its all children
    '''
    obj.hide_set(True)
    obj.hide_render = True
    for child_obj in obj.children:
        hideObjectHierarchy(child_obj)


openBlenderFile()

## Blender Objects
Eye0 = bpy.data.objects["Eye.Wetness"]
Eye1 = None
Armature = bpy.data.objects["Armature"]

## Blender Settings:
bpy.context.scene.unit_settings.scale_length = 0.01
bpy.context.scene.unit_settings.length_unit = 'CENTIMETERS'


## Blender Operations
bpy.ops.object.select_all(action="DESELECT")
selectObjectHierarchy(Eye0)
bpy.ops.object.duplicate()
bpy.ops.object.select_all(action="DESELECT")
Eye1 = bpy.data.objects["Eye.Wetness.001"]

# Temporarily Hide Head Model
hideObjectHierarchy(Armature)

# Position two Eyes
Eye0.location = np.asarray(eye_locations[0]) * 0.1
Eye1.location = np.asarray(eye_locations[1]) * 0.1

# temporarily save to a file
bpy.ops.wm.save_as_mainfile(filepath="./Stage.blend")

#### Blender Scene Edit Ends        ####
