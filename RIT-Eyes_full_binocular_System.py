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
import math

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
gaze_data_dictList = None


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
	global gaze_data_dictList
	gaze_data_dictList = gaze_data.to_dict("records")
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


def printArgs() -> None:
	''' For debug, print input arguments '''
	print(args)

## Initialize Functions Ends

getDataPath()
data_directory = os.path.join(default_data_path, str(person_idx), str(trial_idx))
readGazeData(data_directory)

####	Data Processing	Starts	####

##	Data Processing Functions ##
def splitGazeDataByFrame() -> list:
	'''
	Split gaze_data_dictList into smaller lists of dictionaries. Each smaller list has dictionarys with same world_index

	return: list<list<dict>, list<dict>> 
	'''
	frameDictListsByWorldIndex = []
	singleFrameList:list = []

	last_world_frame = 0
	for i in gaze_data_dictList:
		world_frame = int(i["world_index"])
		if last_world_frame == world_frame:
			singleFrameList.append(i)
		elif (world_frame > last_world_frame):

			# if there is a frame drop
			if (world_frame - last_world_frame) > 1:
				frame_difference = world_frame - last_world_frame
				for x in range(1, frame_difference):
					frameDictListsByWorldIndex.append([])

			frameDictListsByWorldIndex.append(singleFrameList.copy())
			singleFrameList = []
			singleFrameList.append(i)
			last_world_frame = world_frame
		else:
			continue

	if len(singleFrameList) != 0:
		frameDictListsByWorldIndex.append(singleFrameList.copy())

	return frameDictListsByWorldIndex

def _debug_FindMissingFrame(frameDictListsByWorldIndex:list):
	index = 1
	for l in frameDictListsByWorldIndex:
		data_world_index = int(l[0]["world_index"])
		step = data_world_index - index
		if step == 1:
			index = data_world_index
		elif step > 1:
			print("Missing world index:", index+1)
			index = data_world_index

def findBestFrameData(world_frame:int, frameDictListsByWorldIndex:list):
	'''
	Find the best frame data from multiple data tuples in a world frame.
	Consider the highest confidence first. 
	'''
	Eye0_best = None
	Eye1_best = None

	try: 
		data_list = frameDictListsByWorldIndex[world_frame]
	except:
		print("world frame index error:", world_frame)
	# print(data_list) # for debug

	# Loop through all data in a world frame
	# Loop for Eye0 First
	best_confidence = 0.0
	data_list_size = len(data_list)

	for e in range(0,data_list_size):
		data_confidence = float(data_list[e]["confidence"])
		# print(world_frame, e," - Data confidence:", data_confidence) # for debug
		if pd.isna(data_list[e]["eye_center0_3d_x"]):
			continue
		else:
			if data_confidence == 1.0:
				Eye0_best = e
				break
			elif data_confidence > best_confidence:
				best_confidence = data_confidence
				Eye0_best = e
	

	# Loop for Eye1 Next
	best_confidence = 0.0
	data_list_size = len(data_list)

	for e in range(0,data_list_size):
		data_confidence = float(data_list[e]["confidence"])
		#print(world_frame, e," - Data confidence:", data_confidence) # for debug
		if pd.isna(data_list[e]["eye_center1_3d_x"]):
			continue
		else:
			if data_confidence == 1.0:
				Eye1_best = e
				break
			elif data_confidence >= best_confidence:
				best_confidence = data_confidence
				Eye1_best = e
	

	return [Eye0_best, Eye1_best]

def _debug_checkmismatch(frameDictListsByWorldIndex):
	index = 0
	for i in frameDictListsByWorldIndex:
		if len(i) == 0:
			index += 1
			continue

		if int(i[0]["world_index"]) != index:
			print("Find index mismatch", index, int(i[0]["world_index"]))
			break
			index += 1





## Data Processing ##

frameDictListsByWorldIndex = splitGazeDataByFrame()
frame_cap = int(frameDictListsByWorldIndex[-1][-1]["world_index"])
total_frames = len(frameDictListsByWorldIndex)
print("Total world frames in this video: ", total_frames)
print("Framecap: ", frame_cap)
## print(pd.isna(frameDictListsByWorldIndex[0][1]["eye_center1_3d_x"])) # See if a value is missing as nan
#print(findBestFrameData(2, frameDictListsByWorldIndex)) # for debug, testing frameDictListsByWorldIndex



####	Data Processing Ends 	####



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

def setUpGazeAnimationFrames(frameCount:int, Eye0, Eye1, frameDictListsByWorldIndex):
	'''
	Use processed data to set up Eyes' location and rotation for each frame
	'''
	print("Setting up eye frames from gaze data...")
	# Loop through all frames
	for frame_index in range(1, frameCount):
		Eyes_best_data = findBestFrameData(frame_index, frameDictListsByWorldIndex)


		# Setting up Eye0 frame
		if (Eyes_best_data[0] != None):
			Eye0_dataDict = frameDictListsByWorldIndex[frame_index][Eyes_best_data[0]]
			Eye0.location[0] = Eye0_dataDict["eye_center0_3d_x"] * 0.1
			Eye0.location[1] = Eye0_dataDict["eye_center0_3d_y"] * 0.1
			Eye0.location[2] = Eye0_dataDict["eye_center0_3d_z"] * 0.1

			Eye0.rotation_euler[0] = Eye0_dataDict["gaze_normal0_x"]
			Eye0.rotation_euler[1] = Eye0_dataDict["gaze_normal0_y"]
			Eye0.rotation_euler[2] = Eye0_dataDict["gaze_normal0_z"]

			Eye0.keyframe_insert(data_path="location", frame=frame_index)

		Eye0.keyframe_insert(data_path="location", frame=frame_index)
		Eye0.keyframe_insert(data_path="rotation_euler", frame=frame_index)

		# Setting up Eye1 frame
		if (Eyes_best_data[1] != None):
			Eye1_dataDict = frameDictListsByWorldIndex[frame_index][Eyes_best_data[1]]
			Eye1.location[0] = Eye1_dataDict["eye_center1_3d_x"] * 0.1
			Eye1.location[1] = Eye1_dataDict["eye_center1_3d_y"] * 0.1
			Eye1.location[2] = Eye1_dataDict["eye_center1_3d_z"] * 0.1

			Eye1.rotation_euler[0] = Eye1_dataDict["gaze_normal1_x"]
			Eye1.rotation_euler[1] = Eye1_dataDict["gaze_normal1_y"]
			Eye1.rotation_euler[2] = Eye1_dataDict["gaze_normal1_z"]

			
		Eye1.keyframe_insert(data_path="location", frame=frame_index)
		Eye1.keyframe_insert(data_path="rotation_euler", frame=frame_index)
	print("Completed setting eye frames.")



## Start Blender
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

# Position two Eyes, Revising, To be finished
setUpGazeAnimationFrames(total_frames - 1, Eye0, Eye1, frameDictListsByWorldIndex)

# temporarily save to a file
bpy.ops.wm.save_as_mainfile(filepath="./Stage.blend")

#### Blender Scene Edit Ends        ####
