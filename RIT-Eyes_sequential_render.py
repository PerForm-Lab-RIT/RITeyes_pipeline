'''
Copyright (c) 2021 RIT  
G. J. Diaz (PI), R. J. Bailey, J. B. Pelz
N. Nair, A. Romanenko, A. K. Chaudhary
'''

import bpy
import math
import random
import numpy as np
import os
import pdb
import sys
import argparse
import pickle
import subprocess
import mathutils

sys.path.append(os.path.join(r"./"))
if '--' in sys.argv:
    argv = sys.argv[sys.argv.index('--') + 1:]

# Define parameters such as head model, total number of images to render, rendering storage location and gaze vectors.
# The gaze vectors can be random or loaded from a pickle file .
# Note: (1) head model should have pupil_blend and texture files (Folder structure ./static_model/{head_model_id}/{head_model_id}.pupil_blend & Textures 
# Head model can be purchased here https://www.3dscanstore.com/3d-head-models/female-retopologised-3d-head-models/male-female-3d-head-model-24xbundle
# .....................................................................................................................................................................

parser = argparse.ArgumentParser()
output_folder=os.path.join(os.getcwd(), 'renderings/')# #./GIW_renderings'#/scratch/riteyes

#########################################################################################################################################################################
# Define setup parameters/details
# MODEL_ID
parser.add_argument('--model_id', help='Choose a head model (1-24)',default='3')
parser.add_argument('--number', type=str,help='total number of images)',default= 100000) ##Only used for random static image renderings
parser.add_argument('--data_source', type=str,help='Provide a source in pickle file. Else choose random or sequential',default='random' )
parser.add_argument('--index', type=int,help='Provide a index for the sequential render',default=1 )

# This represents the type of rendering.
# random - Use this while rendering static images only(not sequential). This will render images with uniform distribution of the data (eye pose, camera distance and pose etc.)
# seq    - Use this for sequential video rendering. If using this --inp for the file directory of pickle files and --data_source_path for pickle file name( which has giw data) should be specified.
# pickle - Use this which static image rendering. Use this to render


parser.add_argument('--frame_cap', type=int,help='Number of frames to render for blink-data. -1 means there is no limit',default=-1)
#parser.add_argument('--data_source_path', type=str,help='Select data source path for pickle file',default=None)
parser.add_argument('--person_idx', type=int, help='Person index in the GIW Data folder', default=0)
parser.add_argument('--trial_idx', type=int, help='Trial index in the GIW Data folder', default=0)
parser.add_argument('--inp', help='output file',default='./GIW_Data/debug') ##To specify where to save the rendered images and mask
parser.add_argument('--start', type=str,help='start frame',default='1' )
parser.add_argument('--end', type=str,help='end frame',default='10000') # While sequential rendering if you need a specific portion of the video to rendered you can specify the start and end index
parser.add_argument('--gpu', type=str,help='GPU ids to use',default='0,1,2,3')
parser.add_argument('--output_file', type=str,help='Please make sure that there are duplicates',default='mark2')
parser.add_argument('--picklefile', type=str, help='Location of the data file', default='E:\RITEyes\Raw Data')

#########################################################################################################################################################################
# Define control parameters
# Since our base model design has two IR leds: define its location (in cm) and illumination energy (0 means OFF) 
parser.add_argument('--light_1_loc', type=str,help='Location of the light source1 from camera',default='-0.5,0.5,0.5')
parser.add_argument('--light_2_loc', type=str,help='Location of the light source2 from camera',default='-0.5,-0.5,0.5')
parser.add_argument('--light_1_energy', type=str,help='Light source 1 energy',default=120)
parser.add_argument('--light_2_energy', type=str,help='Light source 2 energy',default=0)

# Define camera properties (for sequential it is better to fix the camera position. Wide range of parameters can be provided for static renderings.
# --camera_distance     : Specify the range of the camera distance to be moved from the tip of the eye in cm .
# --camera_azimuthal    : The range of camera horizontal movement in degrees.
# --camera_elevation    : The range of camera vertical movement in degrees.
# --camera_location     : Specify the location of the camera in cm from the eye center.
# --camera_focal_length : Sets the camera focal length
# --camera_sensor_size  : Sets the camera sensor size
parser.add_argument('--camera_focal_length', type=str,help='Focal length of the camera',default=35)
parser.add_argument('--camera_distance', type=str,help='Range of camera distance of the eye from the tip of the eye [from,to]',default='4,5,0')
parser.add_argument('--camera_azimuthal', type=str,help='Range of camera horizontal movement in degrees [from ,to] ',default='-55,-65')
parser.add_argument('--camera_elevation', type=str,help='Range of camera vertical movement in degrees [from ,to] (+ve means down and -ve means up)',default='20,30')
parser.add_argument('--camera_location', type=str,help='Location of the camera',default='20,30')
parser.add_argument('--camera_roll', type=str,help='Range of camera rotation (in z axis) in degrees [from ,to] (+ve means down and -ve means up)',default='-0.5,0.5')
parser.add_argument('--camera_sensor_size', type=str, help='Sensor size of the camera', default=35)

#HEAD ROTATION
parser.add_argument('--head_roll', type=str,help='Range of head rotation (in z axis) in degrees [from ,to] (+ve means down and -ve means up)',default='-0.5,0.5')
parser.add_argument('--head_azimuthal', type=str,help='Range of head rotation', default='-55,-65')
parser.add_argument('--head_elevation', type=str,help='Range of head rotation',default='20,30')


#########################################################################################################################################################################
##Define eye related details
parser.add_argument('--pupil', type=str,help='pupil size variation from 1 mm to 4 mm radius',default='1,3')
# Define cornea shape. We intially have 3 cornea shape from the mathematical equation
parser.add_argument('--cornea', type=str,help='Has to be a value from 0,1,2,sphere (as a list of strings)',default='0,1,2') # If using multiple cornea then it should be strings separated by comma eg: 0,1,2
# Define iris textures pattern to be used for renderings
parser.add_argument('--iris_textures', type=str,help='Possible options 1-9',default='1,2,3,4,5,6,7,8,9')
# Define sclera textures
parser.add_argument('--sclera_textures', type=str,help='Possible options 1-4',default='1,2,3,4')
# Sequential data type
parser.add_argument('--seq_type', type=str,help='Has to be a list of values from 1 to 4',default='all')
# Select samples of images with glasses
# Specify the percentage of images that require glasses.
# 0 means no images will have glasses
# 50% means out of 2000 image 1000 images will have glasses
# 100% means all the images will have glasses
parser.add_argument('--glass', type=str,help='Percentage of glasses (value from 0 to 100)',default=0)
# Special circumstances if you need images with:
#--no_cornea      : Render images without cornea.
#--no_reflection  : Render images without reflection.
parser.add_argument('--no_cornea',type=str,help='Hides cornea in all renderings')
parser.add_argument('--no_reflection',type=str,help='Hides reflections in all renderings')
# --eye_elevation    : Only for static image rendering. The range of eye vertical rotation in degrees(-ve means up and + means down ). Should be in from,to format.
# --eye_azimuthal    : Only for static image rendering. The range of eye horizontal rotation in degrees (-ve means left and + means right ). Should be in from,to format.
parser.add_argument('--eye_elevation', type=str,help='Eye pose elevation (from (up, to (down)))',default='-20,20')
parser.add_argument('--eye_azimuthal', type=str,help='Eye pose azhimuthal (from (left, to (right)))',default='-20,20')
parser.add_argument('--lower_offset', type=int,help='Lower blinker offset',default=0)
parser.add_argument('--upper_offset', type=int,help='Upper blinker offset',default=0)

# ENVIRONMENTAL REFLECTIONS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
parser.add_argument('--env', type=str, help='Sets the environmental reflection')
# Specify which environmental textures to be used. 0 means full black and choose any texture from 1 to 25.
# .....................................................................................................................................................................

args = parser.parse_args(argv)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
gt = {'camera_3d_center': [], 'camera_distance': [], 'iris_loc': [], 'eye_loc': [], 'gaze_angle_az': [],
      'gaze_angle_el': [], 'ortho_scale': [], 'cam_az': [], 'cam_el': [], 'glasses': [], 'pupil': [], 'eye_lid': [],
      'iris_rot': [], 'left_corner': [], 'right_corner': [], 'sclera': [], 'cam_focal_length': [],
      'cam_sensor_size': [], 'iris_texture': [], 'light1_loc': [], 'light2_loc': [], 'light1_az': [], 'light1_el': [],
      'light2_az': [], 'light2_el': [], 'sclera_texture': [], 'sclera_dark': [], 'glass_loc': [], 'glass_rot': [],
      'target': [], 'cornea': []}
sys.path = []
sys.path.append('./blender-2.82a-linux64/2.82/scripts/startup')
sys.path.append('./blender-2.82a-linux64/2.82/scripts/modules')
sys.path.append('./blender-2.82a-linux64/2.82/scripts/freestyle/modules')
sys.path.append('./blender-2.82a-linux64/2.82/scripts/addons/modules')
sys.path.append('./blender-2.82a-linux64/2.82/python/lib/python3.7')
sys.path.append('./blender-2.82a-linux64/2.82/python/lib/python3.7/site-packages')
print(
    "---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")


#########################################################################################################################################################################
# This convert functions converts/filters the arguments specified into blender specific format.
def convert(x):
    x = x.replace(" ", "")
    if '-' in x:
        return float(x[1:]) * -1
    else:

        return float(x)


#Objects are generated by the user through blender  
model=args.model_id
number=int(args.number)

camera=bpy.data.objects['Camera']   #camera as objects (location, angle)
cam=bpy.data.cameras['Camera']      #camera intrinsic properties

#allows to select perspective or orthographic projection
cam.type = 'PERSP'

empty = bpy.data.objects['Empty']
eye = bpy.data.objects["Eye.Wetness"]
iris = bpy.data.objects["EyeIris"]
sclera = bpy.data.objects["Sclera"]
glass = bpy.data.objects["Sunglasses"]

#define ashperical parameters. More parameters can be added here. (We use only three)
c1 = bpy.data.objects["63"]
c2 = bpy.data.objects["75"]
c3 = bpy.data.objects["87"]
sphere = bpy.data.objects["sphere"]
wet = bpy.data.materials['Eye_Wetness_01']
world = bpy.data.worlds['World']
inp = args.inp
if len(args.model_id) == 4 and args.model_id[3:4] == '1':
    head=bpy.data.objects["head"]
else:
    head=bpy.data.objects["Armature"]

# Lamp energy 4mW=40
l1 = bpy.data.objects['Lamp.001']
l2 = bpy.data.objects['Lamp']
l1.data.type = 'AREA'
l2.data.type = 'AREA'

#########################################################################################################################################################################
# Converting the data into format used for rendering : (Data processing)
args.camera_distance = args.camera_distance.split(',')[1:]
for x in range(len(args.camera_distance)):
    args.camera_distance[x] = convert(args.camera_distance[x])
args.camera_focal_length = float(args.camera_focal_length)
args.camera_sensor_size = float(args.camera_sensor_size)
cam.lens = args.camera_focal_length
cam.sensor_width = args.camera_sensor_size
args.pupil = args.pupil.split(',')
for x in range(len(args.pupil)):
    args.pupil[x] = convert(args.pupil[x])

args.eye_azimuthal = args.eye_azimuthal.split(',')[1:]
for x in range(len(args.eye_azimuthal)):
    args.eye_azimuthal[x] = convert(args.eye_azimuthal[x])

args.eye_elevation = args.eye_elevation.split(',')[1:]
for x in range(len(args.eye_elevation)):
    args.eye_elevation[x] = convert(args.eye_elevation[x])

args.cornea = args.cornea.split(',')[1:]
a = args.iris_textures.split(',')[1:] * number
b = args.sclera_textures.split(',')[1:] * number

args.camera_elevation = args.camera_elevation.split(',')[1:]
args.camera_azimuthal = args.camera_azimuthal.split(',')[1:]
args.camera_roll = args.camera_roll.split(',')[1:]

for x in range(len(args.camera_elevation)):
    args.camera_elevation[x] = convert(args.camera_elevation[x])

for x in range(len(args.camera_azimuthal)):
    args.camera_azimuthal[x] = convert(args.camera_azimuthal[x])

for x in range(len(args.camera_roll)):
    args.camera_roll[x] = convert(args.camera_roll[x])

args.head_elevation = args.head_elevation.split(',')[1:]
args.head_azimuthal = args.head_azimuthal.split(',')[1:]
args.head_roll = args.head_roll.split(',')[1:]



for x in range(len(args.head_elevation)):
    args.head_elevation[x] = convert(args.head_elevation[x])

for x in range(len(args.head_azimuthal)):
    args.head_azimuthal[x] = convert(args.head_azimuthal[x])

for x in range(len(args.head_roll)):
    args.head_roll[x] = convert(args.head_roll[x])


l1.data.energy = convert(args.light_1_energy)
l2.data.energy = convert(args.light_2_energy)

start = convert(args.start)
end = convert(args.end)

args.light_1_loc = args.light_1_loc.split(',')[1:]
args.light_2_loc = args.light_2_loc.split(',')[1:]
args.light_1_loc[0] = convert(args.light_1_loc[0])
args.light_1_loc[1] = convert(args.light_1_loc[1])
args.light_1_loc[2] = convert(args.light_1_loc[2])

args.light_2_loc[0] = convert(args.light_2_loc[0])
args.light_2_loc[1] = convert(args.light_2_loc[1])
args.light_2_loc[2] = convert(args.light_2_loc[2])

l1.location[0] = args.light_1_loc[0]
l1.location[1] = args.light_1_loc[1]
l1.location[2] = args.light_1_loc[2]

args.glass = int(args.glass)

l2.location[0] = args.light_2_loc[0]
l2.location[1] = args.light_2_loc[1]
l2.location[2] = args.light_2_loc[2]
sclera_mat = bpy.data.materials['Sclera material.000']
skin = bpy.data.materials['skin']

# Camera Distance from
d1 = args.camera_distance[0]

# Camera Distance to
d2 = args.camera_distance[1]

# pupil size from (0 to 1 means 1mm to 4mm)
p1 = (args.pupil[0] - 1) / 3
# pupil size to
p2 = (args.pupil[1] - 1) / 3
cor = args.cornea * number
output = args.output_file
if_glass = [0] * number
for x in range(number * args.glass // 100):
    if_glass[x] = 1


#inserting keyframes is necessary for setting different positions and model states over time. This timeline is then rendered as separate frames.
def keyframe():
    for i in range(-1, number):
        print('Keyframing: ', i)
        # camera distance from 1 cm to 3 cm
        r = random.uniform(d1, d2)

        # azimuthal
        phi = random.uniform(90 + args.camera_azimuthal[0], 90 + args.camera_azimuthal[1])
        # elevation
        theta = random.uniform(90 + args.camera_elevation[0], 90 + args.camera_elevation[1])

        # calculating camera location
        x = r * math.sin(math.radians(theta)) * math.cos(math.radians(phi))
        y = r * math.sin(math.radians(theta)) * math.sin(math.radians(phi))
        z = r * math.cos(math.radians(theta))

        # camera location based on distance and angle
        camera.location[0] = x
        camera.location[1] = -1 - y
        camera.location[2] = z

        # camera target 1 mm vertically and horizontally
        empty.location[0] = random.uniform(-0.5, 0.5)
        empty.location[1] = -1
        empty.location[2] = random.uniform(-0.5, 0.5)

        eye.rotation_euler[0] = math.radians(90) + math.radians(
            random.uniform(args.eye_elevation[0], args.eye_elevation[1]))
        eye.rotation_euler[2] = math.radians(random.uniform(args.eye_azimuthal[0], args.eye_azimuthal[1]))
        # pupil size from 1mm to 3 mm
        bpy.data.meshes['Roundcube.000'].shape_keys.key_blocks["Pupil contract"].value = random.uniform(p1, p2)

        # iris and sclera rotation (Any value from 0 , 360)
        iris.rotation_euler[2] = random.uniform(0, 6.28319)
        sclera.rotation_euler[2] = random.uniform(0, 6.28319)

        # Making sclera dark
        sclera_mat.node_tree.nodes["sclera_dark"].inputs[0].default_value = random.uniform(0, 0.8)

        #

        # select with or without glasses.(AAYUSH: what are these values. How are they selected any number you should have explanantion of how you used that value
        if if_glass[i] == 1:
            glass.hide_render = False
            glass.location[0] = random.uniform(-1.97501, -2.22824)
            glass.location[1] = random.uniform(-2.00167, -1.7237)
            glass.location[2] = random.uniform(-0.531074, -0.774396)
            glass.rotation_euler[0] = random.uniform(math.radians(88), math.radians(92))
            glass.rotation_euler[1] = random.uniform(math.radians(-2), math.radians(2))
            glass.rotation_euler[2] = random.uniform(math.radians(-2), math.radians(2))
            glass.keyframe_insert(data_path="location", frame=i)
            glass.keyframe_insert(data_path="rotation_euler", frame=i)
        else:
            glass.hide_render = True

        # Selecting random eye lid closure value from 0 to 1
        lid = random.uniform(0.0, 0.5)
        bpy.data.meshes['eye_lid'].shape_keys.key_blocks["Key 1"].value = lid
        bpy.data.meshes['l'].shape_keys.key_blocks["Key 1"].value = lid
        bpy.data.meshes['u'].shape_keys.key_blocks["Key 1"].value = lid

        # Eye lid closure keyframing
        bpy.data.meshes['eye_lid'].shape_keys.key_blocks["Key 1"].keyframe_insert(data_path="value", frame=i)
        bpy.data.meshes['l'].shape_keys.key_blocks["Key 1"].keyframe_insert(data_path="value", frame=i)
        bpy.data.meshes['u'].shape_keys.key_blocks["Key 1"].keyframe_insert(data_path="value", frame=i)

        # blend eye lid
        if lid < 0.2:
            lid = lid
        else:
            lid = lid * 1.5
        skin.node_tree.nodes["blink"].inputs[0].default_value = lid

        # keyframe blend eye lid
        skin.node_tree.nodes["blink"].inputs[0].keyframe_insert(data_path="default_value", frame=i)

        # Select cornea
        c = cor[i]
        if c == '0':
            c1.hide_render = False
            c2.hide_render = True
            c3.hide_render = True

        if c == '1':
            c2.hide_render = False
            c3.hide_render = True
            c1.hide_render = True
        if c == '2':
            c3.hide_render = False
            c2.hide_render = True
            c1.hide_render = True
        camera.rotation_euler[0] = 0
        camera.rotation_euler[1] = 0
        camera.rotation_euler[2] = 0
        camera.keyframe_insert(data_path="rotation_euler", frame=i)
        # Camera location keyframing
        camera.keyframe_insert(data_path="location", frame=i)

        # Camera target keyframing
        empty.keyframe_insert(data_path="location", frame=i)

        # Iris and sclera key framing
        eye.keyframe_insert(data_path="rotation_euler", frame=i)

        # Iris and sclera key framing
        iris.keyframe_insert(data_path="rotation_euler", frame=i)
        sclera.keyframe_insert(data_path="rotation_euler", frame=i)

        # Glass keyframe
        glass.keyframe_insert(data_path="hide_render", frame=i)

        # keyframing pupil size
        bpy.data.meshes['Roundcube.000'].shape_keys.key_blocks["Pupil contract"].keyframe_insert(data_path="value",
                                                                                                 frame=i)

        # keyframing sclera color
        sclera_mat.node_tree.nodes["sclera_dark"].inputs[0].keyframe_insert(data_path="default_value", frame=i)

        ##Cornea key framing
        c1.keyframe_insert(data_path="hide_render", frame=i)
        c2.keyframe_insert(data_path="hide_render", frame=i)
        c3.keyframe_insert(data_path="hide_render", frame=i)

# data that is keyframed from a pickle file
def keyframe():

    lower_blinker = bpy.data.objects["Armature"].pose.bones['lower_blinker']
    upper_blinker = bpy.data.objects["Armature"].pose.bones['upper_blinker']

    u_prj_x = mathutils.Vector((0,upper_blinker.tail[1],upper_blinker.tail[2]))
    l_prj_x = mathutils.Vector((0,lower_blinker.tail[1],lower_blinker.tail[2]))

    u_prj_x = u_prj_x.normalized()
    l_prj_x = l_prj_x.normalized()
    

    for i in range(number):
        print('Keyframing: ', i)
        # camera distance from 1 cm to 3 cm
        #r = random.uniform(d1, d2)

        # azimuthal
        #phi = random.uniform( args.camera_azimuthal[0], args.camera_azimuthal[1])
        # elevation
        #theta = random.uniform( args.camera_elevation[0],  args.camera_elevation[1])

        # calculating camera location
        #x = r * math.sin(math.radians(theta)) * math.cos(math.radians(phi))
        #y = r * math.sin(math.radians(theta)) * math.sin(math.radians(phi))
        #z = r * math.cos(math.radians(theta))

        # camera location based on distance and angle
        #camera.location[0] = x
        #camera.location[1] = -1 - y
        #camera.location[2] = z
        
        camera.location[0] = args.camera_distance[0]
        camera.location[1] = args.camera_distance[1]
        camera.location[2] = args.camera_distance[2]

        camera.rotation_euler[0] = math.radians(args.camera_elevation[0])
        camera.rotation_euler[1] = math.radians(args.camera_roll[0])
        camera.rotation_euler[2] = math.radians(args.camera_azimuthal[0])

        head.rotation_euler[0] = math.radians(args.head_elevation[0])
        head.rotation_euler[1] = math.radians(args.head_roll[0])
        head.rotation_euler[2] = math.radians(args.head_azimuthal[0])

        # camera target 1 mm vertically and horizontally
        empty.location[0] = random.uniform(-0.5, 0.5)
        empty.location[1] = -1
        empty.location[2] = random.uniform(-0.5, 0.5)

        eye.rotation_euler[0] = math.radians(90) + math.radians(
            random.uniform(args.eye_elevation[0], args.eye_elevation[1]))
        eye.rotation_euler[2] = math.radians(random.uniform(args.eye_azimuthal[0], args.eye_azimuthal[1]))
        # pupil size from 1mm to 3 mm
        bpy.data.meshes['Roundcube.000'].shape_keys.key_blocks["Pupil contract"].value = random.uniform(p1, p2)

        # iris and sclera rotation (Any value from 0 , 360)
        iris.rotation_euler[2] = random.uniform(0, 6.28319)
        sclera.rotation_euler[2] = random.uniform(0, 6.28319)

        # Making sclera dark
        sclera_mat.node_tree.nodes["sclera_dark"].inputs[0].default_value = random.uniform(0, 0.8)

        #

        # select with or without glasses.(AAYUSH: what are these values. How are they selected any number you should have explanantion of how you used that value
        if if_glass[i] == 1:
            glass.hide_render = False
            glass.location[0] = random.uniform(-1.97501, -2.22824)
            glass.location[1] = random.uniform(-2.00167, -1.7237)
            glass.location[2] = random.uniform(-0.531074, -0.774396)
            glass.rotation_euler[0] = random.uniform(math.radians(88), math.radians(92))
            glass.rotation_euler[1] = random.uniform(math.radians(-2), math.radians(2))
            glass.rotation_euler[2] = random.uniform(math.radians(-2), math.radians(2))
            glass.keyframe_insert(data_path="location", frame=i)
            glass.keyframe_insert(data_path="rotation_euler", frame=i)
        else:
            glass.hide_render = True

        
        u_angle = delta*0.75-random.uniform(0,delta*0.75)*0.75
        l_angle = delta*0.25-random.uniform(0,delta*0.25)*0.75

        bpy.data.objects['Armature'].pose.bones['upper_blinker'].rotation_euler[0] = u_angle
        bpy.data.objects['Armature'].pose.bones['upper_blinker'].keyframe_insert(data_path="rotation_euler", frame=i)
        bpy.data.objects['Armature'].pose.bones['lower_blinker'].rotation_euler[0] = l_angle
        bpy.data.objects['Armature'].pose.bones['lower_blinker'].keyframe_insert(data_path="rotation_euler", frame=i)

        
        # Selecting random eye lid closure value from 0 to 1
   #     lid = random.uniform(0.0, 0.5)
  #      bpy.data.meshes['eye_lid'].shape_keys.key_blocks["Key 1"].value = lid
    #    bpy.data.meshes['l'].shape_keys.key_blocks["Key 1"].value = lid
 #       bpy.data.meshes['u'].shape_keys.key_blocks["Key 1"].value = lid
#
        # Eye lid closure keyframing
     #   bpy.data.meshes['eye_lid'].shape_keys.key_blocks["Key 1"].keyframe_insert(data_path="value", frame=i)
    #    bpy.data.meshes['l'].shape_keys.key_blocks["Key 1"].keyframe_insert(data_path="value", frame=i)
   #     bpy.data.meshes['u'].shape_keys.key_blocks["Key 1"].keyframe_insert(data_path="value", frame=i)

        # blend eye lid
        #if lid < 0.2:
         #   lid = lid
        #else:
        #    lid = lid * 1.5
      #  skin.node_tree.nodes["blink"].inputs[0].default_value = lid

        # keyframe blend eye lid
       # skin.node_tree.nodes["blink"].inputs[0].keyframe_insert(data_path="default_value", frame=i)

        # Select cornea
        c = cor[i]
        if c == '0':
            c1.hide_render = False
            c2.hide_render = True
            c3.hide_render = True

        if c == '1':
            c2.hide_render = False
            c3.hide_render = True
            c1.hide_render = True
        if c == '2':
            c3.hide_render = False
            c2.hide_render = True
            c1.hide_render = True

        camera.keyframe_insert(data_path="rotation_euler", frame=i)
        # Camera location keyframing
        camera.keyframe_insert(data_path="location", frame=i)

        # Camera target keyframing
        empty.keyframe_insert(data_path="location", frame=i)

        # Iris and sclera key framing
        eye.keyframe_insert(data_path="rotation_euler", frame=i)

        # Iris and sclera key framing
        iris.keyframe_insert(data_path="rotation_euler", frame=i)
        sclera.keyframe_insert(data_path="rotation_euler", frame=i)

        # Glass keyframe
        glass.keyframe_insert(data_path="hide_render", frame=i)

        # keyframing pupil size
        bpy.data.meshes['Roundcube.000'].shape_keys.key_blocks["Pupil contract"].keyframe_insert(data_path="value",
                                                                                                 frame=i)

        # keyframing sclera color
        sclera_mat.node_tree.nodes["sclera_dark"].inputs[0].keyframe_insert(data_path="default_value", frame=i)

        ##Cornea key framing
        c1.keyframe_insert(data_path="hide_render", frame=i)
        c2.keyframe_insert(data_path="hide_render", frame=i)
        c3.keyframe_insert(data_path="hide_render", frame=i)


def data_keyframe(data):
    for i in range(-1, max(number, len(data['camera_3d_center']))):
        print('Keyframing: ', i)

        # camera location based on distance and angle
        camera.location[0] = data['camera_3d_center'][i][0]
        camera.location[1] = data['camera_3d_center'][i][1]
        camera.location[2] = data['camera_3d_center'][i][2]
        # print(camera.location)
        # camera target 1 mm vertically and horizontally
        empty.location[0] = data['target'][i][0]
        empty.location[1] = -1
        empty.location[2] = data['target'][i][2]
        eye.rotation_euler[0] = math.radians(90 + data['gaze_angle_el'][i])
        eye.rotation_euler[2] = math.radians(data['gaze_angle_az'][i])
        # print(eye.rotation_euler[0])
        # print(eye.rotation_euler[2])
        # pupil size from 1mm to 3 mm
        bpy.data.meshes['Roundcube.000'].shape_keys.key_blocks["Pupil contract"].value = (data['pupil'][i] - 1) / 3

        # iris and sclera rotation (Any value from 0 , 360)
        iris.rotation_euler[2] = data['iris_rot'][i][2]
        sclera.rotation_euler[2] = data['sclera'][i][2]

        # Making sclera dark
        sclera_mat.node_tree.nodes["sclera_dark"].inputs[0].default_value = data['sclera_dark'][i]
        # sclera_mat.node_tree.nodes["sclera_dark"].inputs[0].default_value =random.uniform(0,0.8)

        cam.lens = data['cam_focal_length'][i]
        cam.sensor_width = data['cam_sensor_size'][i]

        # select with or without glasses.
        gl = data['glasses'][i]
        if gl:
            glass.hide_render = False
            glass.location[0] = random.uniform(-1.97501, -2.22824)
            glass.location[1] = random.uniform(-2.00167, -1.7237)
            glass.location[2] = random.uniform(-0.531074, -0.774396)
            glass.rotation_euler[0] = random.uniform(math.radians(88), math.radians(92))
            glass.rotation_euler[1] = random.uniform(math.radians(-2), math.radians(2))
            glass.rotation_euler[2] = random.uniform(math.radians(-2), math.radians(2))
            glass.keyframe_insert(data_path="location", frame=i)
            glass.keyframe_insert(data_path="rotation_euler", frame=i)
        else:
            glass.hide_render = True

        # Selecting random eye lid closure value from 0 to 1
        lid = data['eye_lid'][i] / 100

        bpy.data.meshes['eye_lid'].shape_keys.key_blocks["Key 1"].value = lid
        bpy.data.meshes['l'].shape_keys.key_blocks["Key 1"].value = lid
        bpy.data.meshes['u'].shape_keys.key_blocks["Key 1"].value = lid

        # Eye lid closure keyframing
        bpy.data.meshes['eye_lid'].shape_keys.key_blocks["Key 1"].keyframe_insert(data_path="value", frame=i)
        bpy.data.meshes['l'].shape_keys.key_blocks["Key 1"].keyframe_insert(data_path="value", frame=i)
        bpy.data.meshes['u'].shape_keys.key_blocks["Key 1"].keyframe_insert(data_path="value", frame=i)

        # blend eye lid
        if lid < 0.2:
            lid = lid
        else:
            lid = lid * 1.5
        skin.node_tree.nodes["blink"].inputs[0].default_value = lid

        # keyframe blend eye lid
        skin.node_tree.nodes["blink"].inputs[0].keyframe_insert(data_path="default_value", frame=i)

        # Select cornea
        c = cor[i]
        # c=data['cornea'][i]
        if c == '0':
            c1.hide_render = False
            c2.hide_render = True
            c3.hide_render = True

        if c == '1':
            c2.hide_render = False
            c3.hide_render = True
            c1.hide_render = True
        if c == '2':
            c3.hide_render = False
            c2.hide_render = True
            c1.hide_render = True

        camera.rotation_euler[0] = 0
        camera.rotation_euler[1] = 0
        camera.rotation_euler[2] = 0
        camera.keyframe_insert(data_path="rotation_euler", frame=i)
        # Camera location keyframing
        camera.keyframe_insert(data_path="location", frame=i)

        # Camera target keyframing
        empty.keyframe_insert(data_path="location", frame=i)

        # Iris and sclera key framing
        eye.keyframe_insert(data_path="rotation_euler", frame=i)

        # Iris and sclera key framing
        iris.keyframe_insert(data_path="rotation_euler", frame=i)
        sclera.keyframe_insert(data_path="rotation_euler", frame=i)

        # Glass keyframe
        glass.keyframe_insert(data_path="hide_render", frame=i)

        # keyframing pupil size
        bpy.data.meshes['Roundcube.000'].shape_keys.key_blocks["Pupil contract"].keyframe_insert(data_path="value",
                                                                                                 frame=i)

        # keyframing sclera color
        sclera_mat.node_tree.nodes["sclera_dark"].inputs[0].keyframe_insert(data_path="default_value", frame=i)

        ##Cornea key framing
        c1.keyframe_insert(data_path="hide_render", frame=i)
        c2.keyframe_insert(data_path="hide_render", frame=i)
        c3.keyframe_insert(data_path="hide_render", frame=i)


##Renders sequential data
def seq_render():
    import bpy
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cuda_devices, opencl_devices = cycles_preferences.get_devices()
    device_type = "OPTIX"
    print("USING THE FOLLOWING GPUS:")
    devices=[]
    cycles_preferences.compute_device_type = device_type
    for x in range(len(cycles_preferences.devices)):
        if cycles_preferences.devices[x] in cuda_devices:
            devices.append(cycles_preferences.devices[x])
    for x in range(len(devices)):
        print(devices[x].name)
        devices[0].use = True #changed @@yush
    # giw pickle file name
    p = args.data_source_path

    try:
        camera.constraints["Track To"].target = None
    except:
        pass

    # Making a directory with specified output file name
    try:
        os.makedirs(os.path.join(output_folder, model, output, p[:-2]))
    except OSError:
        print("Creation of the directory %s failed" % os.path.join(output_folder, model, output, p[:-2]))
    else:
        print("Successfully created the directory %s" % os.path.join(output_folder, model, output, p[:-2]))
    try:
        os.makedirs(os.path.join(os.getcwd(),'static_model', model, output, p[:-2]))
    except OSError:
        print("Creation of the directory %s failed" % os.path.join(os.getcwd(),'static_model', model, output, p[:-2]))
    else:
        print("Successfully created the directory %s" % os.path.join(os.getcwd(),'static_model', model, output, p[:-2]))
    ## Load the data file
    file = pickle.load(open(os.path.join(inp, p), "rb"), encoding="latin1")
    # Seperate lists for head vectors x,y,z and eye vectors x,y,z and pupil

    # The pickle files need the first 2 index of the eye and head vectors to be 0,0,1. These images will not be rendered but still it needs tobe there.
    # Saving the vectors for transition type rendering
    # Note you select transition when you plan to render eye movements with transition
    # Data in the pickle file has been labeled as 1,2,3,4 for fixation, smooth pursuit, blink and saccade respectively as original GIW dataset
    if args.seq_type == 'transition':
        h_x = []
        h_y = []
        h_z = []
        e_x = []
        e_y = []
        e_z = []
        pupil = []
        for id in ['13', '31', '14', '41', '34', '43']:
            for i in range(len(file[id][6])):
                for x in range(0, 600, 5):
                    h_x.append(file[id][0][i][x])
                    h_y.append(file[id][1][i][x])
                    h_z.append(file[id][2][i][x])
                    e_x.append(file[id][3][i][x])
                    e_y.append(file[id][4][i][x])
                    e_z.append(file[id][5][i][x])
                    pupil.append(file[id][7][i][x])

    # Saving the vectors for all type rendering
    # This does not care about labeled dataset but renders all the data in the pickle file 
    # We are using this setup currently for our pipeline 
    if args.seq_type == 'all':
        h_x = []
        h_y = []
        h_z = []
        e_x = []
        e_y = []
        e_z = []
        pupil = []
        for x in range(len(file['h_x'])):
            h_x.append(file['h_x'][x])
            h_y.append(file['h_y'][x])
            h_z.append(file['h_z'][x])
            e_x.append(file['e_x'][x])
            e_y.append(file['e_y'][x])
            e_z.append(file['e_z'][x])
            pupil.append(file['pupil'][x])

    h_x = [float(x) for x in h_x]
    h_y = [float(x) for x in h_y]
    h_z = [float(x) for x in h_z]
    e_x = [float(x) for x in e_x]
    e_y = [float(x) for x in e_y]
    e_z = [float(x) for x in e_z]

    # Calculating the pupil size in blender format
    pupil = [float(((x / 2) * (0.8 / 3)) - (0.5 / 3)) for x in pupil]

    if args.env != '0':
        try:
            env_obj = bpy.data.objects['Empty.001']
        except:
            bpy.ops.object.add(type='EMPTY', location=(0, 0, 0))
            env_obj = bpy.data.objects['Empty.001']
            world.node_tree.nodes["Texture Coordinate"].object = env_obj

    eye.rotation_mode = 'XYZ'

    s = bpy.context.scene

    # Image resolution
    s.render.resolution_x = 640
    s.render.resolution_y = 480

    # Rendering tile size
    s.render.tile_x = 640
    s.render.tile_y = 480

    s.cycles.device = 'GPU'
    s.render.image_settings.file_format = 'TIFF'

    # Loading all textures
    # Loading all textures
    # Iris texture
    iris_mat = bpy.data.materials['Iris.000']
    # Sclera texture
    sclera_mat = bpy.data.materials['Sclera material.000']


    iris_mat = bpy.data.materials['Iris.000']
    sclera_mat = bpy.data.materials['Sclera material.000']
    ir  =os.path.join(os.getcwd(),'Textures_eye','ir-textures/')
    scl =os.path.join(os.getcwd(),'Textures_eye','sclera/')
    env_path = os.path.join(os.getcwd(),'environmental_textures/')
    sclera_mat.node_tree.nodes["op"].image = bpy.data.images.load(filepath=os.path.join(os.getcwd(),'Textures_eye','opacity.png'))
    sclera_mat.node_tree.nodes["sclera"].image = bpy.data.images.load(filepath=os.path.join(os.getcwd(),'Textures_eye','Sclera color.png'))

    skin.node_tree.nodes["c1"].image =bpy.data.images.load(filepath= os.path.join(os.getcwd(),'static_model',model,'Textures','c.jpg'))
    skin.node_tree.nodes["c2"].image = bpy.data.images.load(filepath=os.path.join(os.getcwd(),'static_model',model,'Textures','c1.jpg'))
    
    skin.node_tree.nodes["g1"].image =bpy.data.images.load(filepath=os.path.join(os.getcwd(),'static_model',model,'Textures','g.jpg'))
    
    skin.node_tree.nodes["s1"].image =bpy.data.images.load(filepath=os.path.join(os.getcwd(),'static_model',model,'Textures','s.jpg'))
    
    skin.node_tree.nodes["n"].image =bpy.data.images.load(filepath=os.path.join(os.getcwd(),'static_model',model,'Textures','n.jpg'))

    # Initially Eye should be looking forward
    eye.rotation_euler[0] = math.radians(90)
    eye.rotation_euler[1] = math.radians(0)
    eye.rotation_euler[2] = math.radians(0)
    eye.keyframe_insert(data_path="rotation_euler", frame=0)

    if args.env != '0':
        env_obj.rotation_mode = 'XYZ'
        env_obj.rotation_euler[0] = math.radians(0)
        env_obj.rotation_euler[1] = math.radians(0)
        env_obj.rotation_euler[2] = math.radians(0)
        env_obj.keyframe_insert(data_path="rotation_euler", frame=0)

    # KEYFRAMING EYE AND HEAD VECTORS
    for x in range(1, len(h_x)):
        if x % 1000 == 0:
            print(x)

        h1 = mathutils.Vector((h_x[x], h_y[x], h_z[x]))
        h2 = mathutils.Vector((h_x[x - 1], h_y[x - 1], h_z[x - 1]))
        h_rot = h1.rotation_difference(h2).to_euler()

        # Finding the angle between the vectors
        eye1 = mathutils.Vector((e_x[x], e_y[x], e_z[x]))
        eye2 = mathutils.Vector((e_x[x - 1], e_y[x - 1], e_z[x - 1]))
        l_rot = eye1.rotation_difference(eye2).to_euler()

        if args.env != '0':
            #     bpy.ops.object.add(type='EMPTY', location=(0, 0, 0))

            env_obj.rotation_euler[0] = env_obj.rotation_euler[0] - h_rot.x
            env_obj.rotation_euler[1] = env_obj.rotation_euler[1] + h_rot.z
            env_obj.rotation_euler[2] = env_obj.rotation_euler[2] + h_rot.y

        # Setting the rotation
        eye.rotation_euler[0] = eye.rotation_euler[0] - l_rot.x
        eye.rotation_euler[1] = eye.rotation_euler[1] + l_rot.z
        eye.rotation_euler[2] = eye.rotation_euler[2] - l_rot.y


        # Calculating eye lid values based on verticle eye gaze
        # value = (math.degrees(eye.rotation_euler[0]) - 75) / 50

        # Use this for shape keys
        # bpy.data.meshes['eye_lid'].shape_keys.key_blocks["Key 1"].value = value
        # bpy.data.meshes['eye_lid'].shape_keys.key_blocks["Key 1"].keyframe_insert(data_path="value",frame=x)
        # bpy.data.meshes['l'].shape_keys.key_blocks["Key 1"].value = value
        # bpy.data.meshes['l'].shape_keys.key_blocks["Key 1"].keyframe_insert(data_path="value",frame=x)
        # bpy.data.meshes['u'].shape_keys.key_blocks["Key 1"].value = value
        # bpy.data.meshes['u'].shape_keys.key_blocks["Key 1"].keyframe_insert(data_path="value",frame=x)

        # Use this for bone lids
        # if eye_vert_degrees > 115:
        #     ratio = (eye_vert_degrees - 115) / (128 - 115)
        #     x_turn_amt = ratio * 0.154
        #     bpy.data.objects['Armature'].pose.bones['lower_blinker'].rotation_quaternion[1] += x_turn_amt
        #     bpy.data.objects['Armature'].pose.bones['lower_blinker'].keyframe_insert(data_path="rotation_quaternion",frame=i, index=1)
        #     # upper retract by -.122 from 79 degrees to 65 degrees
        #
        # if eye_vert_degrees < 79:
        #     ratio = (eye_vert_degrees - 79) / -(79 - 65)
        #     x_turn_amt = ratio * -0.122
        #     bpy.data.objects['Armature'].pose.bones['upper_blinker'].rotation_quaternion[1] += x_turn_amt
        #     bpy.data.objects['Armature'].pose.bones['upper_blinker'].keyframe_insert(data_path="rotation_quaternion",frame=i, index=1)
        #

        bpy.data.objects['Armature'].pose.bones['upper_blinker'].rotation_euler[0] = math.radians(random.uniform(0,5))
        bpy.data.objects['Armature'].pose.bones['upper_blinker'].keyframe_insert(data_path="rotation_euler", frame=x)
        bpy.data.objects['Armature'].pose.bones['lower_blinker'].rotation_euler[0] = math.radians(random.uniform(0,5))
        bpy.data.objects['Armature'].pose.bones['lower_blinker'].keyframe_insert(data_path="rotation_euler", frame=x)

        bpy.data.meshes['Roundcube.000'].shape_keys.key_blocks["Pupil contract"].value = pupil[x]
        bpy.data.meshes['Roundcube.000'].shape_keys.key_blocks["Pupil contract"].keyframe_insert(data_path="value",frame=x)

        # if value < 0.15:
        #     v = value
        # else:
        #     v = value * 1.2
        # skin.node_tree.nodes["blink"].inputs[0].default_value = v
        # skin.node_tree.nodes["blink"].inputs[0].keyframe_insert(data_path="default_value", frame=x)
        # head.keyframe_insert(data_path="rotation_euler", frame=x)
        eye.keyframe_insert(data_path="rotation_euler", frame=x)
        glass.hide_render = True

        # Glass keyframe
        glass.keyframe_insert(data_path="hide_render", frame=x)

        c1.hide_render = False
        c2.hide_render = True
        c3.hide_render = True
        sphere.hide_render = True

        ##Cornea key framing
        c1.keyframe_insert(data_path="hide_render", frame=x)
        c2.keyframe_insert(data_path="hide_render", frame=x)
        c3.keyframe_insert(data_path="hide_render", frame=x)

        camera.location[0] = args.camera_distance[0]
        camera.location[1] = args.camera_distance[1]
        camera.location[2] = args.camera_distance[2]

        camera.rotation_euler[0] = math.radians(args.camera_elevation[0])
        camera.rotation_euler[1] = math.radians(args.camera_roll[0])
        camera.rotation_euler[2] = math.radians(args.camera_azimuthal[0])

        head.rotation_euler[0] = math.radians(args.head_elevation[0])
        head.rotation_euler[1] = math.radians(args.head_roll[0])
        head.rotation_euler[2] = math.radians(args.head_azimuthal[0])
        head.keyframe_insert(data_path="rotation_euler", frame=x)

        # camera target 1 mm vertically and horizontally

        # sclera_mat.node_tree.nodes["sclera_dark"].inputs[0].default_value = data['sclera_dark'][i]
        sclera_mat.node_tree.nodes["sclera_dark"].inputs[0].default_value = 0.65
        # keyframing sclera color
        sclera_mat.node_tree.nodes["sclera_dark"].inputs[0].keyframe_insert(data_path="default_value", frame=x)
        # Camera location keyframing
        camera.keyframe_insert(data_path="location", frame=x)
        camera.keyframe_insert(data_path="rotation_euler", frame=x)

        # Camera target keyframing
        empty.keyframe_insert(data_path="location", frame=x)

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(os.getcwd(),'static_model', model, output, p[:-2], 'seq.blend'))
    iris_mat.node_tree.nodes["iris"].image = bpy.data.images.load(
        filepath=os.path.join(ir, args.iris_textures + '.png'))
    iris_mat.node_tree.nodes["iris"].image = bpy.data.images.load(
        filepath=os.path.join(ir, args.iris_textures + '.png'))
    if args.env in [str(x) for x in range(1, 26)]:
        world.node_tree.nodes["Environment Texture"].image = bpy.data.images.load(
            filepath=os.path.join(env_path, args.env + '.hdr'))
        l1.data.energy = 0
        l2.data.energy = 0
        # bpy.data.materials["Eye wet material.001"].node_tree.nodes["BSDF Reflectivo"].inputs[1].default_value = 0
    for i in range(1, len(h_x)):

        s.frame_current = i
        filename = os.path.join(output_folder, model, output, p[:-2], 'synthetic',
                                str(s.frame_current).zfill(4) + ".tif")
        if os.path.isfile(filename):
            print("skipped ", filename)
            continue
        else:
            print(i)

        s.render.filepath = (filename)

        bpy.ops.render.render(  # {'dict': "override"},
            # 'INVOKE_DEFAULT',
            False,  # undo support
            animation=False,
            write_still=True
        )

    # skin black
    skin.node_tree.links.remove(skin.node_tree.nodes["Material Output"].inputs['Surface'].links[0])
    skin.node_tree.links.remove(skin.node_tree.nodes["Material Output"].inputs['Displacement'].links[0])

    # eye plica black
    wet.node_tree.links.remove(wet.node_tree.nodes["Material Output"].inputs['Surface'].links[0])
    wet.node_tree.links.remove(wet.node_tree.nodes["Material Output"].inputs['Displacement'].links[0])

    # iris to green
    iris_mat.node_tree.links.new(iris_mat.node_tree.nodes["Emission"].outputs['Emission'],
                                 iris_mat.node_tree.nodes["Material Output"].inputs[0])
    # iris_mat.node_tree.links.remove(iris_mat.node_tree.nodes["Material Output"].inputs['Displacement'].links[0])

    retina = bpy.data.materials['Sclera material.001']
    # retina to red
    retina.node_tree.links.new(retina.node_tree.nodes["Emission"].outputs['Emission'],
                               retina.node_tree.nodes["Material Output"].inputs[0])

    glass.hide_render = True
    sclera.hide_render = True
    bpy.data.objects["EyeWet.002"].hide_render = True
    bpy.data.objects["lower"].hide_render = True
    bpy.data.objects["upper"].hide_render = True
    bpy.data.objects["sclera_mask"].hide_render = False
    l1.hide_render = True
    l2.hide_render = True

    # RENDERING MASK IMAGES
    for i in range(1, len(h_x)):
        glass.hide_render = True
        glass.keyframe_insert(data_path="hide_render", frame=i)

        frame = i
        s.frame_current = frame
        filename = os.path.join(output_folder, model, output, p[:-2], 'maskwith_skin',
                                str(s.frame_current).zfill(4) + ".tif")
        if os.path.isfile(filename):
            print("skipped ", filename)
            continue
        else:
            print(frame)

        s.render.filepath = filename

        bpy.ops.render.render(  # {'dict': "override"},
            # 'INVOKE_DEFAULT',
            False,  # undo support
            animation=False,
            write_still=True)

    bpy.data.objects["head"].hide_render = True
    bpy.data.objects["Eye_Plica"].hide_render = True
    bpy.data.objects["Sunglasses"].hide_render = True

    for i in range(1, len(h_x)):
        try:
            s.node_tree.links.new(s.node_tree.nodes["Render Layers"].outputs['Image'],
                                  s.node_tree.nodes["Composite"].inputs[0])
        except:
            print('No node')
        frame = i
        s.frame_current = frame
        filename = os.path.join(output_folder, model, output, p[:-2], 'maskwithout_skin',
                                str(s.frame_current).zfill(4) + ".tif")
        if os.path.isfile(os.path.join(output_folder, model, output, p[:-2], 'maskwithout_skin',
                                       str(s.frame_current).zfill(4) + ".tif")):
            print("skipped ", filename)
            continue
        else:
            print(frame)
            s.render.filepath = filename

            bpy.ops.render.render(  # {'dict': "override"},
                # 'INVOKE_DEFAULT',
                False,  # undo support
                animation=False,
                write_still=True)

        # Rendering depth map but on hold for now
        # s.node_tree.links.new(s.node_tree.nodes["ColorRamp"].outputs['Image'], s.node_tree.nodes["Composite"].inputs[0])
        # s.render.filepath = os.path.join(output_folder,model,output,p[:-2],'depth' ,str(s.frame_current).zfill(4) + ".tif")
        # if os.path.isfile(os.path.join(output_folder,model,output,p[:-2],'maskwithout_skin' ,str(s.frame_current).zfill(4) + ".tif")):
        #   print("skipped ",os.path.isfile(os.path.join(output_folder,model,output,p[:-2],'maskwithout_skin' ,str(s.frame_current).zfill(4) + ".tif")))
        # else:
        #   bpy.ops.render.render(  # {'dict': "override"},
        #          # 'INVOKE_DEFAULT',
        #         False,  # undo support
        #        animation=False,
        #       write_still=True)
    for x in range(1, len(h_x)):
        print(x)
        data(x)
    for x in gt.keys():
        gt[x] = gt[x][1:]
    pickle.dump(gt, open(os.path.join(output_folder, model, output, p[:-2], model + '-pupil.p'), "wb"))
    print("saved")

##Generating 3D data
def data(i):
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            override = bpy.context.copy()
            override['space_data'] = area.spaces.active
            override['region'] = area.regions[-1]
            override['area'] = area
            # space_data = area.spaces.active
    # camera data
    bpy.context.scene.frame_set(i)
    camera = bpy.data.objects['Camera']
    camera.select_set(True)
    bpy.ops.view3d.snap_cursor_to_selected(override)
    loc = bpy.context.scene.cursor.location

    # camera matrix
    gt['cam_focal_length'].append(cam.lens)
    gt['cam_sensor_size'].append(cam.sensor_width)
    gt['camera_3d_center'].append(np.array(loc))
    gt['camera_distance'].append(math.sqrt(math.pow(loc[0], 2) + math.pow(loc[2], 2) + math.pow(-loc[1] - 1, 2)))
    camera.select_set(False)
    bpy.ops.object.select_all(action='DESELECT')

    # iris data
    iris = bpy.data.objects['EyeIris']
    iris.select_set(True)
    bpy.ops.view3d.snap_cursor_to_selected(override)
    iris_loc = bpy.context.scene.cursor.location
    gt['iris_loc'].append([iris_loc[0], iris_loc[1], iris_loc[2]])
    gt['iris_rot'].append([iris.rotation_euler[0], iris.rotation_euler[1], iris.rotation_euler[2]])
    gt['iris_texture'].append(a[i])
    iris.select_set(False)

    # light data
    l1.select_set(True)
    bpy.ops.view3d.snap_cursor_to_selected(override)
    l1_loc = bpy.context.scene.cursor.location
    gt['light1_loc'].append([l1_loc[0], l1_loc[1], l1_loc[2]])
    l1.select_set(False)
    bpy.ops.object.select_all(action='DESELECT')

    l2.select_set(True)
    bpy.ops.view3d.snap_cursor_to_selected(override)
    l2_loc = bpy.context.scene.cursor.location
    gt['light2_loc'].append([l2_loc[0], l2_loc[1], l2_loc[2]])
    l2.select_set(False)
    bpy.ops.object.select_all(action='DESELECT')

    # target
    t = bpy.data.objects['Empty']
    t.select_set(True)
    bpy.ops.view3d.snap_cursor_to_selected(override)
    target = bpy.context.scene.cursor.location
    gt['target'].append(np.array(target))

    # camera angles
    c = np.array(target) - gt['camera_3d_center'][-1]
    c = c / np.sqrt(np.sum(c ** 2))
    gt['cam_az'].append(math.degrees(math.atan2(c[1], c[0])) + 90)
    gt['cam_el'].append(math.degrees(math.atan2(c[2], np.sqrt(c[0] ** 2 + c[1] ** 2))))

    # light1 angles
    c = np.array(target) - gt['light1_loc'][-1]
    c = c / np.sqrt(np.sum(c ** 2))
    gt['light1_az'].append(math.degrees(math.atan2(c[1], c[0])) + 90)
    gt['light1_el'].append(math.degrees(math.atan2(c[2], np.sqrt(c[0] ** 2 + c[1] ** 2))))

    # light2 angles
    c = np.array(target) - gt['light2_loc'][-1]
    c = c / np.sqrt(np.sum(c ** 2))
    gt['light2_az'].append(math.degrees(math.atan2(c[1], c[0])) + 90)
    gt['light2_el'].append(math.degrees(math.atan2(c[2], np.sqrt(c[0] ** 2 + c[1] ** 2))))

    t.select_set(False)
    bpy.ops.object.select_all(action='DESELECT')

    # Sclera data
    sclera = bpy.data.objects['Sclera']
    sclera.select_set(True)
    gt['sclera_dark'].append(sclera_mat.node_tree.nodes["sclera_dark"].inputs[0].default_value)
    gt['sclera'].append([sclera.rotation_euler[0], sclera.rotation_euler[1], sclera.rotation_euler[2]])
    gt['sclera_texture'].append(b[i])
    sclera.select_set(False)

    bpy.ops.object.select_all(action='DESELECT')

    eye = bpy.data.objects['Eye.Wetness']
    eye.select_set(True)
    bpy.ops.view3d.snap_cursor_to_selected(override)
    eye_loc = bpy.context.scene.cursor.location

    gt['eye_loc'].append([eye_loc[0], eye_loc[1], eye_loc[2]])

    gt['gaze_angle_az'].append(math.degrees(eye.rotation_euler[2]))

    gt['gaze_angle_el'].append(math.degrees(eye.rotation_euler[0]) - 90)
    eye.select_set(False)
    bpy.ops.object.select_all(action='DESELECT')

    if bpy.data.objects["Sunglasses"].hide_render:
        gt['glasses'].append(0)
    else:
        gt['glasses'].append(1)

    pupil_s = bpy.data.meshes['Roundcube.000'].shape_keys.key_blocks["Pupil contract"].value
    gt['pupil'].append((pupil_s * 3) + 1)
    gt['eye_lid'].append(bpy.data.meshes['eye_lid'].shape_keys.key_blocks["Key 1"].value * 100)

    # bpy.context.area.ui_type = 'TEXT_EDITOR'


def angle_between_three_3D_points(camera_center, sphere_center, eyeball_points):
    v1 = np.array(eyeball_points) - np.array(sphere_center)  # v1=A-B
    v2 = np.array(camera_center) - np.array(sphere_center)  # v2=C-B
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    roll = []
    pitch = []
    yaw = []

    res = np.cross(v1, v2)
    res = res / np.linalg.norm(res)
    euler = mathutils.Euler((np.arcsin(res[0]), np.arcsin(res[1]), np.arcsin(res[2])))
    roll.append(0)
    yaw.append(euler[1])

    angle = 0
    if (v1[2]>=0 and v2[2]>=0):
        angle = np.arcsin(v1[2])-np.arcsin(v2[2])
    elif (v1[2]<=0 and v2[2]<=0):
        angle = np.arcsin(-v2[2])-np.arcsin(-v1[2])
    elif (v1[2]>0 and v2[2]<0):
        angle = np.arcsin(-v2[2])+np.arcsin(v1[2])
    elif (v1[2]<0 and v2[2]>0):
        angle = -np.arcsin(v2[2])-np.arcsin(-v1[2])
    
    pitch.append(angle)


    # return [np.array(roll) * 180 / np.pi, np.array(pitch) * 180 / np.pi, np.array(yaw) * 180 / np.pi]
    return [np.array(pitch) * 180 / np.pi, np.array(roll) * 180 / np.pi, np.array(yaw) * 180 / np.pi ]

def blink_render():
    import bpy
    #import pandas as pd

    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cuda_devices, opencl_devices = cycles_preferences.get_devices()
    device_type = "OPTIX"
    print("USING THE FOLLOWING GPUS:")
    devices=[]
    cycles_preferences.compute_device_type = device_type
    for x in range(len(cycles_preferences.devices)):
        if cycles_preferences.devices[x] in cuda_devices:
            devices.append(cycles_preferences.devices[x])
    for x in range(len(devices)):
        print(devices[x].name)
        devices[0].use = True #changed @@yush
    # giw pickle file name
    #p = args.data_source_path

    try:
        camera.constraints["Track To"].target = None
    except:
        pass

    # Making a directory with specified output file name
    try:
        os.makedirs(os.path.join(output_folder, str(args.index)))
    except OSError:
        print("Creation of the directory %s failed" % os.path.join(output_folder, str(args.index)))
    else:
        print("Successfully created the directory %s" % os.path.join(output_folder, str(args.index)))
    try:
        os.makedirs(os.path.join(os.getcwd(),'static_model', model, output, str(args.person_idx)+'_'+str(args.trial_idx)))
    except OSError:
        print("Creation of the directory %s failed" % os.path.join(os.getcwd(),'static_model', model, output, str(args.person_idx)+'_'+str(args.trial_idx)))
    else:
        print("Successfully created the directory %s" % os.path.join(os.getcwd(),'static_model', model, output, str(args.person_idx)+'_'+str(args.trial_idx)))
    
    #region old pickle file
    ## Load the data file
#     file = pickle.load(open(os.path.join(inp, p), "rb"), encoding="latin1")
#     # Seperate lists for head vectors x,y,z and eye vectors x,y,z and pupil

#     # The pickle files need the first 2 index of the eye and head vectors to be 0,0,1. These images will not be rendered but still it needs to be there.
#     # Saving the vectors for transition type rendering
#     if args.seq_type == 'transition':
#         h_x = []
#         h_y = []
#         h_z = []
#         e_x = []
#         e_y = []
#         e_z = []
#         pupil = []
#         for id in ['13', '31', '14', '41', '34', '43']:
#             for i in range(len(file[id][6])):
#                 for x in range(0, 600, 5):
#                     h_x.append(file[id][0][i][x])
#                     h_y.append(file[id][1][i][x])
#                     h_z.append(file[id][2][i][x])
#                     e_x.append(file[id][3][i][x])
#                     e_y.append(file[id][4][i][x])
#                     e_z.append(file[id][5][i][x])
#                     pupil.append(file[id][7][i][x])

#     # Saving the vectors for all type rendering
#     if args.seq_type == 'all':

#         e_x= []
#         e_y= []
#         e_z= []
#         u_x= []
#         u_y= []
#         u_z= []
#         l_x= []
#         l_y= []
#         l_z= []
#         pupil = []
#         for x in range(min(int(args.number),len(file['pupil_center_axis'][0]))):

#             e_x.append(file['pupil_center_axis'][0][x])
#             e_y.append(file['pupil_center_axis'][1][x])
#             e_z.append(file['pupil_center_axis'][2][x])
#             u_x.append(file['top_center_axis'][0][x])
#             u_y.append(file['top_center_axis'][1][x])
#             u_z.append(file['top_center_axis'][2][x])
#             l_x.append(file['bottom_center_axis'][0][x])
#             l_y.append(file['bottom_center_axis'][1][x])
#             l_z.append(file['bottom_center_axis'][2][x])
#             pupil.append(file['pupil'][x])

#     e_x = [float(x) for x in e_x]
#    # print ('here',e_x)
#     e_y = [float(x) for x in e_y]
#     e_z = [float(x) for x in e_z]
#     u_x = [float(x) for x in u_x]
#     u_y = [float(x) for x in u_y]
#     u_z = [float(x) for x in u_z]
#     l_x = [float(x) for x in l_x]
#     l_y = [float(x) for x in l_y]
#     l_z = [float(x) for x in l_z]

#     # Calculating the pupil size in blender format
#     pupil = [float(((x / 2) * (0.8 / 3)) - (0.5 / 3)) for x in pupil]
    #endregion 

    # file = pickle.load(open(os.path.join(inp, p), "rb"), encoding="latin1")

    # Pupil player export data
    export_data = pickle.load(open(args.picklefile +str(args.person_idx)+'/'+str(args.trial_idx)+'/export_data.pickle', 'rb'), encoding='latin1')
    
    pupil = export_data['pupil']
    pupil = [float(((x / 2) * (0.8 / 3)) - (0.5 / 3)) for x in pupil]


    if args.env != '0':
        try:
            env_obj = bpy.data.objects['Empty.001']
        except:
            bpy.ops.object.add(type='EMPTY', location=(0, 0, 0))
            env_obj = bpy.data.objects['Empty.001']
            world.node_tree.nodes["Texture Coordinate"].object = env_obj

    eye.rotation_mode = 'XYZ'

    s = bpy.context.scene

    # Image resolution
    s.render.resolution_x = 640
    s.render.resolution_y = 480

    # Rendering tile size
    s.render.tile_x = 640
    s.render.tile_y = 480

    s.cycles.device = 'GPU'
    s.render.image_settings.file_format = 'PNG'

    # Loading all textures
    # Iris texture
    iris_mat = bpy.data.materials['Iris.000']
    # Sclera texture
    sclera_mat = bpy.data.materials['Sclera material.000']

    ir = os.path.join(os.getcwd(),'Textures_eye','ir-textures')
    scl = os.path.join(os.getcwd(),'Textures_eye','sclera')
    env_path = os.path.join(os.getcwd(),'environmental_textures')
    sclera_mat.node_tree.nodes["op"].image = bpy.data.images.load(filepath=os.path.join(os.getcwd(),'Textures_eye','opacity.png'))
    sclera_mat.node_tree.nodes["sclera"].image = bpy.data.images.load(filepath=os.path.join(os.getcwd(),'Textures_eye','Sclera color.png'))
    # Skin Textures
    skin.node_tree.nodes["c1"].image =bpy.data.images.load(filepath= os.path.join(os.getcwd(),'static_model',model,'Textures','c.jpg'))
    skin.node_tree.nodes["c2"].image = bpy.data.images.load(filepath=os.path.join(os.getcwd(),'static_model',model,'Textures/c1.jpg'))
    skin.node_tree.nodes["g1"].image =bpy.data.images.load(filepath=os.path.join(os.getcwd(),'static_model',model,'Textures/g.jpg'))
    skin.node_tree.nodes["s1"].image =bpy.data.images.load(filepath=os.path.join(os.getcwd(),'static_model',model,'Textures/s.jpg'))
    skin.node_tree.nodes["n"].image =bpy.data.images.load(filepath=os.path.join(os.getcwd(),'static_model',model,'Textures/n.jpg'))

    # Initially Eye should be looking forward
    #eye_org=[86.2288,0,-6.11]
    #upper_org=[38.3,0,-10.4]
    #lower_org=[-6.39246,0,-1.17597]
    #eye.keyframe_insert(data_path="rotation_euler", frame=0)

    # sets the eye reflection to be slightly bigger
    bpy.data.objects["EyeWet.002"].scale = bpy.data.objects["EyeWet.002"].scale * 1.02

    # sets the head location androtation
    cam_loc = [args.camera_distance[0], args.camera_distance[1], args.camera_distance[2]]
    eye_loc = [-cam_loc[0], -cam_loc[1], -cam_loc[2]]
    head.location = eye_loc
    eye.location = eye_loc
    
    empty.location[0] = -args.camera_distance[0]
    empty.location[1] = -args.camera_distance[1] - 1.2
    empty.location[2] = -args.camera_distance[2]


    head.rotation_euler[0] = math.radians(float(args.head_elevation[0]))
    head.rotation_euler[1] = math.radians(float(args.head_roll[0]))
    head.rotation_euler[2] = math.radians(float(args.head_azimuthal[0]))
    
    # head.rotation_euler[0] = 0
    # head.rotation_euler[1] = 0
    # head.rotation_euler[2] = 0

    # head.keyframe_insert(data_path="location", frame=x)
    # head.keyframe_insert(data_path="rotation_euler", frame=x)

    # sets up the camera location and rotation
    camera.location = [0,0,0]
    # camera.rotation_euler[0] = math.radians(args.camera_elevation[0])
    # camera.rotation_euler[1] = math.radians(args.camera_roll[0])
    # camera.rotation_euler[2] = math.radians(args.camera_azimuthal[0])
    camera.rotation_euler[0] = math.radians(90)
    camera.rotation_euler[1] = 0
    camera.rotation_euler[2] = 0
    
    # camera.keyframe_insert(data_path="location", frame=x)
    # camera.keyframe_insert(data_path="rotation_euler", frame=x)

    # Setting initial eye and lid angles
    lower_blinker = head.pose.bones['lower_blinker']
    upper_blinker = head.pose.bones['upper_blinker']

    bpy.context.view_layer.update()

    global_mat = head.matrix_world
    # print('Head matrix:')
    # print(head.matrix_world)

    vec = global_mat @ upper_blinker.tail
    upper_org = angle_between_three_3D_points([0,0,0], eye_loc, [vec[0], vec[1], vec[2]])
    upper_org[0] += math.degrees(upper_blinker.rotation_euler[0])

    vec = global_mat @ lower_blinker.tail
    lower_org = angle_between_three_3D_points([0,0,0], eye_loc, [vec[0], vec[1], vec[2]])
    lower_org[0] -= math.degrees(lower_blinker.rotation_euler[0])

    eye_org=angle_between_three_3D_points(cam_loc, [0, 0, 0], [0,0,1])

    eye.rotation_euler[0] = math.radians(eye_org[0])
    eye.rotation_euler[1] = math.radians(eye_org[1])
    eye.rotation_euler[2] = math.radians(eye_org[2])

    eye.keyframe_insert(data_path="rotation_euler", frame=0)
    
    print('upper: ',upper_org)
    print('lower: ', lower_org,eye_org)

    upper_blinker.rotation_euler[0] = math.radians(upper_org[0])
    lower_blinker.rotation_euler[0] = math.radians(-lower_org[0])
    upper_blinker.keyframe_insert(data_path="rotation_euler", frame=0)
    lower_blinker.keyframe_insert(data_path="rotation_euler", frame=0)

    
    if args.env != '0':
        env_obj.rotation_mode = 'XYZ'
        env_obj.rotation_euler[0] = math.radians(0)
        env_obj.rotation_euler[1] = math.radians(0)
        env_obj.rotation_euler[2] = math.radians(0)
        env_obj.keyframe_insert(data_path="rotation_euler", frame=0)



    frame_cap = len(export_data['frame'])

    # KEYFRAMING EYE AND HEAD VECTORS
    for x in range(1, frame_cap):
        if x % 10 == 0:
            print(x/frame_cap)

        # if args.env != '0':
        #     #     bpy.ops.object.add(type='EMPTY', location=(0, 0, 0))

        #     env_obj.rotation_euler[0] = env_obj.rotation_euler[0] - h_rot.x
        #     env_obj.rotation_euler[1] = env_obj.rotation_euler[1] + h_rot.z
        #     env_obj.rotation_euler[2] = env_obj.rotation_euler[2] + h_rot.y

        eye_loc = export_data['eye_loc'][x]
        eye.location = eye_loc
        head.location = eye_loc

        eye_norm = export_data['pupil_normal'][x]
        eye_angle = angle_between_three_3D_points(eye_norm,[0,0,0],[0,0,1])

        print('Eyeball normal: '+str(eye_norm))

        eye.rotation_euler[0] = math.radians(eye_angle[0])
        eye.rotation_euler[1] = math.radians(eye_angle[1])
        eye.rotation_euler[2] = math.radians(eye_angle[2])

        eye.keyframe_insert(data_path="rotation_euler", frame=x)
        eye.keyframe_insert(data_path="location", frame=x)

        # upper_blink= math.radians(upper_org[0] - u_z[x])
        # lower_blink= math.radians(-lower_org[0] + l_z[x])

        #print ('Blink',math.degrees(upper_blink),math.degrees(lower_blink),(upper_org[0] - u_z[x]),lower_org[0] - l_z[x])

        # Setting the rotation

        upper_blinker.rotation_euler[0] = 0 #upper_blink
        upper_blinker.rotation_euler[1] = 0
        upper_blinker.rotation_euler[2] = 0
        upper_blinker.keyframe_insert(data_path="rotation_euler", frame=x)
        lower_blinker.rotation_euler[0] = 0 #lower_blink
        lower_blinker.rotation_euler[1] = 0 
        lower_blinker.rotation_euler[2] = 0
        lower_blinker.keyframe_insert(data_path="rotation_euler", frame=x)

        bpy.data.meshes['Roundcube.000'].shape_keys.key_blocks["Pupil contract"].value = pupil[x] #0.4
        bpy.data.meshes['Roundcube.000'].shape_keys.key_blocks["Pupil contract"].keyframe_insert(data_path="value",frame=x)


        glass.hide_render = True

        # Glass keyframe
        glass.keyframe_insert(data_path="hide_render", frame=x)

        c1.hide_render = False
        c2.hide_render = True
        c3.hide_render = True
        sphere.hide_render = True

        ##Cornea key framing
        c1.keyframe_insert(data_path="hide_render", frame=x)
        c2.keyframe_insert(data_path="hide_render", frame=x)
        c3.keyframe_insert(data_path="hide_render", frame=x)

        # camera.location[0] = args.camera_distance[0]
        # camera.location[1] = args.camera_distance[1]
        # camera.location[2] = args.camera_distance[2]



        # camera target 1 mm vertically and horizontally

        # sclera_mat.node_tree.nodes["sclera_dark"].inputs[0].default_value = data['sclera_dark'][i]
        sclera_mat.node_tree.nodes["sclera_dark"].inputs[0].default_value = 0.65
        # keyframing sclera color
        sclera_mat.node_tree.nodes["sclera_dark"].inputs[0].keyframe_insert(data_path="default_value", frame=x)

        # Camera target keyframing
        empty.keyframe_insert(data_path="location", frame=x)

    iris_mat.node_tree.nodes["iris"].image = bpy.data.images.load(
        filepath=os.path.join(ir, args.iris_textures + '.png'))

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(os.getcwd(),'static_model', model, output, str(args.person_idx) + '_' + str(args.trial_idx) , 'seq.blend'))
    if args.env in [str(x) for x in range(1, 26)]:
        world.node_tree.nodes["Environment Texture"].image = bpy.data.images.load(
            filepath=os.path.join(env_path, args.env + '.hdr'))
        l1.data.energy = 0
        l2.data.energy = 0
        # bpy.data.materials["Eye wet material.001"].node_tree.nodes["BSDF Reflectivo"].inputs[1].default_value = 0

    # temp HIDES the irrelavant stuff
    # bpy.data.objects["head"].hide_render = True
    # bpy.data.objects["Eye_Plica"].hide_render = True
    # bpy.data.objects["lower"].hide_render = True
    # bpy.data.objects["upper"].hide_render = True
    
    for i in range(1, frame_cap):

        s.frame_current = i
        filename = os.path.join(output_folder,'rotated', 'Prsn_' + str(args.person_idx) + '_Trial_' + str(args.trial_idx) + '_Head_' + str(args.model_id), 'synthetic',
                                str(s.frame_current).zfill(4) + ".png")
        if os.path.isfile(filename):
            print("skipped ", filename)
            continue
        else:
            print(i)

        s.render.filepath = (filename)

        bpy.ops.render.render(  # {'dict': "override"},
            # 'INVOKE_DEFAULT',
            False,  # undo support
            animation=False,
            write_still=True
        )

    # skin black
    skin.node_tree.links.remove(skin.node_tree.nodes["Material Output"].inputs['Surface'].links[0])
    skin.node_tree.links.remove(skin.node_tree.nodes["Material Output"].inputs['Displacement'].links[0])

    # eye plica black
    wet.node_tree.links.remove(wet.node_tree.nodes["Material Output"].inputs['Surface'].links[0])
    wet.node_tree.links.remove(wet.node_tree.nodes["Material Output"].inputs['Displacement'].links[0])

    # iris to green
    iris_mat.node_tree.links.new(iris_mat.node_tree.nodes["Emission"].outputs['Emission'],
                                 iris_mat.node_tree.nodes["Material Output"].inputs[0])
    # iris_mat.node_tree.links.remove(iris_mat.node_tree.nodes["Material Output"].inputs['Displacement'].links[0])

    retina = bpy.data.materials['Sclera material.001']
    # retina to red
    retina.node_tree.links.new(retina.node_tree.nodes["Emission"].outputs['Emission'],
                               retina.node_tree.nodes["Material Output"].inputs[0])

    glass.hide_render = True
    sclera.hide_render = True
    bpy.data.objects["EyeWet.002"].hide_render = True
    bpy.data.objects["lower"].hide_render = True
    bpy.data.objects["upper"].hide_render = True
    bpy.data.objects["sclera_mask"].hide_render = False
    l1.hide_render = True
    l2.hide_render = True

    # RENDERING MASK IMAGES
    for i in range(1, frame_cap):
        glass.hide_render = True
        glass.keyframe_insert(data_path="hide_render", frame=i)

        frame = i
        s.frame_current = frame
        filename = os.path.join(output_folder, 'Prsn_' + str(args.person_idx) + '_Trial_' + str(args.trial_idx) + '_Head_' + str(args.model_id), 'maskwith_skin',
                                str(s.frame_current).zfill(4) + ".png")
        if os.path.isfile(filename):
            print("skipped ", filename)
            continue
        else:
            print(frame)

        s.render.filepath = filename

        # bpy.ops.render.render(  # {'dict': "override"},
        #     # 'INVOKE_DEFAULT',
        #     False,  # undo support
        #     animation=False,
        #     write_still=True)

    bpy.data.objects["head"].hide_render = True
    bpy.data.objects["Eye_Plica"].hide_render = True
    bpy.data.objects["Sunglasses"].hide_render = True

    for i in range(1, frame_cap):
        try:
            s.node_tree.links.new(s.node_tree.nodes["Render Layers"].outputs['Image'],
                                  s.node_tree.nodes["Composite"].inputs[0])
        except:
            print('No node')
        frame = i
        s.frame_current = frame
        filename = os.path.join(output_folder, 'Prsn_' + str(args.person_idx) + '_Trial_' + str(args.trial_idx) + '_Head_' + str(args.model_id), 'maskwithout_skin',
                                str(s.frame_current).zfill(4) + ".png")
        if os.path.isfile(filename):
            print("skipped ", filename)
            continue
        else:
            print(frame)
            s.render.filepath = filename

            # bpy.ops.render.render(  # {'dict': "override"},
            #     # 'INVOKE_DEFAULT',
            #     False,  # undo support
            #     animation=False,
            #     write_still=True)

        # Rendering depth map but on hold for now
        # s.node_tree.links.new(s.node_tree.nodes["ColorRamp"].outputs['Image'], s.node_tree.nodes["Composite"].inputs[0])
        # s.render.filepath = os.path.join(output_folder,model,output,p[:-2],'depth' ,str(s.frame_current).zfill(4) + ".tif")
        # if os.path.isfile(os.path.join(output_folder,model,output,p[:-2],'maskwithout_skin' ,str(s.frame_current).zfill(4) + ".tif")):
        #   print("skipped ",os.path.isfile(os.path.join(output_folder,model,output,p[:-2],'maskwithout_skin' ,str(s.frame_current).zfill(4) + ".tif")))
        # else:
        #   bpy.ops.render.render(  # {'dict': "override"},
        #          # 'INVOKE_DEFAULT',
        #         False,  # undo support
        #        animation=False,
        #       write_still=True)
    for x in range(1, frame_cap):
        print(x)
        data(x)
    for x in gt.keys():
        gt[x] = gt[x][1:]
    #pickle.dump(gt, open(os.path.join(output_folder, model, output, p[:-2], model + '-pupil.p'), "wb"))
    print("saved")

def render():

    import bpy
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cuda_devices, opencl_devices = cycles_preferences.get_devices()
    device_type = "CUDA"
    print("USING THE FOLLOWING GPUS:")
    devices=[]
    cycles_preferences.compute_device_type = device_type
    for x in range(len(cycles_preferences.devices)):
        if cycles_preferences.devices[x] in cuda_devices:
            devices.append(cycles_preferences.devices[x])
    for x in range(len(devices)):
        print(devices[x].name)
        devices[0].use = True #changed @@yush
    # Image resolution
    s = bpy.context.scene
    s.cycles.device = 'GPU'

    s.render.resolution_x = 400
    s.render.resolution_y = 640

    # Rendering tile size
    s.render.tile_x = 400
    s.render.tile_y = 640

    iris_mat = bpy.data.materials['Iris.000']
    sclera_mat = bpy.data.materials['Sclera material.000']
    ir  =os.path.join(os.getcwd(),'Textures_eye','ir-textures/')
    scl =os.path.join(os.getcwd(),'Textures_eye','sclera/')
    sclera_mat.node_tree.nodes["op"].image = bpy.data.images.load(filepath=os.path.join(os.getcwd(),'Textures_eye','opacity.png'))

    skin.node_tree.nodes["c1"].image =bpy.data.images.load(filepath= os.path.join(os.getcwd(),'static_model',model,'Textures','c.jpg'))
    skin.node_tree.nodes["c2"].image = bpy.data.images.load(filepath=os.path.join(os.getcwd(),'static_model',model,'Textures','c1.jpg'))
    
    skin.node_tree.nodes["g1"].image =bpy.data.images.load(filepath=os.path.join(os.getcwd(),'static_model',model,'Textures','g.jpg'))
    
    skin.node_tree.nodes["s1"].image =bpy.data.images.load(filepath=os.path.join(os.getcwd(),'static_model',model,'Textures','s.jpg'))
    
    skin.node_tree.nodes["n"].image =bpy.data.images.load(filepath=os.path.join(os.getcwd(),'static_model',model,'Textures','n.jpg'))


    for i in range(number):

        s.frame_current = i
        filename = os.path.join(output_folder, model, output, 'synthetic', str(s.frame_current).zfill(4) + ".tif")
        if os.path.isfile(filename):
            print("skipped ", filename)
            continue
        else:
            print(i)

        iris_mat.node_tree.nodes["iris"].image = bpy.data.images.load(filepath=os.path.join(ir, a[i] + '.png'))
        sclera_mat.node_tree.nodes["sclera"].image = bpy.data.images.load(filepath=os.path.join(scl, b[i] + '.png'))

        s.render.filepath = (filename)

        bpy.ops.render.render(  # {'dict': "override"},
            # 'INVOKE_DEFAULT',
            False,  # undo support
            animation=False,
            write_still=True
        )

    # skin black
    skin.node_tree.links.remove(skin.node_tree.nodes["Material Output"].inputs['Surface'].links[0])
    skin.node_tree.links.remove(skin.node_tree.nodes["Material Output"].inputs['Displacement'].links[0])

    # eye plica black
    wet.node_tree.links.remove(wet.node_tree.nodes["Material Output"].inputs['Surface'].links[0])
    wet.node_tree.links.remove(wet.node_tree.nodes["Material Output"].inputs['Displacement'].links[0])

    # iris to green
    iris_mat.node_tree.links.new(iris_mat.node_tree.nodes["Emission"].outputs['Emission'],
                                 iris_mat.node_tree.nodes["Material Output"].inputs[0])
    # iris_mat.node_tree.links.remove(iris_mat.node_tree.nodes["Material Output"].inputs['Displacement'].links[0])

    retina = bpy.data.materials['Sclera material.001']
    # retina to red
    retina.node_tree.links.new(retina.node_tree.nodes["Emission"].outputs['Emission'],
                               retina.node_tree.nodes["Material Output"].inputs[0])

    glass.hide_render = True
    sclera.hide_render = True
    bpy.data.objects["EyeWet.002"].hide_render = True
    bpy.data.objects["lower"].hide_render = True
    bpy.data.objects["upper"].hide_render = True
    bpy.data.objects["sclera_mask"].hide_render = False
    l1.hide_render = True
    l2.hide_render = True

    # RENDERING MASK IMAGES
    for i in range(number):
        glass.hide_render = True
        glass.keyframe_insert(data_path="hide_render", frame=i)

        frame = i
        s.frame_current = frame
        filename = os.path.join(output_folder, model, output, 'maskwith_skin',
                                str(s.frame_current).zfill(4) + ".tif")
        if os.path.isfile(filename):
            print("skipped ", filename)
            continue
        else:
            print(frame)

        s.render.filepath = filename

        bpy.ops.render.render(  # {'dict': "override"},
            # 'INVOKE_DEFAULT',
            False,  # undo support
            animation=False,
            write_still=True)

    bpy.data.objects["head"].hide_render = True
    bpy.data.objects["Eye_Plica"].hide_render = True
    bpy.data.objects["Sunglasses"].hide_render = True

    for i in range(number):
        try:
            s.node_tree.links.new(s.node_tree.nodes["Render Layers"].outputs['Image'],
                                  s.node_tree.nodes["Composite"].inputs[0])
        except:
            print("No nodes")

        frame = i
        s.frame_current = frame
        filename = os.path.join(output_folder, model, output, 'maskwithout_skin',
                                str(s.frame_current).zfill(4) + ".tif")
        if os.path.isfile(
                os.path.join(output_folder, model, output, 'depth', str(s.frame_current).zfill(4) + ".tif")):
            print("skipped ", filename)
            continue
        else:
            print(frame)

        s.render.filepath = filename

    #    bpy.ops.render.render(  # {'dict': "override"},
    #        # 'INVOKE_DEFAULT',
   #         False,  # undo support
    #        animation=False,
    #        write_still=True)
     #   s.node_tree.links.new(s.node_tree.nodes["ColorRamp"].outputs['Image'], s.node_tree.nodes["Composite"].inputs[0])
    #    s.render.filepath = os.path.join(output_folder, model, output, 'depth',
    #                                     str(s.frame_current).zfill(4) + ".tif")

    #    bpy.ops.render.render(  # {'dict': "override"},
    #        # 'INVOKE_DEFAULT',
    #        False,  # undo support
    #        animation=False,
    #        write_still=True)

def example_render():
    import bpy 
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cuda_devices, opencl_devices = cycles_preferences.get_devices()
    device_type = "OPTIX"

    print(output_folder)
    try:
        os.makedirs(os.path.join(output_folder, 'example'))
    except OSError:
        print("Creation of the directory %s failed" % os.path.join(output_folder, 'example'))
    


    s = bpy.context.scene

    # Image resolution
    s.render.resolution_x = 640
    s.render.resolution_y = 480

    # Rendering tile size
    s.render.tile_x = 640
    s.render.tile_y = 480

    s.cycles.device = 'GPU'
    s.render.image_settings.file_format = 'PNG'

    model = args.model_id[0:1]

    # Loading all textures
    # Iris texture
    iris_mat = bpy.data.materials['Iris.000']
    # Sclera texture
    sclera_mat = bpy.data.materials['Sclera material.000']

    ir = os.path.join(os.getcwd(),'Textures_eye','ir-textures')
    scl = os.path.join(os.getcwd(),'Textures_eye','sclera')
    sclera_mat.node_tree.nodes["op"].image = bpy.data.images.load(filepath=os.path.join(os.getcwd(),'Textures_eye','opacity.png'))
    sclera_mat.node_tree.nodes["sclera"].image = bpy.data.images.load(filepath=os.path.join(os.getcwd(),'Textures_eye','Sclera color.png'))

    # picking a cornea
    model_int = int(model)
    if model_int%4 == 0:
        c1.hide_render = False
        c2.hide_render = True
        c2.hide_render = True
        sphere.hide_render = True
    if model_int%4 == 1:
        c1.hide_render = True
        c2.hide_render = False
        c2.hide_render = True
        sphere.hide_render = True
    if model_int%4 == 2:
        c1.hide_render = True
        c2.hide_render = True
        c2.hide_render = False
        sphere.hide_render = True
    if model_int%4 == 3:
        c1.hide_render = True
        c2.hide_render = True
        c2.hide_render = True
        sphere.hide_render = False

    # Skin Textures
    skin.node_tree.nodes["c1"].image =bpy.data.images.load(filepath=os.path.join(os.getcwd(),'static_model',model,'Textures','c.jpg'))
    skin.node_tree.nodes["c2"].image =bpy.data.images.load(filepath=os.path.join(os.getcwd(),'static_model',model,'Textures/c1.jpg'))
    skin.node_tree.nodes["g1"].image =bpy.data.images.load(filepath=os.path.join(os.getcwd(),'static_model',model,'Textures/g.jpg'))
    skin.node_tree.nodes["s1"].image =bpy.data.images.load(filepath=os.path.join(os.getcwd(),'static_model',model,'Textures/s.jpg'))
    skin.node_tree.nodes["n"].image = bpy.data.images.load(filepath=os.path.join(os.getcwd(),'static_model',model,'Textures/n.jpg'))

    # iris texture
    iris_mat.node_tree.nodes["iris"].image = bpy.data.images.load(
        filepath=os.path.join(ir, model+'.png'))

    # sets the eye reflection to be slightly bigger
    bpy.data.objects["EyeWet.002"].scale = bpy.data.objects["EyeWet.002"].scale * 1.02
    #bpy.data.objects["EyeWet.002"].hide_render = True
    l1.data.energy = 0
    l2.data.energy = 0
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0.5, 0.5, 0.5, 1)

    
    l1.location[1] = -l1.location[1]
    l2.location[1] = -l2.location[1]

    # sets the head location and rotation
    head.location[1] = head.location[1]+4
    if head.name == 'head':
        head.rotation_euler[0] = math.radians(90)
        camera.data.type = 'ORTHO'
        camera.data.ortho_scale = 3
        bpy.data.objects['upper'].location[1] += 4
        bpy.data.objects['lower'].location[1] += 4
        bpy.data.objects['Eye_Plica'].location[1] += 4
        camera.rotation_euler[0] = 0


    else:
        head.rotation_euler[0] = 0
        camera.rotation_euler[0] = math.radians(90)

    head.rotation_euler[1] = 0
    head.rotation_euler[2] = 0

    eye.location = [0,4,0]
    eye.rotation_euler[1] = 0
    eye.rotation_euler[2] = 0
    eye.rotation_euler[0] = math.radians(90)

    # sets the pupil
    bpy.data.meshes['Roundcube.000'].shape_keys.key_blocks["Pupil contract"].value = 0.5

    # sets the empty location  
    empty.location = [0,3,0]

    # sets up the camera location and rotation
    camera.location = [0,0,0]
    camera.rotation_euler[1] = 0
    camera.rotation_euler[2] = 0

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(output_folder, 'example', args.model_id + '_seq.blend'))

    filename = os.path.join(output_folder, 'example', args.model_id + ".png")
    if os.path.isfile(filename):
        print("skipped ", filename)
    else:
        s.render.filepath = filename
        bpy.ops.render.render(
            False,
            animation=False,
            write_still=True)


def main():
    if args.data_source == 'pickle':
        print('LOADING DATAFROM GIVEN PICKLE FILE')
        data_pickle = pickle.load(open(os.path.join(args.data_source_path), "rb"), encoding="latin1")
        data_keyframe(data_pickle)
        if not os.path.isdir(os.path.join('static_model', model, output)):
            os.mkdir(os.path.join('static_model', model, output))
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join('static_model', model, output, 'seq.blend'))
        for x in range(-1, number):
            print(x)
            data(x)
        for x in gt.keys():
            gt[x] = gt[x][1:]
        pickle.dump( gt, open(os.path.join(os.getcwd(),'static_model',model,output,model+'-pupil.p'), "wb" ))
        print("saved")
        render()
    elif args.data_source == 'random':
        print('SETTING RANDOM VALUES')
        keyframe()
        if not os.path.isdir(os.path.join('static_model', model, output)):
            os.mkdir(os.path.join('static_model', model, output))
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join('static_model', model, output, 'seq.blend'))
        for x in range(-1, number):
            print(x)
            data(x)
        for x in gt.keys():
            gt[x] = gt[x][1:]
        pickle.dump( gt, open(os.path.join(os.getcwd(),'static_model',model,output,model+'-pupil.p'), "wb" ))
        print("saved")
        render()
    elif args.data_source == 'seq':
        if 'blink' in args.inp:
    	        print("RENDERING SEQUENTIAL DATA with blink")
    	        blink_render()
        else:
        
    	        print("RENDERING SEQUENTIAL DATA")
    	        seq_render()
    elif args.data_source == 'example':
        print('CREATING EXAMPLE IMAGE')
        example_render()

    


main()

