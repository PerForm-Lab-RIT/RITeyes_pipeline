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
import mathutils

if '--' in sys.argv:
    argv = sys.argv[sys.argv.index('--') + 1:]

# Define parameters such as head model, total number of images to render, rendering storage location and gaze vectors.
# The gaze vectors can be random or loaded from a pickle file .
# Note: (1) head model should have pupil_blend and texture files (Folder structure ./static_model/{head_model_id}/{head_model_id}.pupil_blend & Textures 

#We can share the head models used in our pipeline only after the sponsor licenses the head models, as described in the SOW.
#Head model can be purchased here https://www.3dscanstore.com/3d-head-models/female-retopologised-3d-head-models/male-female-3d-head-model-24xbundle
parser = argparse.ArgumentParser()

#########################################################################################################################################################################
# Define setup parameters/details
parser.add_argument('--model_id', help='Choose a head model (1-24)',default='3')
parser.add_argument('--number', type=str,help='total number of images)',default= 100) ##Only used for random static image renderings
parser.add_argument('--data_source', type=str,help='Provide a source in pickle file. Else choose random or sequential',default='random' )
parser.add_argument('--data_source_path', type=str,help='Select data source path for pickle file',default=None)
parser.add_argument('-inp', help='output file',default='./giw_small') ##To specify where to save the rendered images and mask
parser.add_argument('--start', type=str,help='start frame',default='1' )
parser.add_argument('--end', type=str,help='end frame',default='10000') # While sequential rendering if you need a specific portion of the video to rendered you can specify the start and end index
parser.add_argument('--gpu', type=str,help='GPU ids to use',default='0,1,2,3' )
parser.add_argument('--output_file', type=str,help='Please make sure that there are duplicates',default='mark2')

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

parser.add_argument('--camera_focal_length', type=str,help='Focal length of the camera',default=50)
parser.add_argument('--camera_distance', type=str,help='Range of camera distance of the eye from the tip of the eye [from,to]',default='4,5')
parser.add_argument('--camera_azimuthal', type=str,help='Range of camera horizontal movement in degrees [from ,to] ',default='-55,-65')
parser.add_argument('--camera_elevation', type=str,help='Range of camera vertical movement in degrees [from ,to] (+ve means down and -ve means up)',default='20,30')
parser.add_argument('--camera_location', type=str,help='Location of the camera',default='20,30')

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

#########################################################################################################################################################################
args = parser.parse_args(argv)
gt={'camera_3d_center':[],'camera_distance':[],'iris_loc':[],'eye_loc':[],'gaze_angle_az':[],'gaze_angle_el':[],'ortho_scale':[],'cam_az':[],'cam_el':[],'glasses':[],'pupil':[],'eye_lid':[],'iris_rot':[],'left_corner':[],'right_corner':[],'sclera':[],'cam_focal_length':[],'cam_sensor_size':[],'iris_texture':[],'light1_loc':[],'light2_loc':[],'light1_az':[],'light1_el':[],'light2_az':[],'light2_el':[],'sclera_texture':[],'sclera_dark':[],'glass_loc':[],'glass_rot':[],'target':[],'cornea':[]}
sys.path=[]
sys.path.append('./blender-2.82a-linux64/2.82/scripts/startup')
sys.path.append('./blender-2.82a-linux64/2.82/scripts/modules')
sys.path.append('./blender-2.82a-linux64/2.82/scripts/freestyle/modules')
sys.path.append('./blender-2.82a-linux64/2.82/scripts/addons/modules')
sys.path.append('./blender-2.82a-linux64/2.82/python/lib/python3.7')
sys.path.append('./blender-2.82a-linux64/2.82/python/lib/python3.7/site-packages')
print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

#########################################################################################################################################################################
# This convert functions converts/filters the arguments specified into blender specific format.
def convert(x):
    x=x.replace(" ","")
    if '-' in x:
        return float(x[1:])*-1
    else:

        return float(x)
      
#Objects are generated by the user through blender  
model=args.model_id
number=int(args.number)

camera=bpy.data.objects['Camera']   #camera as objects (location, angle)
cam=bpy.data.cameras['Camera']      #camera intrinsic properties

#allows to select perspective or orthographic projection
cam.type='PERSP'
empty=bpy.data.objects['Empty']
eye=bpy.data.objects["Eye.Wetness"]
iris=bpy.data.objects["EyeIris"]
sclera=bpy.data.objects["Sclera"]
glass=bpy.data.objects["Sunglasses"]

#define ashperical parameters. More parameters can be added here. (We use only three)
c1=bpy.data.objects["63"] 
c2=bpy.data.objects["75"]
c3=bpy.data.objects["87"]
sphere=bpy.data.objects["sphere"]
wet=bpy.data.materials['Eye_Wetness_01']
inp = args.inp
sclera_mat=bpy.data.materials['Sclera material.000']
skin=bpy.data.materials['skin']

#Lamp energy set as 4mW=40
l1=bpy.data.objects['Lamp.001']
l2=bpy.data.objects['Lamp']
l1.data.type = 'AREA'
l2.data.type = 'AREA'

#########################################################################################################################################################################
# Converting the data into format used for rendering : (Data processing)
args.camera_distance=args.camera_distance.split(',')[1:]

for x in range(len(args.camera_distance)):
    args.camera_distance[x]=convert(args.camera_distance[x])
    
args.camera_focal_length=float(args.camera_focal_length)
cam.lens=args.camera_focal_length
args.pupil=args.pupil.split(',')[1:]
for x in range(len(args.pupil)):
    args.pupil[x]=convert(args.pupil[x])

args.eye_azimuthal=args.eye_azimuthal.split(',')[1:]
for x in range(len(args.eye_azimuthal)):
    args.eye_azimuthal[x]=convert(args.eye_azimuthal[x])
    
args.eye_elevation=args.eye_elevation.split(',')[1:]
for x in range(len(args.eye_elevation)):
    args.eye_elevation[x]=convert(args.eye_elevation[x])

args.cornea=args.cornea.split(',')[1:]
a=args.iris_textures.split(',')[1:]*number
b=args.sclera_textures.split(',')[1:]*number

args.camera_elevation=args.camera_elevation.split(',')[1:]
args.camera_azimuthal=args.camera_azimuthal.split(',')[1:]

for x in range(len(args.camera_elevation)):
    args.camera_elevation[x]=convert(args.camera_elevation[x])
    
for x in range(len(args.camera_azimuthal)):
    args.camera_azimuthal[x]=convert(args.camera_azimuthal[x])

l1.data.energy=convert(args.light_1_energy)
l2.data.energy=convert(args.light_2_energy)

start=convert(args.start)
end=convert(args.end)

args.light_1_loc=args.light_1_loc.split(',')[1:]
args.light_2_loc=args.light_2_loc.split(',')[1:]
args.light_1_loc[0]=convert(args.light_1_loc[0])
args.light_1_loc[1]=convert(args.light_1_loc[1])
args.light_1_loc[2]=convert(args.light_1_loc[2])

args.light_2_loc[0]=convert(args.light_2_loc[0])
args.light_2_loc[1]=convert(args.light_2_loc[1])
args.light_2_loc[2]=convert(args.light_2_loc[2])

l1.location[0]=args.light_1_loc[0]
l1.location[1]=args.light_1_loc[1]
l1.location[2]=args.light_1_loc[2]

args.glass=int(args.glass)

l2.location[0]=args.light_2_loc[0]
l2.location[1]=args.light_2_loc[1]
l2.location[2]=args.light_2_loc[2]

#########################################################################################################################################################################
# Camera Distance from
closest_camera_distance=args.camera_distance[0]
# Camera Distance to 
farthest_camera_distance=args.camera_distance[1]

# pupil size from (0 to 1 means 1mm to 4mm)
pupil_min=(args.pupil[0]-1)/3
# pupil size to
pupil_max=(args.pupil[1]-1)/3

cor=args.cornea*number              
output=args.output_file
if_glass=[0]*number
for x in range(number*args.glass//100):
    if_glass[x]=1
    

#inserting keyframes is necessary for setting different positions and model states over time. This timeline is then rendered as separate frames.
def keyframe():
    for i in range(-1,number):
        print('Keyframing: ',i)
        # camera distance from 1 cm to 3 cm
        r = random.uniform(closest_camera_distance,farthest_camera_distance)
        
        # azimuthal
        phi = random.uniform(90+args.camera_azimuthal[0],90+args.camera_azimuthal[1])
        # elevation
        theta = random.uniform(90+args.camera_elevation[0],90+args.camera_elevation[1])

        # calculating camera location 
        x=r*math.sin(math.radians(theta))*math.cos(math.radians(phi))
        y=r*math.sin(math.radians(theta))*math.sin(math.radians(phi))
        z=r*math.cos(math.radians(theta))
        
        # camera location based on distance and angle
        camera.location[0] = x
        camera.location[1] = -1-y
        camera.location[2] = z
        
        # camera target 1 mm vertically and horizontally : Adding noise to the camera location
        empty.location[0] = random.uniform(-0.5,0.5)
        empty.location[1] = -1
        empty.location[2] = random.uniform(-0.5,0.5)
        
        eye.rotation_euler[0] = math.radians(90)+math.radians(random.uniform(args.eye_elevation[0],args.eye_elevation[1]))
        eye.rotation_euler[2] = math.radians(random.uniform(args.eye_azimuthal[0],args.eye_azimuthal[1]))
        # pupil size from 1mm to 3 mm
        bpy.data.meshes['Roundcube.000'].shape_keys.key_blocks["Pupil contract"].value = random.uniform(pupil_min,pupil_max)
        
        # iris and sclera rotation (Any value from 0 , 360)
        iris.rotation_euler[2] = random.uniform(0,6.28319)
        sclera.rotation_euler[2] = random.uniform(0,6.28319)
        
        # Making sclera dark
        sclera_mat.node_tree.nodes["sclera_dark"].inputs[0].default_value = random.uniform(0,0.8)    
    
        # select with or without glasses.
        if if_glass[i] == 1:
            glass.hide_render = False
            glass.location[0] = random.uniform(-1.97501,-2.22824)
            glass.location[1] = random.uniform(-2.00167, -1.7237)
            glass.location[2] = random.uniform(-0.531074, -0.774396)
            glass.rotation_euler[0] = random.uniform(math.radians(88), math.radians(92))
            glass.rotation_euler[1] = random.uniform(math.radians(-2), math.radians(2))
            glass.rotation_euler[2] = random.uniform(math.radians(-2), math.radians(2))
            glass.keyframe_insert(data_path="location", frame=i)
            glass.keyframe_insert(data_path="rotation_euler", frame=i)
        else:
           glass.hide_render=True
        
        # Selecting random eye lid closure value from 0 to 1
        lid=random.uniform(0.0,0.5)
        bpy.data.meshes['eye_lid'].shape_keys.key_blocks["Key 1"].value = lid
        bpy.data.meshes['l'].shape_keys.key_blocks["Key 1"].value = lid
        bpy.data.meshes['u'].shape_keys.key_blocks["Key 1"].value = lid
        
        # Eye lid closure keyframing
        bpy.data.meshes['eye_lid'].shape_keys.key_blocks["Key 1"].keyframe_insert(data_path="value",frame=i)
        bpy.data.meshes['l'].shape_keys.key_blocks["Key 1"].keyframe_insert(data_path="value",frame=i)
        bpy.data.meshes['u'].shape_keys.key_blocks["Key 1"].keyframe_insert(data_path="value",frame=i)
        
        # blend eye lid: changing eyelid texture based on eyelid openness
        if lid < 0.2:
            lid=lid
        else:
            lid=lid*1.5
        skin.node_tree.nodes["blink"].inputs[0].default_value = lid
        
        # keyframe blend eye lid
        skin.node_tree.nodes["blink"].inputs[0].keyframe_insert(data_path="default_value",frame=i)
        
        # Select cornea 
        c=cor[i]
        if c=='0':
            c1.hide_render=False
            c2.hide_render=True
            c3.hide_render=True
        if c=='1':
            c2.hide_render=False
            c3.hide_render=True
            c1.hide_render=True
        if c=='2':
            c3.hide_render=False
            c2.hide_render=True
            c1.hide_render=True
        camera.rotation_euler[0]=0
        camera.rotation_euler[1]=0
        camera.rotation_euler[2]=0
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
        bpy.data.meshes['Roundcube.000'].shape_keys.key_blocks["Pupil contract"].keyframe_insert(data_path="value",frame=i)
        
        # keyframing sclera color
        sclera_mat.node_tree.nodes["sclera_dark"].inputs[0].keyframe_insert(data_path="default_value",frame=i)
        
        # Cornea key framing
        c1.keyframe_insert(data_path="hide_render", frame=i)
        c2.keyframe_insert(data_path="hide_render", frame=i)
        c3.keyframe_insert(data_path="hide_render", frame=i)
        
# data that is keyframed from a pickle file
def data_keyframe(data):
    for i in range(-1,max(number,len(data['camera_3d_center']))):
        print('Keyframing: ',i)

        #camera location based on distance and angle
        camera.location[0]=data['camera_3d_center'][i][0]
        camera.location[1]=data['camera_3d_center'][i][1]
        camera.location[2]=data['camera_3d_center'][i][2]

        #camera target 1 mm vertically and horizontally
        empty.location[0]=data['target'][i][0]
        empty.location[1]=-1
        empty.location[2]=data['target'][i][2]
        eye.rotation_euler[0] = math.radians(90+data['gaze_angle_el'][i])
        eye.rotation_euler[2] =  math.radians(data['gaze_angle_az'][i])

        #pupil size from 1mm to 3 mm
        bpy.data.meshes['Roundcube.000'].shape_keys.key_blocks["Pupil contract"].value =(data['pupil'][i]-1)/3
        
        #iris and sclera rotation (Any value from 0 , 360)
        iris.rotation_euler[2]= data['iris_rot'][i][2]
        sclera.rotation_euler[2]= data['sclera'][i][2]
        
        #Making sclera dark
        sclera_mat.node_tree.nodes["sclera_dark"].inputs[0].default_value = data['sclera_dark'][i]
        #sclera_mat.node_tree.nodes["sclera_dark"].inputs[0].default_value =random.uniform(0,0.8)
        
        cam.lens= data['cam_focal_length'][i]
        cam.sensor_width=data['cam_sensor_size'][i]
        
        #select with or without glasses.
        gl=data['glasses'][i]
        if gl:
            glass.hide_render=False
            glass.location[0] = random.uniform(-1.97501,-2.22824)
            glass.location[1] = random.uniform(-2.00167, -1.7237)
            glass.location[2] = random.uniform(-0.531074, -0.774396)
            glass.rotation_euler[0] = random.uniform(math.radians(88), math.radians(92))
            glass.rotation_euler[1] = random.uniform(math.radians(-2), math.radians(2))
            glass.rotation_euler[2] = random.uniform(math.radians(-2), math.radians(2))
            glass.keyframe_insert(data_path="location", frame=i)
            glass.keyframe_insert(data_path="rotation_euler", frame=i)
        else:
           glass.hide_render=True

        # Selecting random eye lid closure value from 0 to 1
        lid=data['eye_lid'][i]/100

        bpy.data.meshes['eye_lid'].shape_keys.key_blocks["Key 1"].value = lid
        ## 'l' and 'u' are lower and upper eyelid meshes that are responsible for eyelashes
        bpy.data.meshes['l'].shape_keys.key_blocks["Key 1"].value = lid
        bpy.data.meshes['u'].shape_keys.key_blocks["Key 1"].value = lid
        
        # Eye lid closure keyframing
        bpy.data.meshes['eye_lid'].shape_keys.key_blocks["Key 1"].keyframe_insert(data_path="value",frame=i)
        bpy.data.meshes['l'].shape_keys.key_blocks["Key 1"].keyframe_insert(data_path="value",frame=i)
        bpy.data.meshes['u'].shape_keys.key_blocks["Key 1"].keyframe_insert(data_path="value",frame=i)

        #blend eye lid
        if lid < 0.2:
            lid=lid
        else:
            lid=lid*1.5
        skin.node_tree.nodes["blink"].inputs[0].default_value = lid
        
        #keyframe blend eye lid
        skin.node_tree.nodes["blink"].inputs[0].keyframe_insert(data_path="default_value",frame=i)
        
        #Select cornea 
        c=cor[i]
        if c=='0':
            c1.hide_render=False
            c2.hide_render=True
            c3.hide_render=True
            
        if c=='1':
            c2.hide_render=False
            c3.hide_render=True
            c1.hide_render=True
        if c=='2':
            c3.hide_render=False
            c2.hide_render=True
            c1.hide_render=True
            
        camera.rotation_euler[0]=0
        camera.rotation_euler[1]=0
        camera.rotation_euler[2]=0
        
        camera.keyframe_insert(data_path="rotation_euler", frame=i)

        #Camera location keyframing
        camera.keyframe_insert(data_path="location", frame=i)
        
        #Camera target keyframing
        empty.keyframe_insert(data_path="location", frame=i)
        
        #Iris and sclera key framing
        eye.keyframe_insert(data_path="rotation_euler", frame=i)
        
        #Iris and sclera key framing
        iris.keyframe_insert(data_path="rotation_euler", frame=i)
        sclera.keyframe_insert(data_path="rotation_euler", frame=i)
        
        #Glass keyframe
        glass.keyframe_insert(data_path="hide_render", frame=i)
        
        #keyframing pupil size
        bpy.data.meshes['Roundcube.000'].shape_keys.key_blocks["Pupil contract"].keyframe_insert(data_path="value",frame=i)
        
        #keyframing sclera color
        sclera_mat.node_tree.nodes["sclera_dark"].inputs[0].keyframe_insert(data_path="default_value",frame=i)
        
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

    # pickle file name for sequential rendering
    p=args.data_source_path

    try:
        camera.constraints["Track To"].target = None
    except:
        pass

    # Making a directory with specified output file name
    try:
      os.makedirs(os.path.join('./GIW_Renderings',model,output,p[:-2]))
    except OSError:
        print ("Creation of the directory %s failed" % os.path.join('./GIW_Renderings',model,output,p[:-2]))
    else:
        print ("Successfully created the directory %s" % os.path.join('./GIW_Renderings',model,output,p[:-2]))

    ## Load the data file
    file = pickle.load( open(os.path.join(inp,p), "rb" ),encoding="latin1" )
    # Seperate lists for head vectors x,y,z and eye vectors x,y,z and pupil 
    # The pickle files need the first 2 index of the eye and head vectors to be 0,0,1. 
    # These images will not be rendered but still it needs to be there as reference for the start position 

    # Saving the vectors for transition type rendering
    # Note you select transition when you plan to render eye movements with transition
    # Data in the pickle file has been labeled as 1,2,3,4 for fixation, smooth pursuit, blink and saccade respectively as original GIW dataset
    if args.seq_type == 'transition':
            head_x=[]
            head_y=[]
            head_z=[]
            eye_x=[]
            eye_y=[]
            eye_z=[]
            pupil=[]
            for id in ['13','31','14','41','34','43']:
                for i in range(len(file[id][6])):
                    for x in range(0,600,5):
                        head_x.append(file[id][0][i][x])
                        head_y.append(file[id][1][i][x])
                        head_z.append(file[id][2][i][x])
                        eye_x.append(file[id][3][i][x])
                        eye_y.append(file[id][4][i][x])
                        eye_z.append(file[id][5][i][x])
                        pupil.append(file[id][7][i][x])

    # Saving the vectors for all type rendering
    # This does not care about labeled dataset but renders all the data in the pickle file 
    # We are using this setup currently for our pipeline 
    if args.seq_type == 'all':
         head_x=[] 
         head_y=[] 
         head_z=[] 
         eye_x=[] 
         eye_y=[] 
         eye_z=[]
         pupil=[] 
         for x in range(len(file['h_x'])):
             head_x.append(file['h_x'][x])
             head_y.append(file['h_y'][x])  
             head_z.append(file['h_z'][x])
             eye_x.append(file['e_x'][x])
             eye_y.append(file['e_y'][x]) 
             eye_z.append(file['e_z'][x])
             pupil.append(file['pupil'][x])

    
            
    head_x = [ float(x) for x in head_x ]     
    head_y = [ float(x) for x in head_y ]  
    head_z = [ float(x) for x in head_z ]  
    eye_x = [ float(x) for x in eye_x ]  
    eye_y = [ float(x) for x in eye_y ]  
    eye_z = [ float(x) for x in eye_z ]

    # Calculating the pupil size in blender format
    pupil = [ float(((x/2)*(0.8/3))-(0.5/3)) for x in pupil ]   ##How did you choose pupil size

    eye.rotation_mode='XYZ'
    #head.rotation_mode='XYZ'
    s = bpy.context.scene

    # Image resolution
    s.render.resolution_x = 640
    s.render.resolution_y = 480

    # Rendering tile size
    s.render.tile_x = 640
    s.render.tile_y = 480


    s.cycles.device = 'GPU'
    s.render.image_settings.file_format='TIFF'

    # Loading all textures
    #Iris texture
    iris_mat=bpy.data.materials['Iris.000']
    # Sclera texture
    sclera_mat=bpy.data.materials['Sclera material.000']
    ir = './Textures_eye/ir-textures/'
    scl = './Textures_eye/sclera/'
    sclera_mat.node_tree.nodes["op"].image =bpy.data.images.load(filepath='./Textures_eye/opacity.png')
    #Skin Textures
    ## c1 and c2 are for eyelid openess and closeness and takes a weighted sum in intermediate positions.
    skin.node_tree.nodes["c1"].image =bpy.data.images.load(filepath= os.path.join('./static_model',model,'Textures','c.jpg'))
    skin.node_tree.nodes["c2"].image = bpy.data.images.load(filepath=os.path.join('./static_model',model,'Textures/c1.jpg'))
    skin.node_tree.nodes["g1"].image =bpy.data.images.load(filepath=os.path.join('./static_model',model,'Textures/g.jpg'))
    skin.node_tree.nodes["s1"].image =bpy.data.images.load(filepath=os.path.join('./static_model',model,'Textures/s.jpg'))
    skin.node_tree.nodes["n"].image =bpy.data.images.load(filepath=os.path.join('./static_model',model,'Textures/n.jpg'))
    #Initially Eye should be looking forward
    eye.rotation_euler[0]=math.radians(90)
    eye.rotation_euler[1]=math.radians(0)
    eye.rotation_euler[2]=math.radians(0)
    eye.keyframe_insert(data_path="rotation_euler", frame=0)

    #KEYFRAMING EYE AND HEAD VECTORS
    for x in range(1,len(head_x)):
        if x % 1000 == 0:
            print(x)
        #head1 = mathutils.Vector((head_x[x], head_y[x], head_z[x]))
        #head2= mathutils.Vector((head_x[x-1], head_y[x-1], head_z[x-1]))
        #head_rot = head1.rotation_difference(head2).to_euler()

        #Finding the angle between the vectors
        eye1 = mathutils.Vector((eye_x[x], eye_y[x], eye_z[x]))
        eye2= mathutils.Vector((eye_x[x-1], eye_y[x-1], eye_z[x-1]))
        
        l_rot = eye1.rotation_difference(eye2).to_euler() 
    
        #head.rotation_euler[0]=head.rotation_euler[0]-head_rot.x
        #head.rotation_euler[1]=head.rotation_euler[1]-head_rot.y
        #head.rotation_euler[2]=head.rotation_euler[2]-head_rot.z

        #Setting the rotation
        eye.rotation_euler[0]=eye.rotation_euler[0]-l_rot.x
        eye.rotation_euler[1]=eye.rotation_euler[1]+l_rot.z
        eye.rotation_euler[2]=eye.rotation_euler[2]-l_rot.y
    
        #Calculating eye lid values based on verticle eye gaze
        value=(math.degrees(eye.rotation_euler[0])-75)/50
        
        bpy.data.meshes['eye_lid'].shape_keys.key_blocks["Key 1"].value = value
        bpy.data.meshes['eye_lid'].shape_keys.key_blocks["Key 1"].keyframe_insert(data_path="value",frame=x)
        bpy.data.meshes['l'].shape_keys.key_blocks["Key 1"].value = value
        bpy.data.meshes['l'].shape_keys.key_blocks["Key 1"].keyframe_insert(data_path="value",frame=x)         
        bpy.data.meshes['u'].shape_keys.key_blocks["Key 1"].value = value
        bpy.data.meshes['u'].shape_keys.key_blocks["Key 1"].keyframe_insert(data_path="value",frame=x)
        bpy.data.meshes['Roundcube.000'].shape_keys.key_blocks["Pupil contract"].value = pupil[x]
        bpy.data.meshes['Roundcube.000'].shape_keys.key_blocks["Pupil contract"].keyframe_insert(data_path="value",frame=x)
        if value < 0.15:
            v=value
        else:
            v=value*1.2
        skin.node_tree.nodes["blink"].inputs[0].default_value = v
        skin.node_tree.nodes["blink"].inputs[0].keyframe_insert(data_path="default_value",frame=x)
        #head.keyframe_insert(data_path="rotation_euler", frame=x)
        eye.keyframe_insert(data_path="rotation_euler", frame=x)
        glass.hide_render=True

        #Glass keyframe
        glass.keyframe_insert(data_path="hide_render", frame=x)
        
        c1.hide_render=False
        c2.hide_render=True
        c3.hide_render=True
        sphere.hide_render=True

        ##Cornea key framing
        c1.keyframe_insert(data_path="hide_render", frame=x)
        c2.keyframe_insert(data_path="hide_render", frame=x)
        c3.keyframe_insert(data_path="hide_render", frame=x)
    
        camera.location[0] = args.camera_distance[0]	
        camera.location[1] = args.camera_distance[1]	
        camera.location[2] = args.camera_distance[2]	
	
        camera.rotation_euler[0] = math.radians(args.camera_elevation[0])	
        camera.rotation_euler[1] = math.radians(0)	
        camera.rotation_euler[2] = math.radians(args.camera_azimuthal[0])
        
        #camera target 1 mm vertically and horizontally
        
        #sclera_mat.node_tree.nodes["sclera_dark"].inputs[0].default_value = data['sclera_dark'][i]
        sclera_mat.node_tree.nodes["sclera_dark"].inputs[0].default_value =0.65
        #keyframing sclera color
        sclera_mat.node_tree.nodes["sclera_dark"].inputs[0].keyframe_insert(data_path="default_value",frame=x)
        #Camera location keyframing
        camera.keyframe_insert(data_path="location", frame=x)
        camera.keyframe_insert(data_path="rotation_euler", frame=x)
        #Camera target keyframing
        empty.keyframe_insert(data_path="location", frame=x)
        
    
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join('./GIW_Renderings',model,output,p[:-2],'seq.blend'))
    
    for i in range(1,len(head_x)):      

        s.frame_current = i
        filename=os.path.join('./GIW_Renderings',model,output,p[:-2],'synthetic' ,str(s.frame_current).zfill(4) + ".tif")
        if os.path.isfile(filename):
            print("skipped ",filename)
            continue
        else:
            print(i)

        iris_mat.node_tree.nodes["iris"].image = bpy.data.images.load(filepath=os.path.join(ir, args.iris_textures+'.png'))
        sclera_mat.node_tree.nodes["sclera"].image = bpy.data.images.load(filepath=os.path.join(scl, args.sclera_textures+'.png'))
        
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
    
    #eye plica black
    wet.node_tree.links.remove(wet.node_tree.nodes["Material Output"].inputs['Surface'].links[0])
    wet.node_tree.links.remove(wet.node_tree.nodes["Material Output"].inputs['Displacement'].links[0])
    
    #iris to green
    iris_mat.node_tree.links.new(iris_mat.node_tree.nodes["Emission"].outputs['Emission'], iris_mat.node_tree.nodes["Material Output"].inputs[0])
    #iris_mat.node_tree.links.remove(iris_mat.node_tree.nodes["Material Output"].inputs['Displacement'].links[0])
    
    retina=bpy.data.materials['Sclera material.001']
    #retina to red
    retina.node_tree.links.new(retina.node_tree.nodes["Emission"].outputs['Emission'], retina.node_tree.nodes["Material Output"].inputs[0])
    
    glass.hide_render = True
    sclera.hide_render = True
    bpy.data.objects["EyeWet.002"].hide_render = True
    bpy.data.objects["lower"].hide_render = True
    bpy.data.objects["upper"].hide_render = True
    bpy.data.objects["sclera_mask"].hide_render = False
    l1.hide_render = True
    l2.hide_render = True
        
        
        
        #RENDERING MASK IMAGES
    for i in range(1,len(head_x)):      
        glass.hide_render = True
        glass.keyframe_insert(data_path="hide_render", frame=i)
        
        frame=i
        s.frame_current = frame
        filename=os.path.join('./GIW_Renderings',model,output,p[:-2],'maskwith_skin' ,str(s.frame_current).zfill(4) + ".tif")
        if os.path.isfile(filename):
            print("skipped ",filename)
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
    
    
    for i in range(1,len(head_x)):      
        s.node_tree.links.new(s.node_tree.nodes["Render Layers"].outputs['Image'], s.node_tree.nodes["Composite"].inputs[0])
        frame=i
        s.frame_current = frame
        filename=os.path.join('./GIW_Renderings',model,output,p[:-2],'maskwithout_skin' ,str(s.frame_current).zfill(4) + ".tif")
        if os.path.isfile(os.path.join('./GIW_Renderings',model,output,p[:-2],'maskwithout_skin' ,str(s.frame_current).zfill(4) + ".tif")):
            print("skipped ",filename)
            continue
        else:
            print(frame)
            s.render.filepath = filename
                
            bpy.ops.render.render(  # {'dict': "override"},
                    # 'INVOKE_DEFAULT',
                    False,  # undo support
                    animation=False,
                    write_still=True)
        
    for x in range(1,len(head_x)):
        print(x)
        data(x)
    for x in gt.keys():
        gt[x]=gt[x][1:]
    pickle.dump( gt, open(os.path.join('./GIW_Renderings',model,output,p[:-2],model+'-pupil.p'), "wb" ))
    print("saved")
    

##Generating 3D data
def data(i):
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            override = bpy.context.copy()
            override['space_data'] = area.spaces.active
            override['region'] = area.regions[-1]
            override['area'] = area
            #space_data = area.spaces.active
    #camera data
    bpy.context.scene.frame_set(i)
    camera=bpy.data.objects['Camera']
    camera.select_set(True)
    bpy.ops.view3d.snap_cursor_to_selected(override)    
    loc=bpy.context.scene.cursor.location
    
    #camera matrix
    gt['cam_focal_length'].append(cam.lens)
    gt['cam_sensor_size'].append(cam.sensor_width)
    gt['camera_3d_center'].append(np.array(loc))
    gt['camera_distance'].append(math.sqrt(math.pow(loc[0],2)+math.pow(loc[2],2)+math.pow(-loc[1]-1,2)))
    camera.select_set(False)
    bpy.ops.object.select_all(action='DESELECT')
    
    #iris data
    iris=bpy.data.objects['EyeIris']
    iris.select_set(True)
    bpy.ops.view3d.snap_cursor_to_selected(override)
    iris_loc=bpy.context.scene.cursor.location
    gt['iris_loc'].append([iris_loc[0],iris_loc[1],iris_loc[2]])    
    gt['iris_rot'].append([iris.rotation_euler[0],iris.rotation_euler[1],iris.rotation_euler[2]])
    gt['iris_texture'].append(a[i])
    iris.select_set(False)
    
    #light data
    l1.select_set(True)
    bpy.ops.view3d.snap_cursor_to_selected(override)
    l1_loc=bpy.context.scene.cursor.location
    gt['light1_loc'].append([l1_loc[0],l1_loc[1],l1_loc[2]])    
    l1.select_set(False)
    bpy.ops.object.select_all(action='DESELECT')
    
    l2.select_set(True)
    bpy.ops.view3d.snap_cursor_to_selected(override)
    l2_loc=bpy.context.scene.cursor.location
    gt['light2_loc'].append([l2_loc[0],l2_loc[1],l2_loc[2]])    
    l2.select_set(False)
    bpy.ops.object.select_all(action='DESELECT')
    
    #target
    t=bpy.data.objects['Empty']
    t.select_set(True)
    bpy.ops.view3d.snap_cursor_to_selected(override)
    target=bpy.context.scene.cursor.location
    gt['target'].append(np.array(target))
    
    #camera angles
    c=np.array(target)-gt['camera_3d_center'][-1]
    c=c/np.sqrt(np.sum(c**2))
    gt['cam_az'].append(math.degrees(math.atan2(c[1],c[0]))+90)
    gt['cam_el'].append(math.degrees(math.atan2(c[2],np.sqrt(c[0]**2+c[1]**2))))
    
    #light1 angles
    c=np.array(target)-gt['light1_loc'][-1]
    c=c/np.sqrt(np.sum(c**2))
    gt['light1_az'].append(math.degrees(math.atan2(c[1],c[0]))+90)
    gt['light1_el'].append(math.degrees(math.atan2(c[2],np.sqrt(c[0]**2+c[1]**2))))
    
    #light2 angles
    c=np.array(target)-gt['light2_loc'][-1]
    c=c/np.sqrt(np.sum(c**2))
    gt['light2_az'].append(math.degrees(math.atan2(c[1],c[0]))+90)
    gt['light2_el'].append(math.degrees(math.atan2(c[2],np.sqrt(c[0]**2+c[1]**2))))
    
    t.select_set(False)
    bpy.ops.object.select_all(action='DESELECT')
    
    #Sclera data
    sclera=bpy.data.objects['Sclera']
    sclera.select_set(True)
    gt['sclera_dark'].append(sclera_mat.node_tree.nodes["sclera_dark"].inputs[0].default_value)
    gt['sclera'].append([sclera.rotation_euler[0],sclera.rotation_euler[1],sclera.rotation_euler[2]])
    gt['sclera_texture'].append(b[i])
    sclera.select_set(False)
    
    bpy.ops.object.select_all(action='DESELECT')
    
    eye=bpy.data.objects['Eye.Wetness']
    eye.select_set(True)
    bpy.ops.view3d.snap_cursor_to_selected(override)
    eye_loc=bpy.context.scene.cursor.location

    gt['eye_loc'].append([eye_loc[0],eye_loc[1],eye_loc[2]])    
    gt['gaze_angle_az'].append(math.degrees(eye.rotation_euler[2]))
    gt['gaze_angle_el'].append(math.degrees(eye.rotation_euler[0])-90)
    eye.select_set(False)
    bpy.ops.object.select_all(action='DESELECT')
    
    if bpy.data.objects["Sunglasses"].hide_render:
        gt['glasses'].append(0)
    else:
        gt['glasses'].append(1)
            
    
    pupil_s=bpy.data.meshes['Roundcube.000'].shape_keys.key_blocks["Pupil contract"].value
    gt['pupil'].append((pupil_s*3)+1)
    gt['eye_lid'].append(bpy.data.meshes['eye_lid'].shape_keys.key_blocks["Key 1"].value*100)
    
    
## Renders static data
def render():
    s = bpy.context.scene
    s.cycles.device = 'GPU'

    iris_mat=bpy.data.materials['Iris.000']
    sclera_mat=bpy.data.materials['Sclera material.000']
    ir = './Textures_eye/ir-textures/'
    scl = './Textures_eye/sclera/'
    sclera_mat.node_tree.nodes["op"].image =bpy.data.images.load(filepath='./Textures_eye/opacity.png')
    
    skin.node_tree.nodes["c1"].image =bpy.data.images.load(filepath= os.path.join('./static_model',model,'Textures','c.jpg'))
    skin.node_tree.nodes["c2"].image = bpy.data.images.load(filepath=os.path.join('./static_model',model,'Textures/c1.jpg'))
    
    skin.node_tree.nodes["g1"].image =bpy.data.images.load(filepath=os.path.join('./static_model',model,'Textures/g.jpg'))
    
    skin.node_tree.nodes["s1"].image =bpy.data.images.load(filepath=os.path.join('./static_model',model,'Textures/s.jpg'))
    
    skin.node_tree.nodes["n"].image =bpy.data.images.load(filepath=os.path.join('./static_model',model,'Textures/n.jpg'))


    for i in range(number):      
            
            
        s.frame_current = i
        filename=os.path.join('./static_model',model,output,'synthetic' ,str(s.frame_current).zfill(4) + ".tif")
        if os.path.isfile(filename):
            print("skipped ",filename)
            continue
        else:
            print(i)

        iris_mat.node_tree.nodes["iris"].image = bpy.data.images.load(filepath=os.path.join(ir, a[i]+'.png'))
        sclera_mat.node_tree.nodes["sclera"].image = bpy.data.images.load(filepath=os.path.join(scl, b[i]+'.png'))
        
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
    
    #eye plica black
    wet.node_tree.links.remove(wet.node_tree.nodes["Material Output"].inputs['Surface'].links[0])
    wet.node_tree.links.remove(wet.node_tree.nodes["Material Output"].inputs['Displacement'].links[0])
    
    #iris to green
    iris_mat.node_tree.links.new(iris_mat.node_tree.nodes["Emission"].outputs['Emission'], iris_mat.node_tree.nodes["Material Output"].inputs[0])
    #iris_mat.node_tree.links.remove(iris_mat.node_tree.nodes["Material Output"].inputs['Displacement'].links[0])
    
    retina=bpy.data.materials['Sclera material.001']
    #retina to red
    retina.node_tree.links.new(retina.node_tree.nodes["Emission"].outputs['Emission'], retina.node_tree.nodes["Material Output"].inputs[0])
    
    glass.hide_render = True
    sclera.hide_render = True
    bpy.data.objects["EyeWet.002"].hide_render = True
    bpy.data.objects["lower"].hide_render = True
    bpy.data.objects["upper"].hide_render = True
    bpy.data.objects["sclera_mask"].hide_render = False
    l1.hide_render = True
    l2.hide_render = True
        
        
        
        #RENDERING MASK IMAGES
    for i in range(number):      
        glass.hide_render = True
        glass.keyframe_insert(data_path="hide_render", frame=i)
        
        frame=i
        s.frame_current = frame
        filename=os.path.join('./static_model',model,output,'maskwith_skin' ,str(s.frame_current).zfill(4) + ".tif")
        if os.path.isfile(filename):
            print("skipped ",filename)
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
        s.node_tree.links.new(s.node_tree.nodes["Render Layers"].outputs['Image'], s.node_tree.nodes["Composite"].inputs[0])
        frame=i
        s.frame_current = frame
        filename=os.path.join('./static_model',model,output,'maskwithout_skin' ,str(s.frame_current).zfill(4) + ".tif")
        if os.path.isfile(os.path.join('./static_model',model,output,'depth' ,str(s.frame_current).zfill(4) + ".tif")):
            print("skipped ",filename)
            continue
        else:
            print(frame)
        
        s.render.filepath = filename
                
        bpy.ops.render.render(  # {'dict': "override"},
                    # 'INVOKE_DEFAULT',
                    False,  # undo support
                    animation=False,
                    write_still=True)
        s.node_tree.links.new(s.node_tree.nodes["ColorRamp"].outputs['Image'], s.node_tree.nodes["Composite"].inputs[0])
        s.render.filepath = os.path.join('./static_model',model,output,'depth' ,str(s.frame_current).zfill(4) + ".tif")
                
        bpy.ops.render.render(  # {'dict': "override"},
                    # 'INVOKE_DEFAULT',
                    False,  # undo support
                    animation=False,
                    write_still=True)


def main():
    
    if args.data_source=='pickle':
        print('LOADING DATAFROM GIVEN PICKLE FILE')
        data_pickle=pickle.load( open(os.path.join(args.data_source_path), "rb" ),encoding="latin1" )
        data_keyframe(data_pickle)
        if not os.path.isdir(os.path.join('static_model',model,output)):
            os.mkdir(os.path.join('static_model',model,output))
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join('static_model',model,output,'seq.blend'))
        for x in range(-1,number):
            print(x)
            data(x)
        for x in gt.keys():
            gt[x]=gt[x][1:]
        pickle.dump( gt, open(os.path.join('./static_model',model,output,model+'-pupil.p'), "wb" ))
        print("saved")
        render()
    elif args.data_source=='random':
        print('SETTING RANDOM VALUES')
        keyframe()
        if not os.path.isdir(os.path.join('static_model',model,output)):
            os.mkdir(os.path.join('static_model',model,output))
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join('static_model',model,output,'seq.blend'))
        for x in range(-1,number):
            print(x)
            data(x)
        for x in gt.keys():
            gt[x]=gt[x][1:]
        pickle.dump( gt, open(os.path.join('./static_model',model,output,model+'-pupil.p'), "wb" ))
        print("saved")
        render()
    elif args.data_source=='seq':
        print("RENDERING SEQUENTIAL DATA")
        seq_render()
        
    
main()
