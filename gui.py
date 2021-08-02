'''
Copyright (c) 2021 RIT  
G. J. Diaz (PI), R. J. Bailey, J. B. Pelz
N. Nair, A. Romanenko, A. K. Chaudhary
'''

from tkinter import *
import subprocess
import os
def convert(x):
    if '-' in x:
        
        return float(x[1:])*-1
    else:
        return float(x[1:])
    
    
def main():
    if os.environ.get('DISPLAY','') == '':
        print('no display found. Using :0.0')
        os.environ.__setitem__('DISPLAY', ':0.0')
    master = Tk()
    
    
    #MODEL_ID
    frm = Frame(master,width=30, bd=30, relief='ridge')
    frm.grid(row=0, column=0)
    lab = Label(frm, width=30, text='Model_ID', anchor='w')
    lab1 = Label(frm, width=50, text='Select one of the head model from 1 to 24', anchor='w')
    lab1.pack()
    model_id = StringVar()
    ent = Entry(frm,textvariable=model_id)
    ent.insert(0, '1')
    lab.pack(side=LEFT)
    
    ent.pack(side=RIGHT, expand=YES, fill=X)
    
    #NUMBER
    frm = Frame(master,width=30, bd=30, relief='ridge')
    frm.grid(row=2, column=0)
    lab = Label(frm, width=30, text='Number', anchor='w')
    lab1 = Label(frm, width=50,text='Enter number of images to be rendered Eg: 100', anchor='w')
    lab1.pack()
    number = StringVar()
    ent = Entry(frm,textvariable=number)
    ent.insert(0, '100')
    lab.pack(side=LEFT)
    ent.pack(side=RIGHT, expand=YES, fill=X)
    
    #DATA_SOURCE
    frm = Frame(master,width=30, bd=30, relief='ridge')
    frm.grid(row=4, column=0)
    data_source = StringVar()
    data_source.set("random")
    lab1 = Label(frm, width=50,text='Uniform distribution/ Sequential/ Data from file ', anchor='w')
    lab1.pack()
    b_dict = {'Random':'random', 'Sequential':'seq', 'Read from file':'pickle'}
    lab = Label(frm, width=15, text='Number', anchor='w')
    for key,value in b_dict.items():
        b_dict[key] = Radiobutton(frm, text=key,width=15)
        b_dict[key].config(indicatoron=1, variable=data_source, value=value)
        b_dict[key].pack(side='left')
        
    #DATA_SOURCE_PATH
    frm = Frame(master,width=30, bd=30, relief='ridge')
    frm.grid(row=6, column=0)
    lab = Label(frm, width=30, text='Data Path', anchor='w')
    lab1 = Label(frm, width=50, text='Enter pickle file path with gaze info.', anchor='w')
    lab1.pack()
    data_path = StringVar()
    ent = Entry(frm,textvariable=data_path)
    ent.insert(0, 'PrIdx_1_TrIdx_1.p')
    lab.pack(side=LEFT)
    ent.pack(side=RIGHT, expand=YES, fill=X)
    
    #LIGHT-1_LOCATION
    frm = Frame(master,width=30, bd=30, relief='ridge')
    frm.grid(row=8, column=0)
    lab = Label(frm, width=30, text='Light Source Location 1:', anchor='w')
    lab1 = Label(frm, width=50, text='Light-1 location x,y,z Eg: -0.5,0.5,0.5', anchor='w')
    lab1.pack()
    light_loc_1 = StringVar()
    ent = Entry(frm,textvariable=light_loc_1)
    ent.insert(0,'-1,1,0')
    lab.pack(side=LEFT)
    ent.pack(side=RIGHT, expand=YES, fill=X)
    
    #LIGHT-2_LOCATION
    frm = Frame(master,width=30, bd=30, relief='ridge')
    frm.grid(row=10, column=0)
    lab = Label(frm, width=30, text='Light Source Location 2:', anchor='w')
    light_loc_2 = StringVar()
    ent = Entry(frm,textvariable=light_loc_2)
    lab1 = Label(frm, width=50, text='Light-2 location x,y,z Eg: -0.5,-0.5,0.5', anchor='w')
    lab1.pack()
    ent.insert(0, '-1,0.5,0')
    lab.pack(side=LEFT)
    ent.pack(side=RIGHT, expand=YES, fill=X)
    
    #LIGHT-1_ENERGY
    frm = Frame(master,width=30, bd=30, relief='ridge')
    frm.grid(row=12, column=0)
    lab = Label(frm, width=30, text='Light Engery 1:', anchor='w')
    light_energy_1 = StringVar()
    ent = Entry(frm,textvariable=light_energy_1)
    lab1 = Label(frm, width=50, text='Light-1 energy Eg: 100', anchor='w')
    lab1.pack()
    ent.insert(0, '100')
    lab.pack(side=LEFT)
    ent.pack(side=RIGHT, expand=YES, fill=X)
    
    #START END FRAME
    frm = Frame(master,width=30, bd=30, relief='ridge')
    frm.grid(row=13, column=0)
    lab = Label(frm, width=30, text='Start,End Frame', anchor='w')
    time = StringVar()
    ent = Entry(frm,textvariable=time)
    lab1 = Label(frm, width=50, text='GIW start and end frame: 10,1000', anchor='w')
    lab1.pack()
    ent.insert(0, '1,2000')
    lab.pack(side=LEFT)
    ent.pack(side=RIGHT, expand=YES, fill=X)
    
    #LIGHT-2_ENERGY
    frm = Frame(master,width=30, bd=30, relief='ridge')
    frm.grid(row=0, column=2)
    lab = Label(frm, width=30, text='Light Engery 2:', anchor='w')
    light_energy_2 = StringVar()
    ent = Entry(frm,textvariable=light_energy_2)
    lab1 = Label(frm, width=50, text='Light Energy 2 Eg: 100', anchor='w')
    lab1.pack()
    ent.insert(0, '100')
    lab.pack(side=LEFT)
    ent.pack(side=RIGHT, expand=YES, fill=X)
    
    #CAMERA_FOCAL LENGTH
    frm = Frame(master,width=30, bd=30, relief='ridge')
    frm.grid(row=2, column=2)
    lab = Label(frm, width=30, text='Camera Focal Length', anchor='w')
    focal_length = StringVar()
    ent = Entry(frm,textvariable=focal_length)
    lab1 = Label(frm, width=50, text='Camera Focal Eg: 50', anchor='w')
    lab1.pack()
    ent.insert(0, '50')
    lab.pack(side=LEFT)
    ent.pack(side=RIGHT, expand=YES, fill=X)
    
    #CAMERA DISTANCE
    frm = Frame(master,width=30, bd=30, relief='ridge')
    frm.grid(row=4, column=2)
    lab = Label(frm, width=30, text='Camera Distance Range', anchor='w')
    cam_dist = StringVar()
    ent = Entry(frm,textvariable=cam_dist)
    lab1 = Label(frm, width=50, text='Camera Distance Range Eg: 4,5', anchor='w')
    lab1.pack()
    ent.insert(0, '4,5')
    lab.pack(side=LEFT)
    ent.pack(side=RIGHT, expand=YES, fill=X)
    
    #CAMERA AZHIMUTHAL RANGE
    frm = Frame(master,width=30, bd=30, relief='ridge')
    frm.grid(row=6, column=2)
    lab = Label(frm, width=30, text='Camera Azhimuthal Range', anchor='w')
    cam_azhimuthal = StringVar()
    ent = Entry(frm,textvariable=cam_azhimuthal)
    lab1 = Label(frm, width=50, text='Camera Azhimuthal Range Eg: from,to', anchor='w')
    lab1.pack()
    ent.insert(0, '-55,-65')
    lab.pack(side=LEFT)
    ent.pack(side=RIGHT, expand=YES, fill=X)
    
    #CAMERA ELEVATION RANGE
    frm = Frame(master,width=30, bd=30, relief='ridge')
    frm.grid(row=8, column=2)
    lab = Label(frm, width=30, text='Camera Elevation Range', anchor='w')
    cam_elevation = StringVar()
    ent = Entry(frm,textvariable=cam_elevation)
    lab1 = Label(frm, width=50, text='Camera Elevation Range Eg: from,to', anchor='w')
    lab1.pack()
    ent.insert(0, '20,30')
    lab.pack(side=LEFT)
    ent.pack(side=RIGHT, expand=YES, fill=X)
    
    #PUPIL SIZE 
    frm = Frame(master,width=30, bd=30, relief='ridge')
    frm.grid(row=10, column=2)
    lab = Label(frm, width=30, text='Pupil Size Range', anchor='w')
    pupil_size = StringVar()
    ent = Entry(frm,textvariable=pupil_size)
    lab1 = Label(frm, width=50, text='Pupil Radius in mm from 1- 4 mm Eg: 4,5', anchor='w')
    lab1.pack()
    ent.insert(0, '1,3')
    lab.pack(side=LEFT)
    ent.pack(side=RIGHT, expand=YES, fill=X)
    
    #CORNEA TYPE
    frm = Frame(master,width=30, bd=30, relief='ridge')
    frm.grid(row=12, column=2)
    lab = Label(frm, width=30, text='Cornea Type', anchor='w')
    cornea = StringVar()
    ent = Entry(frm,textvariable=cornea)
    lab1 = Label(frm, width=50, text='Select the corneas 0,1,2,sphere ', anchor='w')
    lab1.pack()
    ent.insert(0, '0')
    lab.pack(side=LEFT)
    ent.pack(side=RIGHT, expand=YES, fill=X)
    
    #GPU IDS
    frm = Frame(master,width=30, bd=30, relief='ridge')
    frm.grid(row=13, column=2)
    lab = Label(frm, width=30, text='GPU IDS', anchor='w')
    gpu = StringVar()
    ent = Entry(frm,textvariable=gpu)
    lab1 = Label(frm, width=50, text='Gpus to be used 0,1,2,3 ', anchor='w')
    lab1.pack()
    ent.insert(0, '0')
    lab.pack(side=LEFT)
    ent.pack(side=RIGHT, expand=YES, fill=X)
    
    #IRIS TEXTURES
    frm = Frame(master,width=30, bd=30, relief='ridge')
    frm.grid(row=0, column=4)
    lab = Label(frm, width=30, text='Iris Textures', anchor='w')
    iris_tex = StringVar()
    ent = Entry(frm,textvariable=iris_tex)
    lab1 = Label(frm, width=50, text='Select the iris textures  Eg: 1,2,3,4,5', anchor='w')
    lab1.pack()
    ent.insert(0, '1')
    lab.pack(side=LEFT)
    ent.pack(side=RIGHT, expand=YES, fill=X)
    
    #SCLERA TEXTURES
    frm = Frame(master,width=30, bd=30, relief='ridge')
    frm.grid(row=2, column=4)
    lab = Label(frm, width=30, text='Sclera Textures', anchor='w')
    sclera_tex = StringVar()
    ent = Entry(frm,textvariable=sclera_tex)
    lab1 = Label(frm, width=50, text='Select the sclera textures  Eg: 1,2', anchor='w')
    lab1.pack()
    ent.insert(0, '1')
    lab.pack(side=LEFT)
    ent.pack(side=RIGHT, expand=YES, fill=X)
    
    #GLASS PERCENTAGE
    frm = Frame(master,width=30, bd=30, relief='ridge')
    frm.grid(row=4, column=4)
    lab = Label(frm, width=30, text='Glass Frequency', anchor='w')
    glass = StringVar()
    ent = Entry(frm,textvariable=glass)
    lab1 = Label(frm, width=50, text='Percentage of images with glasses Eg: 50', anchor='w')
    lab1.pack()
    ent.insert(0, '0')
    lab.pack(side=LEFT)
    ent.pack(side=RIGHT, expand=YES, fill=X)
    
    #EYE ELEVATION
    frm = Frame(master,width=30, bd=30, relief='ridge')
    frm.grid(row=6, column=4)
    lab = Label(frm, width=30, text='Eye Elevation Range', anchor='w')
    eye_elevation = StringVar()
    ent = Entry(frm,textvariable=eye_elevation)
    lab1 = Label(frm, width=50, text='Eye Elevation Range from,to Eg: -20,20', anchor='w')
    lab1.pack()
    ent.insert(0, '-20,20')
    lab.pack(side=LEFT)
    ent.pack(side=RIGHT, expand=YES, fill=X)
    
    #EYE AZHIMUTHAL
    frm = Frame(master,width=30, bd=30, relief='ridge')
    frm.grid(row=8, column=4)
    lab = Label(frm, width=30, text='Eye Azhimuthal Range', anchor='w')
    eye_azhimuthal = StringVar()
    ent = Entry(frm,textvariable=eye_azhimuthal)
    lab1 = Label(frm, width=50, text='Eye Azhimuthal Range from,to Eg: -20,20', anchor='w')
    lab1.pack()
    ent.insert(0, '-20,20')
    lab.pack(side=LEFT)
    ent.pack(side=RIGHT, expand=YES, fill=X)
    
    #OUTPUT FILE LOCATION
    frm = Frame(master,width=30, bd=30, relief='ridge')
    frm.grid(row=10, column=4)
    lab = Label(frm, width=30, text='Output File location', anchor='w')
    output_file = StringVar()
    ent = Entry(frm,textvariable=output_file)
    lab1 = Label(frm, width=50, text='Storage location ./rendered_images/$model_id$/', anchor='w')
    lab1.pack()
    ent.insert(0, 'test')
    output_file.set("test")
    lab.pack(side=LEFT)
    ent.pack(side=RIGHT, expand=YES, fill=X)
    
    # WITH/WITHOUT CORNEA
    frm = Frame(master,width=30, bd=30, relief='ridge')
    frm.grid(row=12, column=4)
    cor = StringVar()
    cor.set(1)
    lab1 = Label(frm, width=50, text='With and without cornea', anchor='w')
    lab1.pack()
    b_dict = {'With cornea':1, 'Without cornea':0}

    for key,value in b_dict.items():
        b_dict[key] = Radiobutton(frm, text=key, bd=4, width=20)
        b_dict[key].config(indicatoron=1, variable=cor, value=value)
        b_dict[key].pack(side='left')
        
    #WITH/WITHOUT REFLECTION   
    frm = Frame(master,width=30, bd=30, relief='ridge')
    frm.grid(row=14, column=4)
    refl = StringVar()
    refl.set(1)
    lab1 = Label(frm, width=50, text='With and without reflection', anchor='w')
    lab1.pack()
    b_dict = {'With reflection':1, 'Without reflection':0}

    for key,value in b_dict.items():
        b_dict[key] = Radiobutton(frm, text=key, bd=4, width=20)
        b_dict[key].config(indicatoron=1, variable=refl, value=value)
        b_dict[key].pack(side='left')
        
        
        
    frm = Frame(master,width=30, bd=30, relief='ridge')
    frm.grid(row=18, column=2)   
    okButton = Button(frm, text="OK", command=master.destroy)
    okButton.pack(side=BOTTOM)
    master.mainloop()
    start=time.get().split(',')[0]
    end=time.get().split(',')[1]
    m=model_id.get()
    m=m.split(',')
    for model_id in m:
        model="static_model/"+ model_id +"/"+model_id+"-pupil.blend"
        if data_source.get()== 'seq':
            if not os.path.isdir(os.path.join(os.getcwd(),"static_model",model_id,str(output_file.get()),data_path.get()[:-2])):
            
                os.makedirs(os.path.join(os.getcwd(),"static_model",model_id,str(output_file.get()),data_path.get()[:-2]))
            f = open(os.path.join(os.getcwd(),"static_model",model_id,str(output_file.get()),data_path.get()[:-2],"log.txt"), "w+")
            f.write("./blender-2.82a-linux64/blender"+" -b"+" "+model+" -P"+ "RIT-Eyes.py"+" --"+" --model_id"+" "+model_id+" --number"+" " +str(number.get())+" --data_source"+" "+data_source.get()+" --data_source_path"+" "+data_path.get()+" --light_1_loc"+" ,"+str(light_loc_1.get())+" --light_2_loc"+" ,"+str(light_loc_2.get())+" --light_1_energy"+" "+str(light_energy_1.get())+" --light_2_energy"+" "+str(light_energy_2.get())+" --camera_focal_length"+" "+str(focal_length.get())+" --camera_distance"+" ,"+str(cam_dist.get())+" --camera_azimuthal"+" ,"+str(cam_azhimuthal.get())+" --camera_elevation"+" ,"+str(cam_elevation.get())+" --pupil"+" ,"+str(pupil_size.get())+" --cornea"+" "+str(cornea.get())+" --iris_textures"+" "+iris_tex.get()+" --sclera_textures"+" "+sclera_tex.get()+" --glass"+" "+str(glass.get())+" --eye_elevation"+" ,"+str(eye_elevation.get())+" --eye_azimuthal"+" ,"+str(eye_azhimuthal.get())+" --no_cornea"+" "+str(cor.get())+" --no_reflection"+" "+str(refl.get())+" --output_file"+" "+str(output_file.get()) +" --start"+" "+str(start) +" --end"+" "+str(end) +" --gpu"+" "+str(gpu.get()))
            f.close()
        else:
            if not os.path.isdir(os.path.join(os.getcwd(),"static_model",model_id,str(output_file.get()),data_path.get()[:-2])):
            
                os.makedirs(os.path.join(os.getcwd(),"static_model",model_id,str(output_file.get()),data_path.get()[:-2]))
            f = open(os.path.join(os.getcwd(),"static_model",model_id,str(output_file.get()),"log.txt"), "w")
            f.close()
        
        #./blender-2.82a-linux64/blender -b $model -P static_render.py -- --model_id ${model_id} 
        subprocess.call(["./blender-2.82a-linux64/blender", "-b", model,"-P", "RIT-Eyes.py","--","--model_id",model_id,"--number", number.get(),"--data_source",data_source.get(),"--data_source_path",data_path.get(),"--light_1_loc",','+light_loc_1.get(),"--light_2_loc",','+light_loc_2.get(),"--light_1_energy",light_energy_1.get(),"--light_2_energy",light_energy_2.get(),"--camera_focal_length",focal_length.get(),"--camera_distance",','+cam_dist.get(),"--camera_azimuthal",','+cam_azhimuthal.get(),"--camera_elevation",','+cam_elevation.get(),"--pupil",','+pupil_size.get(),"--cornea",','+cornea.get(),"--iris_textures",','+iris_tex.get(),"--sclera_textures",','+sclera_tex.get(),"--glass",glass.get(),"--eye_elevation",','+eye_elevation.get(),"--eye_azimuthal",','+eye_azhimuthal.get(),"--no_cornea",cor.get(),"--no_reflection",str(refl.get()),"--output_file",str(output_file.get())])

if __name__ == '__main__':
    main()
