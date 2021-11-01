@ECHO OFF
FOR /f %%p in ('where python') do SET PYTHONPATH=%%p

echo Just hit enter to use the default value inside the []
SET /P model_id=    "Enter Model Id (which head is being used) [1]: " || SET model_id=1
SET /P person_idx=  "Enter Person Index (which person we are using for the data) [0]: " || SET person_idx=0
SET /P trial_idx=   "Enter Trial Index (which trial we are using for the data) [0]: " || SET trial_idx=0
SET /P iris_idx=    "Enter Iris Index (which iris texture we are using for rendering) [1]: " || SET iris_idx=1
SET /P start_frame= "Enter Start Frame (the first frame we render from in the world video) [0]: " || SET start_frame=0
SET /P end_frame=   "Enter End Frame (the last frame we render from in the world video, 0 means there is no limit)  [0]: " || SET end_frame=0
SET /P mode=        "Enter Mode (changes render node to either binocular, each eye, or observe, from behind the eyes) [binocular]: " || SET mode=binocular
SET /P framerate=   "Enter Frame Rate (120 fps animation is True, 30 fps animation is False) [True]: " || SET mode=True
echo -----------------------------------------------------------------------
python RIT-Eyes_full_binocular_System.py --model_id %model_id% --person_idx %person_idx% --trial_idx %trial_idx% --iris_idx %iris_idx% --start_frame %start_frame% --end_frame %end_frame% --mode %mode%

echo Pipeline finished rendering
pause