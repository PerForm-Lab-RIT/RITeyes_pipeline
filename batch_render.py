import os

trials = [  2, 3, 2, 3, 3, 3, 0, 3, 3, 4, 
            2, 4, 4, 4, 4, 4, 4, 4, 3, 4, 
            4, 4, 4
]
total = 0
for person in range(len(trials)):
    trial_num = trials[person]

    if (trial_num > 0):
        for i in range(1,trial_num+1):
            print('person: ' + str(person+1) + '   trial: ' + str(i))
            print('python rendering_script.py --head_model 2 --person_idx '+str(person+1) + ' --trial_idx ' + str(i) + ' --frame_cap 2')
            os.system('python rendering_script.py --head_model 2 --person_idx '+str(person+1) + ' --trial_idx ' + str(i) + ' --frame_cap 2')
            total += 1

print(total)