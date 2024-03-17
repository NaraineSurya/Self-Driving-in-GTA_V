# create_training_data.py

import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os

w  = [1,0,0,0,0,0,0,0,0]
s  = [0,1,0,0,0,0,0,0,0]
a  = [0,0,1,0,0,0,0,0,0]
d  = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]

WIDTH = 256
HEIGHT = 144
WIDTH = 256
HEIGHT = 144

starting_value = 1

while True:
    file_name = 'Dataset/training_data_v{}.npy'.format(starting_value)

    if os.path.isfile(file_name):
        print('File exists, moving along',starting_value)
        starting_value += 1

    else:
        print('File does not exist, starting fresh!',starting_value)
        break


def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    '''
    output = [0,0,0,0,0,0,0,0,0]

    if 'W' in keys and 'A' in keys:
        output = wa
    elif 'W' in keys and 'D' in keys:
        output = wd
    elif 'S' in keys and 'A' in keys:
        output = sa
    elif 'S' in keys and 'D' in keys:
        output = sd
    elif 'W' in keys:
        output = w
    elif 'S' in keys:
        output = s
    elif 'A' in keys or 'left' in keys:
        output = a
    elif 'D' in keys or 'right' in keys: 
        output = d
    else:
        output = nk
    return output


def main(file_name, starting_value):
    file_name = file_name
    starting_value = starting_value
    training_data = []
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    last_time = time.time()
    paused = False
    print('STARTING!!!')
    while(True):
        
        if not paused:
            # windowed mode, this is 1920x1080, but you can change this to suit whatever res you're running.
            screen = grab_screen(region=(0,0,1600,900))
            last_time = time.time()
            # resize to something a bit more acceptable for a CNN
            screen = cv2.resize(screen, (WIDTH,HEIGHT))
            # run a color convert:
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            
            keys = key_check()
            output = np.array(keys_to_output(keys))
            training_data.append([screen,output])

            print(output)
            print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()

            if len(training_data) % 100 == 0:
                np.save(file_name,training_data)
                print('SAVED')
                print(len(training_data))
                
            if len(training_data) % 700 == 0:
                np.save(file_name,training_data)
                print('SAVED')
            if len(training_data) % 7500 == 0:
                np.save(file_name,training_data)
                print('SAVED')
                training_data = []
                starting_value += 1
                file_name = 'Dataset/training_data_v{}.npy'.format(starting_value)
                    
        keys = key_check()
        if 'B' in keys:
            if paused:
                paused = False
                for i in list(range(4))[::-1]:
                    print(i+1)
                    time.sleep(1)
                print('Resuming')
                time.sleep(0.5)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)


main(file_name, starting_value)