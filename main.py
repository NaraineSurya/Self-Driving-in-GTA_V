import numpy as np
import cv2
import time
from grabscreen import grab_screen
from getkeys import key_check
import os


def keys_to_output(keys):
    output = [0,0,0]

    if 'A' in keys :
        output[0] = 1
    elif 'D' in keys :
        output[2] = 1
    else :
        output[1] = 1
    
    return output

file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print("File exists, loading previous data")
    training_data = list(np.load(file_name))
else :
    print("file does not exist , starting fresh")
    training_data = []


def main():

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    # last_time = time.time()
    while True:
        # bounding_box = {'top': 0, 'left': 0, 'width':  1920, 'height': 1100}
        screen = grab_screen(region=(0,0,1920,1100))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (80,60))
        # screen = np.expand_dims(screen, axis=-1)  # Expand dimensions to make it 3D
        keys = key_check()
        output = keys_to_output(keys)
        output = np.array(output)  # Convert output to numpy array
        print("Screen shape:", screen.shape, "Screen type:", type(screen))
        print("Output:", output)
        training_data.append([screen, output])
        # print(f"Loop took seconds {time.time()-last_time}")
        # last_time = time.time()

        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.save(file_name, training_data)

main()