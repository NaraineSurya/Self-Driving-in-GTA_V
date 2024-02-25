import numpy as np
import cv2
import time
from grabscreen import grab_screen
from directkeys import pressKey, releaseKey, A, W, S, D
from alexnet import alexnet
from getkeys import key_check
import os

import tensorflow as tf


WIDTH = 160
HEIGHT = 120
LR = 0.001
EPOCHS = 8
t_time = 0.08


MODEL_NAME = f"San-self-driving-car_{LR}_alexnet_{EPOCHS}-epochs.model"

def straight():
    pressKey(W)
    releaseKey(A)
    releaseKey(D)

def left():
    pressKey(A) 
    pressKey(W)
    releaseKey(D)
    time.sleep(t_time)
    releaseKey(A)

def right():
    pressKey(D)
    pressKey(W)
    releaseKey(A)
    time.sleep(t_time)
    releaseKey(D)

model = alexnet(WIDTH, HEIGHT, 1)

model = tf.keras.models.load_model(MODEL_NAME)

def main():

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    last_time = time.time()

    paused = False 

    while True:
        if not paused:
            screen = grab_screen(region=(0,0,1920,1100))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (WIDTH,HEIGHT))

            print(f"Loop took seconds {time.time()-last_time}")
            last_time = time.time()


            prediction = model.predict([screen.reshape(-1, WIDTH, HEIGHT, 1)])[0]

            # moves = list(np.around(prediction))
            print(prediction)

            turn_thresh = 0.75
            fwd_thresh = 0.7

            if prediction[1] > fwd_thresh:
                straight()
            elif prediction[0] > turn_thresh:
                left()
            elif prediction[2] > turn_thresh:
                right()
            else :
                 straight()
            
        keys = key_check()

        if 'T' in keys :
            if paused :
                paused = False
                time.sleep(1)
            else :
                paused = True
                releaseKey(A)
                releaseKey(W)
                releaseKey(D)
                time.sleep(1)

main()
