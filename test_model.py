import numpy as np
import cv2
import time
from grabscreen import grab_screen
from directkeys import pressKey, releaseKey, A, W, S, D
from alexnet import alexnet
from getkeys import key_check
import os

WIDTH = 80
HEIGHT = 60
LR = 0.001
EPOCHS = 8

MODEL_NAME = f"San-self-driving-car {LR} {'alexnet'} {EPOCHS}-epochs.model"

def straight():
    pressKey(W)
    releaseKey(A)
    releaseKey(D)

def left():
    pressKey(A)
    releaseKey(W)
    releaseKey(D)

def right():
    pressKey(D)
    releaseKey(W)
    releaseKey(A)

model = alexnet(WIDTH, HEIGHT, 1)
model.load(MODEL_NAME)

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
            screen = cv2.resize(screen, (80,60))

            print(f"Loop took seconds {time.time()-last_time}")
            last_time = time.time()

            prediction = model.predict([screen.reshape(WIDTH, HEIGHT, 1)])[0]
            moves = list(np.around(prediction))
            print(moves, prediction)

            if moves == [1,0,0]:
                left()
            elif moves == [0,1,0]:
                straight()
            elif moves == [0,0,1]:
                right()
            
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
