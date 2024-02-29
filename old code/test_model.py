import numpy as np
import cv2
import time
from grabscreen import grab_screen
from directkeys import pressKey, releaseKey, A, W, S, D
from alexnet import alexnet
from getkeys import key_check
import tensorflow as tf


WIDTH = 480
HEIGHT = 360
LR = 0.001
EPOCHS = 8
t_time = 0.08
OUTPUT = 9

MODEL_NAME = f"GTA_V_{LR}_alexnet_{EPOCHS}_epochs.model"

w  = [1,0,0,0,0,0,0,0,0]
s  = [0,1,0,0,0,0,0,0,0]
a  = [0,0,1,0,0,0,0,0,0]
d  = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]

def straight():
    pressKey(W)
    releaseKey(S)
    releaseKey(A)
    releaseKey(D)

def left():
    pressKey(A) 
    releaseKey(W)
    releaseKey(S)
    releaseKey(D)

def right():
    pressKey(D)
    releaseKey(W)
    releaseKey(S)
    releaseKey(A)

def reverse():
    pressKey(S)
    releaseKey(W)
    releaseKey(A)
    releaseKey(D)

def fwd_left():
    pressKey(W)
    pressKey(A) 
    releaseKey(D)
    releaseKey(S)

def fwd_right():
    pressKey(W)
    pressKey(D) 
    releaseKey(S)
    releaseKey(A)

def rev_left():
    pressKey(S)
    pressKey(A) 
    releaseKey(D)
    releaseKey(W)

def rev_right():
    pressKey(S)
    pressKey(D) 
    releaseKey(W)
    releaseKey(A)

def nokeys():
    releaseKey(W) 
    releaseKey(S)
    releaseKey(A)
    releaseKey(D)



model = alexnet(WIDTH, HEIGHT, OUTPUT)

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

            print(prediction)


            if np.argmax(prediction) == np.argmax(w):
                straight()
            elif np.argmax(prediction) == np.argmax(s):
                reverse()
            elif np.argmax(prediction) == np.argmax(a):
                left()
            elif np.argmax(prediction) == np.argmax(d):
                right()
            elif np.argmax(prediction) == np.argmax(wa):
                fwd_left()
            elif np.argmax(prediction) == np.argmax(wd):
                fwd_right()
            elif np.argmax(prediction) == np.argmax(sa):
                rev_left()
            elif np.argmax(prediction) == np.argmax(sd):
                rev_right()
            elif np.argmax(prediction) == np.argmax(nk):
                nokeys()
            
            
        keys = key_check()

        if 'B' in keys :
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
