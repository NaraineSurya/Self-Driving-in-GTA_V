import numpy as np
import cv2
import time
from grabscreen import grab_screen
from directkeys import pressKey, releaseKey, A, W, S, D
from getkeys import key_check
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor

WIDTH = 480
HEIGHT = 360
t_time = 0.08

MODEL_NAME = f"GTA_V_alexnet.model"  # Assuming you have saved the PyTorch model as "GTA_V_alexnet.model"

w  = torch.tensor([1,0,0,0,0,0,0,0,0])
s  = torch.tensor([0,1,0,0,0,0,0,0,0])
a  = torch.tensor([0,0,1,0,0,0,0,0,0])
d  = torch.tensor([0,0,0,1,0,0,0,0,0])
wa = torch.tensor([0,0,0,0,1,0,0,0,0])
wd = torch.tensor([0,0,0,0,0,1,0,0,0])
sa = torch.tensor([0,0,0,0,0,0,1,0,0])
sd = torch.tensor([0,0,0,0,0,0,0,1,0])
nk = torch.tensor([0,0,0,0,0,0,0,0,1])

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


# Load the PyTorch model
model = torch.load(MODEL_NAME)
model.eval()  # Set model to evaluation mode

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
            screen = ToTensor()(screen).unsqueeze(0)  # Convert to tensor and add batch dimension

            print(f"Loop took seconds {time.time()-last_time}")
            last_time = time.time()

            with torch.no_grad():
                prediction = F.softmax(model(screen), dim=1).squeeze().numpy()  # Perform prediction and convert to numpy

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
