import numpy as np
import cv2
import time
from grabscreen import grab_screen
from directkeys import pressKey, releaseKey, A, W, S, D
from getkeys import key_check
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from alexnet import AlexNet

WIDTH = 256
HEIGHT = 144
t_time = 0.08
LR = 0.001
EPOCHS = 9

# MODEL_NAME = f"GTA_V_{LR}_alexnet_{EPOCHS}_epochs.pth"  # Assuming you have saved the PyTorch model as "GTA_V_alexnet.model"
MODEL_NAME = 'models/model_9.pth'

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


# Load the PyTorch model
model = AlexNet()
model.load_state_dict(torch.load(MODEL_NAME))
# Move the model to CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# model.eval()  # Set model to evaluation mode

def main():

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    last_time = time.time()

    paused = False 

    while True:
        if not paused:
            screen = grab_screen(region=(0,0,1920,1100))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            screen = cv2.resize(screen, (WIDTH,HEIGHT))
            screen = ToTensor()(screen).unsqueeze(0).to(device)  # Convert to tensor and add batch dimension

            print(f"Loop took seconds {time.time()-last_time}")
            last_time = time.time()

            with torch.no_grad():
                prediction = F.softmax(model(screen), dim=1).squeeze().cpu().numpy()  # Perform prediction and convert to numpy
            
            prediction = np.array(prediction) * np.array([1, 1, 1, 1,  1,  1, 1, 1, 0.2])
            print(prediction)

            if np.argmax(prediction) == np.argmax(w):
                straight()
                print("st")
            elif np.argmax(prediction) == np.argmax(s):
                reverse()
                print("rev")
            elif np.argmax(prediction) == np.argmax(a):
                left()
                print("lef")
            elif np.argmax(prediction) == np.argmax(d):
                right()
                print("rig")
            elif np.argmax(prediction) == np.argmax(wa):
                fwd_left()
                print("fwdlef")
            elif np.argmax(prediction) == np.argmax(wd):
                fwd_right()
                print("fwdrig")
            elif np.argmax(prediction) == np.argmax(sa):
                rev_left()
                print("revlef")
            elif np.argmax(prediction) == np.argmax(sd):
                rev_right()
                print("revrig")
            elif np.argmax(prediction) == np.argmax(nk):
                straight()
                print("nokey")
            
            
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
