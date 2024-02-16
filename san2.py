import numpy as np
import cv2
from mss import mss
from PIL import Image
import time
import pyautogui
from directkeys import releaseKey,pressKey , W , S ,A , D 

# for i in list(range(4))[:: -1]:
#     print(i+1)
#     time.sleep(1)

# bounding_box = {'top': 0, 'left': 0, 'width':  1920, 'height': 1100}
# last_time = time.time()
# sct = mss()

def process_img(original_image):
    img = np.array(original_image)
    processed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.Canny(processed_image, threshold1=200 , threshold2=300)
    return processed_image


# Code provides more than 10 frames in a second  
while True:
    sct_img = sct.grab(bounding_box)
    new_screen = process_img(sct_img)
    # print("down")
    # pressKey(W)
    # time.sleep(2)
    # print("up")
    # releaseKey(W)
    # cv2.imshow('screen', np.array(sct_img))
    cv2.imshow('screen', new_screen)
    print(f"Loop took seconds {time.time()-last_time}")
    last_time = time.time()
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break