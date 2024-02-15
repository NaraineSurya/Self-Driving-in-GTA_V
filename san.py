import numpy as np
import cv2
from mss import mss
from PIL import Image
import time



bounding_box = {'top': 0, 'left': 0, 'width':  1920, 'height': 1100}
last_time = time.time()
sct = mss()

# Code provides more than 10 frames in a second  
while True:
    sct_img = sct.grab(bounding_box)
    cv2.imshow('screen', np.array(sct_img))
    print(f"Loop took seconds {time.time()-last_time}")
    last_time = time.time()
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break