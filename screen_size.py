import numpy as np
from PIL import ImageGrab
import cv2
import time

last_time = time.time()

def process_img(original_image):
    # img = np.array(original_image)
    processed_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.Canny(processed_image, threshold1=200 , threshold2=300)
    return processed_image

while(True):
    screen = np.array(ImageGrab.grab(bbox=(0,0,1920,1000)) )
    new_screen = process_img(screen)
    cv2.imshow('screen', new_screen)
    print(f"Loop took seconds {time.time()-last_time}")
    last_time = time.time()
    # cv2.imshow('window', cv2.cvtColor(np.array(screen), cv2.COLOR_BGR2RGB))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break