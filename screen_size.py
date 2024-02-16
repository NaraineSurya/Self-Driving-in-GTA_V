import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui

last_time = time.time()

def roi(img, vertices):
    mask = np.zeros_like(img)
    vertices = [vertices]  # Convert vertices to a list containing one array
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked
    
# Coverts the images into black and gray using canny()
def process_img(original_image):
    img = np.array(original_image)
    processed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.Canny(processed_image, threshold1=200 , threshold2=300)
    vertices = np.array([[80,1000], [80,500], [500,300], [1200,300], [1840,500], [1840,1000]])    #fvhbfvibvfhvbrfvuhvbfuhv
    # vertices = np.array([[10,500], [10,300], [300,200], [500,200], [800,300], [800,500]])     sfvhf  vfjhv df jhd sfvfbdgbsvs
    processed_image = roi(processed_image, vertices)
    return processed_image
    
def main() :
    last_time = time.time()
    while(True):
        screen = np.array(ImageGrab.grab(bbox=(0,0,1920,1100)) )
        new_screen = process_img(screen)
        cv2.imshow('screen', new_screen)
        print(f"Loop took seconds {time.time()-last_time}")
        last_time = time.time()
        # cv2.imshow('window', cv2.cvtColor(np.array(screen), cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()