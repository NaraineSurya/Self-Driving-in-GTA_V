import numpy as np
import cv2
from mss import mss
from PIL import Image
import time
from directkeys import releaseKey,pressKey , W , S ,A , D 

def draw_line(img, lines):
    for line in lines:
        coords = line[0]
        cv2.line(img, (coords[0],coords[1]), (coords[2],coords[3]), (255,255,255), 3)


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
    processed_image = cv2.GaussianBlur(processed_image, (3,3), 0)
    vertices = np.array([[80,1000], [80,500], [500,300], [1200,300], [1840,500], [1840,1000]])     #fvhbfvibvfhvbrfvuhvbfuhv
    # vertices = np.array([[10,500], [10,300], [300,200], [500,200], [800,300], [800,500]])     sfvhf  vfjhv df jhd sfvfbdgbsvs
    processed_image = roi(processed_image, vertices)
    lines = cv2.HoughLinesP(processed_image, 1, np.pi/180, 180, np.array([]), 100, 5)
    draw_line(processed_image, lines)
    return processed_image
# Countdown 
# for i in list(range(4))[:: -1]:
#     print(i+1)
#     time.sleep(1)

# Code provides more than 10 frames in a second  
def main() :
    while True:
        bounding_box = {'top': 0, 'left': 0, 'width':  1920, 'height': 1100}
        last_time = time.time()
        sct = mss()
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

main()