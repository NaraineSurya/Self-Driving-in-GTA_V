import numpy as np
import cv2
from mss import mss
import time
from directkeys import releaseKey,pressKey ,W ,S ,A ,D  
from numpy import ones,vstack
from numpy.linalg import lstsq 
from grabscreen import grab_screen
from draw_lanes import draw_lanes
from getkeys import key_check

# Finds the region of interest in the screen
        # Creating blank mask
        # Convert vertices to a list containing one array
        # Filling piels inside the polygon defined by vertices with fill color
        # returning pixels which are non zero 
def roi(img, vertices):
    mask = np.zeros_like(img)
    vertices = [vertices]
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

# Coverts the images into black and gray using canny()
    # convert to gray
    # edge detection 
    # blurring the image
    # region of interest
def process_img(original_image):
    img = np.array(original_image)
    processed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.Canny(processed_image, threshold1=200 , threshold2=300)
    processed_image = cv2.GaussianBlur(processed_image, (3,3), 0)
    vertices = np.array([[80,1000], [80,500], [500,300], [1200,300], [1840,500], [1840,1000]], np.int32)   
    processed_image = roi(processed_image, vertices)
    lines = cv2.HoughLinesP(processed_image, 1, np.pi/180, 180, np.array([]), 1000, 5)
    m1,m2 = 0,0
    try: 
        l1,l2,m1,m2 = draw_lanes(img,lines)
        cv2.line(img, (l1[0], l1[1]), (l1[2], l1[3]), (0,255,0), 30)
        cv2.line(img, (l2[0], l2[1]), (l2[2], l2[3]), (0,255,0), 30)
    except Exception as e:
        print (str(e))
        pass

    try:
        for coords in lines:
            coords = coords[0]
            try :
                cv2.line(processed_image, (coords[0], coords[1]), (coords[2], coords[3]), (255,255,255), 3)
            except Exception as e:
                print(str(e))
    except Exception as e :
        pass
    
    return processed_image, img, m1, m2


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
    releaseKey(A)
    releaseKey(W)

def slow():
    releaseKey(W)
    releaseKey(A)
    releaseKey(D)

for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)

def main():
    while True:
        bounding_box = {'top': 0, 'left': 0, 'width':  1920, 'height': 1100}
        last_time = time.time()
        sct = mss()
        sct_img = sct.grab(bounding_box)
        processed_img, original_img = process_img(sct_img)  # Separate processed and original images
        # cv2.imshow('processed_image', processed_img)  # Display processed image
        cv2.imshow('original_image', original_img)  # Display original image
        print(f"Loop took seconds {time.time()-last_time}")
        last_time = time.time()

        if m1 < 0 and m2 < 0:
            right()
        elif m1 > 0 and m2 >0:
            left()
        else :
            straight()

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break

main()