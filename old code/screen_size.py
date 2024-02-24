import numpy as np
import cv2
from mss import mss
from PIL import ImageGrab
import time
from directkeys import releaseKey,pressKey , W , S ,A , D  
from statistics import mean
from numpy import ones,vstack
import pyautogui


def draw_lanes(img, lines, color =[0,255,255], thickness =3):
    # if fails go with default lines
    try :
        # finds max y value for a lane marker (which will not be same always)
        lanes = []
        ys = []
        for i in lanes :
            for ii in i:
                ys += [ii[i],ii[3]]
        min_y = min(ys)
        max_y = 600
        new_lines = []
        line_dict = []

        for idx,i in enumerate(lines):
            for xyxy in i:
                x_coords = (xyxy[0],xyxy[2])
                y_coords = (xyxy[1],xyxy[3])
                A = vstack([x_coords, ones(len(x_coords))]).T
                m,b = lstsq(A , y_coords)[0]

                x1 = (min_y-b) / m
                x2 = (max_y-b) / m

                line_dict[idx] = [m,b,[int(x1), min_y, int(x2), max_y]]
                new_lines.append([int(x1), min_y, int(x2), max_y])
            
            final_lanes = []

            for idx in line_dict:
                final_lanes_copy = final_lanes.copy()
                m = line_dict[idx][0]
                b = line_dict[idx][1]
                line = line_dict[idx][2]

                if len(final_lanes) == 0:
                    final_lanes[m] = [ [m, b, line] ]
                
                else :
                    found_copy = False

                    for other_ms in final_lanes_copy :

                        if not found_copy :
                            if abs(other_ms * 1.1) > abs(m) > abs(other_ms * 0.9):
                                   if abs(final_lanes_copy[other_ms][0][1] * 1.1) > abs(b) :
                                       final_lanes[other_ms].append([m, b, line])
                                       found_copy= True
                                       break
                            else :
                                final_lanes[m] = [ [m, b, line] ]
        line_counter = []

        for lanes in final_lanes:
            line_counter[lanes] = len(final_lanes[lanes])
        
        top_lanes = sorted(line_counter.items(), keys=lambda item: item[1]) [::-1][:2]

        lane1_id = top_lanes[0][0]
        lane2_id = top_lanes[1][0]

        def average_lane(lane_data):
            x1s = []
            y1s = []
            x2s = []
            y2s = []

            for data in lane_data :
                x1s.append(data[2][0])
                y1s.append(data[2][1])
                x2s.append(data[2][2])
                y2s.append(data[2][3])
            return int(mean(x1s)) , int(mean(x1s)), int(mean(y1s)), int(mean(y2s))
        
        l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
        l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])

        return [l1_x1, l1_y1, l1_x2, l1_y2] , [l2_x1, l2_y1, l2_x2, l2_y2], lane1_id, lane2_id
    except Exception as e:
        print(str(e))

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
    # Edges
    lines = cv2.HoughLinesP(processed_image, 1, np.pi/180, 180, np.array([]), 100, 5)
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


    
def main() :
    last_time = time.time()
    while(True):
        screen = np.array(ImageGrab.grab(bbox=(0,0,1920,1100)) )
        processed_img, original_img, m1, m2 = process_img(screen)
        # cv2.imshow('processed_image', processed_img)  # Display processed image
        cv2.imshow('original_image', cv2.cvtColor(original_img,cv2.COLOR_BGR2RGB))  # Display original image
        print(f"Loop took seconds {time.time()-last_time}")
        last_time = time.time()
        # cv2.imshow('window', cv2.cvtColor(np.array(screen), cv2.COLOR_BGR2RGB))

        if m1 < 0 and m2 < 0:
            right()
        elif m1 > 0 and m2 >0:
            left()
        else :
            straight()

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()