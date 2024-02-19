import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui


def draw_lanes(img, lines, color =[0,255,255], thickness =3):
    # if fails go with default lines
    try :
        # finds max y value for a lane marker (which will not be same always)
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
                A = vstak([x_coords,ones(len(x_coords))]).T
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

        return [l1_x1, l1_y1, l1_x2, l1_y2] , [l2_x1, l2_y1, l2_x2, l2_y2]
    except Exception as e:
        print(str(e))

def roi(img, vertices):
    # Creating blank mask
    mask = np.zeros_like(img)
    vertices = [vertices]  # Convert vertices to a list containing one array
    # Filling piels inside the polygon defined by vertices with fill color
    cv2.fillPoly(mask, vertices, 255)
    # returning pixels which are non zero  
    masked = cv2.bitwise_and(img, mask)
    return masked
    
# Coverts the images into black and gray using canny()
# def process_img(original_image):
#     img = np.array(original_image)
#     processed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     processed_image = cv2.Canny(processed_image, threshold1=200 , threshold2=300)
#     vertices = np.array([[80,1000], [80,500], [500,300], [1200,300], [1840,500], [1840,1000]])    #fvhbfvibvfhvbrfvuhvbfuhv
#     # vertices = np.array([[10,500], [10,300], [300,200], [500,200], [800,300], [800,500]])     sfvhf  vfjhv df jhd sfvfbdgbsvs
#     processed_image = roi(processed_image, vertices)
#     return processed_image

# Coverts the images into black and gray using canny()
def process_img(original_image):
    img = np.array(original_image)
    # convert to gray
    processed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edge detection 
    processed_image = cv2.Canny(processed_image, threshold1=200 , threshold2=300)
    # blurring the image
    processed_image = cv2.GaussianBlur(processed_image, (3,3), 0)
    vertices = np.array([[80,1000], [80,500], [500,300], [1200,300], [1840,500], [1840,1000]])    
    # region of interest   
    processed_image = roi(processed_image, vertices)
    lines = cv2.HoughLinesP(processed_image, 1, np.pi/180, 180, np.array([]), 100, 5)
    try: 
        l1,l2 = draw_lanes(original_image,lines)
        cv2.line(original_image, (l1[0], l1[1]), (l1[2], l1[3]), [0,255,0], 30)
        cv2.line(original_image, (l2[0], l2[1]), (l2[2], l2[3]), [0,255,0], 30)
    except Exception as e:
        print (str(e))
        pass

    try:
        for coords in lines:
            coords = coords[0]
            try :
                cv2.line(processed_image, (coords[0], coords[1]), (coords[2], coords[3]))
            except Exception as e:
                print(str(e))
    except Exception as e :
        pass
    
    return processed_image, original_image

    
def main() :
    last_time = time.time()
    while(True):
        screen = np.array(ImageGrab.grab(bbox=(0,0,1920,1100)) )
        processed_img, original_img = process_img(screen)
        # new_screen = process_img(screen)
        cv2.imshow('processed_image', processed_img)
        print(f"Loop took seconds {time.time()-last_time}")
        last_time = time.time()
        # cv2.imshow('window', cv2.cvtColor(np.array(screen), cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()