import numpy as np
from PIL import ImageGrab
import cv2
import time

last_time = time.time()

while(True):
    screen = np.array(ImageGrab.grab(bbox=(0,0,1920,1000)) )
    print(f"Loop took seconds {time.time()-last_time}")
    last_time = time.time()
    cv2.imshow('window', cv2.cvtColor(np.array(screen), cv2.COLOR_BGR2RGB))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break