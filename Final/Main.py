from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time
import math

import recordXY
import detect_block
import findXY
import Pipeline
import Kinematics

def Caliberate_camera(vs):
    recordXY.caliberate(vs)
    
def detect_block_grab(vs):
    while(1):
        frame = vs.read()
        frame = imutils.resize(frame, width=1600)
        frame = frame[0:frame.shape[0]-300,300:1200]
        cv2.imshow("frame", frame)
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if k%256 == 27:
            # ESC pressed
            print("picture chosen")
            break
    
    [img, threshold, blur, edges] = detect_block.find_edge(frame)
    #detect_block.show_edge(img, threshold, blur, edges)
    block = detect_block.detect(edges, img)
    detect_block.draw_result(img, block)
    return detect_block.output_coordinates(block)

def transfer_to_real(block_pixel_position):
    result = dict()
    for key in range(len(block_pixel_position)):
        result[key] = findXY.find(block_pixel_position[key])
    return result

def main():
    #vs = VideoStream(src=0).start()
    #time.sleep(2.0)

    #Caliberate_camera(vs)
    #block_pixel_position = detect_block_grab(vs)
    #block_real_position = transfer_to_real(block_pixel_position)
    #vs.stop()
    #print(block_pixel_position)
    #print(block_real_position)
    position = [(18.601009864224885, -21.158336929758949), 0, -37, 0, 0]
    traj= Kinematics.pipline_position_encoder([25.5, 17.3, 5], [25.5, 0, 5])
    print(traj)
    #traj= Kinematics.pipline_position_encoder([18.60, -21.583, 10, 0, 0, 0],[18.60, -11.583, 10, 0, 0, 0])
    #print(traj)
main()
    
    
    