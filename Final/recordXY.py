# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import math

def caliberate(vs):
    img_counter = 0
    
    # perspective change
    perspective_changed = 0
    chesscol = 9
    chessrow = 6
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessrow*chesscol,3), np.float32)
    objp[:,:2] = np.mgrid[0:chesscol,0:chessrow].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    while True:
        # grab the current frame
        frame = vs.read()
     
        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        if frame is None:
            break
     
        # resize the frame, blur it, and convert it to the HSV
        # color space
        frame = imutils.resize(frame, width=1600)
        frame = frame[0:frame.shape[0]-300,330:1200]
        x = 466
        y = 84
        cv2.circle(frame, (x, y), 3, (0, 255, 0))
        h,  w = frame.shape[:2]
        copy = frame.copy()
        cv2.imshow("test", frame)
        cv2.imwrite("before_calibrate.jpg", frame)
        
        k = cv2.waitKey(1)
    
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (chesscol,chessrow),None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)
    
                # Draw and display the corners
                img = cv2.drawChessboardCorners(frame, (chesscol,chessrow), corners2,ret)
                cv2.imshow('img%d'%(img_counter),img)
                cv2.waitKey(50)
                cv2.imwrite("calibrate.jpg", img)
                img_counter += 1
    
    obj0 = objpoints[0].tolist()[0][:2]
    img0 = imgpoints[0].tolist()[0][0]
    obj1 = objpoints[0].tolist()[1][:2]
    img1 = imgpoints[0].tolist()[1][0]
    
    np.save("./chessboard_params/obj0", obj0)
    np.save("./chessboard_params/img0", img0)
    np.save("./chessboard_params/obj1", obj1)
    np.save("./chessboard_params/img1", img1)
     
    cv2.destroyAllWindows()
