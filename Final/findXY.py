# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import math

def find(L):

    obj0 = np.load('./chessboard_params/obj0.npy')
    img0 = np.load('./chessboard_params/img0.npy')
    obj1 = np.load('./chessboard_params/obj1.npy')
    img1 = np.load('./chessboard_params/img1.npy')
    print(obj0, img0, obj1, img1)

    actd = 2.3
    d = math.sqrt((img1[0]-img0[0])**2+(img1[1]-img0[1])**2)
    r = d/actd

    x = L[0][0]
    y = L[0][1]

    actualx = (x-img0[0])/r
    actualy = (y-img0[1])/r

    return([(actualx+29, -actualy-18), L[1], L[2], L[3], L[4]])
