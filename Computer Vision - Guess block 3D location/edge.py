import numpy as np
import cv2 as cv

THRES = 160;

#threshold
img = cv.imread('test.JPG')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Find countours
rett, thresh = cv.threshold(gray, THRES , 255, cv.THRESH_BINARY)
cv.imshow('thresh',thresh)
cv.waitKey(0)
cv.destroyAllWindows()
thresh, contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img, contours, -1, (0,255,0), 3)
cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()