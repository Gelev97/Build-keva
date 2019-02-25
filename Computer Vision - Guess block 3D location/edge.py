import numpy as np
import cv2 as cv

'''
Define Macro and Tunable Variable
'''
THRES = 1.2
BLUR_LEVEL = 3

CANNY_EDGE_LOWER_THRES = 100
CANNY_EDGE_UPPER_THRES = 120

'''
Main Function
'''
def find_edge(image_name):
    #threshold
    img = cv.imread(image_name)
    threshold = np.copy(img)

    #threshold out white and black background
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = (int(img[i,j,0]),int(img[i,j,1]),int(img[i,j,2]))
            pixel_avg = (pixel[0]+pixel[1]+pixel[2]) / 3.0
            if(pixel_avg == 0): pixel_avg = 1;
            pixel_normalized = (pixel[0]/pixel_avg,pixel[1]/pixel_avg,pixel[2]/pixel_avg)
            if(pixel_normalized[0] < THRES and pixel_normalized[1] < THRES and pixel_normalized[2] < THRES):
                threshold[i,j] = [0,0,0]

    #Blur the image and show the result
    blur = cv.blur(threshold, (BLUR_LEVEL, BLUR_LEVEL))

    #Detect edges using canny method and show the result
    edges = cv.Canny(blur, CANNY_EDGE_LOWER_THRES, CANNY_EDGE_UPPER_THRES)

    return [img, threshold, blur, edges]

'''
Draw Function
'''
def show_edge(img, threshold, blur, edges):
    edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    imstack_horizontal1 = np.hstack((img, threshold))
    imstack_horizontal2 = np.hstack((blur, edges))
    imstack = np.vstack((imstack_horizontal1,imstack_horizontal2))
    cv.imshow('stack', imstack)
    cv.waitKey(0)
    cv.destroyAllWindows()