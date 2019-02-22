import numpy as np
import cv2 as cv
import sys

from edge import find_edge,show_edge
from block import detect_block, test_find_line, test_find_parallel_line, test_find_line_pair, test_find_rectangle

def main():
    if __name__ == '__main__':
        image_name = sys.argv[1]

    #find and show edges
    [img, threshold, blur, edges] = find_edge(image_name)
    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imshow('threshold', threshold)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imshow('blur', blur)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imshow('edges', edges)
    cv.waitKey(0)
    cv.destroyAllWindows()
    show_edge(img, threshold, blur, edges)

    #detect block and find their places
    #test_find_line(edges)
    #test_find_parallel_line(edges)
    #test_find_line_pair(edges)
    detect_block(edges)

main()
