import numpy as np
import cv2 as cv
import sys

from edge import find_edge,show_edge
from block import detect_block

def main():
    if __name__ == '__main__':
        image_name = sys.argv[1]

    #find and show edges
    [img, threshold, blur, edges] = find_edge(image_name)
    # show_edge(img, threshold, blur, edges)

    #detect block and find their places
    detect_block(edges)

main()
