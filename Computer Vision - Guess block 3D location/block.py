import numpy as np
import cv2 as cv
import math

MIN_LINE_LENGTH = 15
MAX_LINE_GAP = 25
NUMBER_OF_INTERSECTION = 0 #The probability value for the hough line transform

MAX_LINE_LENGTH = 300
WIDTH_HEIGHT_DIST_MIN = 100.0
WIDTH_HEIGHT_DIST_MAX = 300.0

def detect_block(edges):
    #edge with three channel used to display
    edges_BGR = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    edges_BGR_copy = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    #find lines using hough lines transform with probability
    linesP = cv.HoughLinesP(edges, 1, np.pi / 180, NUMBER_OF_INTERSECTION, None, MIN_LINE_LENGTH, MAX_LINE_GAP)

    #declare a map which contains line with same angle that also with reasonable distance
    detect_rectangle = dict()

    #Hashmap all parallel lines
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]

            # find line angles
            Angle = round(math.atan2(l[3]- l[1], l[2] - l[0]) * 180.0 / np.pi);
            if (Angle in detect_rectangle):
                detect_rectangle[Angle].append(list(l))
            else:
                detect_rectangle[Angle] = [list(l)]
            cv.line(edges_BGR_copy, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1, cv.LINE_AA)

    #Delete parallel lines that not form keva block
    for Angle in detect_rectangle:
        if (len(detect_rectangle[Angle]) > 1):
            line_arr = detect_rectangle[Angle].copy()
            for line in line_arr:
                if (not line_distance(line, line_arr)):
                    #not witin the line distance we delete it
                    detect_rectangle[Angle].remove(line)

    # draw lines
    for Angle in detect_rectangle:
        if(len(detect_rectangle[Angle]) > 1):
            line_arr = detect_rectangle[Angle]
            edges_BGR_test = np.copy(edges_BGR)
            for line in line_arr:
                cv.line(edges_BGR_test, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 1, cv.LINE_AA)
            cv.imshow('stack', edges_BGR_test)
            cv.waitKey(0)
            cv.destroyAllWindows()

    imstack = np.hstack((edges_BGR, edges_BGR_copy))
    cv.imshow('stack', imstack)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return

def line_distance(line, lines):
    for line_to_compare in lines:
        dist_btw_lines_1 = math.sqrt((line[0] - line_to_compare[0])**2 + (line[1] - line_to_compare[1])**2)
        dist_btw_lines_2 = math.sqrt((line[2] - line_to_compare[2])**2 + (line[3] - line_to_compare[3])**2)
        average_distance = (dist_btw_lines_1 + dist_btw_lines_2)/2.0
        print(average_distance)
        if(average_distance >= WIDTH_HEIGHT_DIST_MIN and average_distance <= WIDTH_HEIGHT_DIST_MAX):
            return True
    return False