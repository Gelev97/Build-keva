import numpy as np
import cv2 as cv
import math

'''
Define Macro and Tunable Variable
'''

MIN_LINE_LENGTH = 15
MAX_LINE_GAP = 30
NUMBER_OF_INTERSECTION = 20# The probability value for the hough line transform

ANGLE_GAP = 5# Degree difference, we make angles same wihtin these gap
PERPENDICULAR_THRES = 0.5# Threshold tolerance for perpendiculars
INTERSECT_THRES = 20# Distance threshold to determine if two line intersect

MAX_LINE_LENGTH = 300
WIDTH_HEIGHT_DIST_MIN = 20.0
WIDTH_HEIGHT_DIST_MAX = 400.0


'''
Find Parallel line
'''

def detect_angle_difference(angle, angle_arr):
    for angle_compare in angle_arr:
        if(abs(angle - angle_compare[1]) > ANGLE_GAP):
            return False
    return True

def angle_approximation(lines):
    # find parallel line
    parallel_line_group = []
    line_index = 0
    being_addded_flag = 0
    while(line_index < len(lines)):
        # calculate angle of each line
        line = lines[line_index]
        angle = round(math.atan2(line[3] - line[1], line[2] - line[0]) * 180.0 / np.pi)
        # find corresponding group
        for group_index in range(0, len(parallel_line_group)):
            group = parallel_line_group[group_index]
            if(len(group) == 0): continue
            if(detect_angle_difference(angle, group)):
                being_addded_flag = 1
                parallel_line_group[group_index].append((lines[line_index],angle))
        # Create new group if no match
        if(being_addded_flag == 0):
            parallel_line_group.append([(lines[line_index],angle)])
        line_index += 1
        being_addded_flag = 0
    # [[(line, angle), (line, angle)], [(line, angle), (line, angle)], ...]
    return parallel_line_group

'''
Find Perpendicular line
'''

def distance(p0, p1, p2, p3):
    return math.sqrt((p0 - p1) ** 2 + (p2 - p3) ** 2)

def two_line_close_enough(line,line_to_compare):
    (line_x_1, line_y_1) = (line[0], line[1])
    (line_x_2, line_y_2) = (line[2], line[3])
    (line_to_compare_x_1, line_to_compare_y_1) = (line_to_compare[0], line_to_compare[1])
    (line_to_compare_x_2, line_to_compare_y_2) = (line_to_compare[2], line_to_compare[3])
    #Calculate distance between them
    if(distance(line_x_1, line_to_compare_x_1, line_y_1, line_to_compare_y_1) < INTERSECT_THRES):
        return True
    if(distance(line_x_1, line_to_compare_x_2, line_y_1, line_to_compare_y_2) < INTERSECT_THRES):
        return True
    if(distance(line_x_2, line_to_compare_x_1, line_y_2, line_to_compare_y_1) < INTERSECT_THRES):
        return True
    if(distance(line_x_2, line_to_compare_x_2, line_y_2, line_to_compare_y_2) < INTERSECT_THRES):
        return True
    return False

def perpendicular_approximation(line_index_dict, lines):
    perpendicular_line_group = dict()
    for unique_index in line_index_dict:
        #create a new dict for this line
        perpendicular_line_group[unique_index] = []
        slope_arr = [0,0]
        line = line_index_dict[unique_index]
        for line_to_compare in lines:
            if (len(set(line_to_compare).intersection(line)) == len(set(line))): continue
            # avoid divide by 0
            if (line[0] - line[2] == 0): line[2] = line[2] + 1
            if (line_to_compare[0] - line_to_compare[2] == 0): line_to_compare[2] = line_to_compare[2] + 1
            slope_arr[0] = (line[1] - line[3]) / (line[0] - line[2])
            slope_arr[1] = (line_to_compare[1] - line_to_compare[3]) / (line_to_compare[0] - line_to_compare[2])
            threshold_value = slope_arr[0] * slope_arr[1]
            if(threshold_value > -1 - PERPENDICULAR_THRES and threshold_value < -1 + PERPENDICULAR_THRES):
               if(two_line_close_enough(line,line_to_compare)):
                   perpendicular_line_group[unique_index].append(line_to_compare)
    return perpendicular_line_group


'''
Find Parallel line distance
'''
def line_distance(line, line_angle_arr):
    # find distance between parallel lines
    for pair in line_angle_arr:
        line_to_compare = pair[0]
        # do not calculate distance with itself
        if(len(set(line_to_compare).intersection(line)) == len(set(line))): continue
        slope_arr = [0,0]
        intercept_arr = [0,0]
        # avoid divide by 0
        if(line[0] - line[2] == 0): line[2] = line[2] + 1
        if (line_to_compare[0] - line_to_compare[2] == 0): line_to_compare[2] = line_to_compare[2] + 1
        slope_arr[0] = (line[1] - line[3])/(line[0] - line[2])
        slope_arr[1] = (line_to_compare[1] - line_to_compare[3])/(line_to_compare[0] - line_to_compare[2])
        slope = (slope_arr[0] + slope_arr[1]) / 2.0
        intercept_arr[0] = line[1] - slope_arr[0]*line[0]
        intercept_arr[1] = line_to_compare[1] - slope_arr[1]*line_to_compare[0]
        distance = abs(intercept_arr[0] - intercept_arr[1])/math.sqrt(1+slope**2)
        if(distance >= WIDTH_HEIGHT_DIST_MIN and distance <= WIDTH_HEIGHT_DIST_MAX):
            return True
    return False

'''
Find Rectangles
'''

'''
Main Function
'''
def detect_block(edges):
    # edge with three channel used to display
    edges_BGR = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    edges_BGR_modified = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    # find lines using hough lines transform with probability
    linesP = cv.HoughLinesP(edges, 1, np.pi / 180, NUMBER_OF_INTERSECTION, None, MIN_LINE_LENGTH, MAX_LINE_GAP)

    # array that contains all lines
    lines = []

    # A dictionary that maps every line with an unique index
    line_index_dict = dict()
    index_dict = 0

    # Hashmap all lines with its angle
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            lines.append(list(l))
            line_index_dict[index_dict] = l
            index_dict += 1
            cv.line(edges_BGR, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 1, cv.LINE_AA)

    # [[(line, angle), (line, angle)], [(line, angle), (line, angle)], ...]
    parallel_line_group = angle_approximation(lines)
    perpendicular_line_group = perpendicular_approximation(line_index_dict, lines)

    # Draw functions used to debug
    draw_parallel_line(edges, parallel_line_group)
    draw_perpendicular_line(edges, perpendicular_line_group, line_index_dict)

    # Delete parallel lines that not form keva block
    for group_index in range(0,len(parallel_line_group)):
        group = parallel_line_group[group_index]
        if (len(group) > 1):
            line_angle_arr = group.copy()
            for pair in line_angle_arr:
                line = pair[0]
                if (not line_distance(line, line_angle_arr)):
                    #not witin the line distance we delete it
                    group.remove(pair)

    block_group = find_rectangle(parallel_line_group, draw_perpendicular_line, line_index_dict)

    #draw lines
    for block in block_group:
        for pair in block:
            line = pair[0]
            cv.line(edges_BGR_modified, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 1, cv.LINE_AA)

    imstack = np.hstack((edges_BGR_modified, edges_BGR))
    cv.imshow('stack', imstack)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


'''
Draw functions
'''
def draw_parallel_line(edges, parallel_line_group):
    for group in parallel_line_group:
        if(len(group) > 1):
            edges_BGR_modified = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
            for pair in group:
                line = pair[0]
                cv.line(edges_BGR_modified, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 1, cv.LINE_AA)
            cv.imshow('parallel', edges_BGR_modified)
            cv.waitKey(0)
            cv.destroyAllWindows()

def draw_perpendicular_line(edges, perpendicular_line_group, line_index_dict):
    edges_BGR_modified = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    for key in perpendicular_line_group:
        if(len(perpendicular_line_group[key]) > 0):
            line = line_index_dict[key]
            cv.line(edges_BGR_modified, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 1, cv.LINE_AA)
            for line_perpendicular in perpendicular_line_group[key]:
                cv.line(edges_BGR_modified, (line_perpendicular[0], line_perpendicular[1]),
                        (line_perpendicular[2], line_perpendicular[3]), (0, 255, 0), 1, cv.LINE_AA)
            cv.imshow('perpendicular', edges_BGR_modified)
            cv.waitKey(0)
            cv.destroyAllWindows()
            edges_BGR_modified = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

