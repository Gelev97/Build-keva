import numpy as np
import cv2 as cv
import math

MIN_LINE_LENGTH = 50
MAX_LINE_GAP = 30
NUMBER_OF_INTERSECTION = 20# The probability value for the hough line transform

ANGLE_GAP = 2# Degree difference, we make angles same wihtin these gap

MAX_LINE_LENGTH = 300
WIDTH_HEIGHT_DIST_MIN = 50.0
WIDTH_HEIGHT_DIST_MAX = 300.0

def detect_block(edges):
    # edge with three channel used to display
    edges_BGR = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    edges_BGR_modified = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    # find lines using hough lines transform with probability
    linesP = cv.HoughLinesP(edges, 1, np.pi / 180, NUMBER_OF_INTERSECTION, None, MIN_LINE_LENGTH, MAX_LINE_GAP)

    # array that contains all lines
    lines = []

    # Hashmap all lines with its angle
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            lines.append(list(l))
            cv.line(edges_BGR, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 1, cv.LINE_AA)

    # [[(line, angle), (line, angle)], [(line, angle), (line, angle)], ...]
    parallel_line_group = angle_approximation(lines)
    # print(parallel_line_group)

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

    print(parallel_line_group)
    # draw lines
    for group in parallel_line_group:
        if(len(group) > 1):
            for pair in group:
                line = pair[0]
                print(line)
                cv.line(edges_BGR_modified, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 1, cv.LINE_AA)

    imstack = np.hstack((edges_BGR_modified, edges_BGR))
    cv.imshow('stack', imstack)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return

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

def detect_angle_difference(angle, angle_arr):
    for angle_compare in angle_arr:
        if(abs(angle - angle_compare[1]) > ANGLE_GAP):
            return False
    return True

def test_find_line(edges):
    # edge with three channel used to display
    edges_BGR = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    # find lines using hough lines transform with probability
    linesP = cv.HoughLinesP(edges, 1, np.pi / 180, NUMBER_OF_INTERSECTION, None, MIN_LINE_LENGTH, MAX_LINE_GAP)

    # Hashmap all parallel lines
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            #draw line
            edges_BGR_copy = np.copy(edges_BGR)
            cv.line(edges_BGR_copy, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 1, cv.LINE_AA)
            #show each line
            cv.imshow('test_find_line', edges_BGR_copy)
            cv.waitKey(0)
            cv.destroyAllWindows()

def test_find_parallel_line(edges):
    # edge with three channel used to display
    edges_BGR = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    # find lines using hough lines transform with probability
    linesP = cv.HoughLinesP(edges, 1, np.pi / 180, NUMBER_OF_INTERSECTION, None, MIN_LINE_LENGTH, MAX_LINE_GAP)

    # array that contains all lines
    lines = []

    # Hashmap all lines with its angle
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            lines.append(list(l))

    # [[(line, angle), (line, angle)], [(line, angle), (line, angle)], ...]
    parallel_line_group = angle_approximation(lines)

    for group_index in range(0, len(parallel_line_group)):
        group = parallel_line_group[group_index]
        if (len(group) > 1):
            edges_BGR_copy = np.copy(edges_BGR)
            for pair in group:
                line = pair[0]
                cv.line(edges_BGR_copy, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 1, cv.LINE_AA)
            cv.imshow('test_find_parallel_line', edges_BGR_copy)
            cv.waitKey(0)
            cv.destroyAllWindows()

def test_find_line_pair(edges):
    # edge with three channel used to display
    edges_BGR = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    # find lines using hough lines transform with probability
    linesP = cv.HoughLinesP(edges, 1, np.pi / 180, NUMBER_OF_INTERSECTION, None, MIN_LINE_LENGTH, MAX_LINE_GAP)

    # array that contains all lines
    lines = []

    # Hashmap all lines with its angle
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            lines.append(list(l))

    # [[(line, angle), (line, angle)], [(line, angle), (line, angle)], ...]
    parallel_line_group = angle_approximation(lines)
    print(parallel_line_group)

    # Delete parallel lines that not form keva block
    for group_index in range(0, len(parallel_line_group)):
        group = parallel_line_group[group_index]
        if (len(group) > 1):
            line_angle_arr = group.copy()
            for pair in line_angle_arr:
                line = pair[0]
                if (not line_distance(line, line_angle_arr)):
                    # not witin the line distance we delete it
                    group.remove(pair)

    # draw lines
    for group in parallel_line_group:
        if (len(group) > 1):
            edges_BGR_copy = np.copy(edges_BGR)
            for pair in group:
                line = pair[0]
                cv.line(edges_BGR_copy, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 1, cv.LINE_AA)
            cv.imshow('test_find_parallel_line', edges_BGR_copy)
            cv.waitKey(0)
            cv.destroyAllWindows()


def test_find_rectangle(edges):
    return