import numpy as np
import cv2 as cv
import math
import sys
from operator import itemgetter

SPECIFIC_AREA = [590, 30, 590, 80, 745, 53, 747, 87]

PARAM_LIST = "./Params/Test3_Param.txt"

# Load Params
param_groups = []
text_file = open(PARAM_LIST, "r")
lines = text_file.readlines()
for line in lines:
    param_group = line.split('=')
    param = param_group[1].strip()
    if ("." in param):
        param_convert = float(param)
    else:
        param_convert = int(param)
    param_groups.append(param_convert)
text_file.close()

'''
Define Macro and Tunable Variable
'''

# Params to find edges
MIN_LINE_LENGTH = param_groups[0]
MAX_LINE_GAP = param_groups[1]
NUMBER_OF_INTERSECTION = param_groups[2]# The probability value for the hough line transform

ANGLE_GAP = param_groups[3]# Degree difference, we make angles same wihtin these gap
PERPENDICULAR_THRES = param_groups[4]# Threshold tolerance for perpendiculars

MAX_LINE_LENGTH = param_groups[5]
WIDTH_HEIGHT_DIST_MIN = param_groups[6]
WIDTH_HEIGHT_DIST_MAX = param_groups[7]

# Params to check block length
WIDTH_MIN = param_groups[8]
WIDTH_MAX = param_groups[9]
HEIGHT_MIN = param_groups[10]
HEIGHT_MAX = param_groups[11]

# Params to filter the line
FAKE_RANGE = param_groups[12]

# Same Center threshold
CENTER_SAME_THRESHOLD = param_groups[13]

# Check fake length's qualify rate
QUALIFY_RATE = param_groups[14]

# Threshold used to get edges
THRES = param_groups[15]
BLUR_LEVEL = param_groups[16]
CANNY_EDGE_LOWER_THRES = param_groups[17]
CANNY_EDGE_UPPER_THRES = param_groups[18]

# Rate to determine whether its an edge or not
SIMILARITY = param_groups[19]
PIXEL_SIMILAR_TRHES = param_groups[20]

'''
Edge Detection
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
Find Parallel line
'''
def detect_angle_difference(angle, angle_compare):
    if(abs(angle - angle_compare) > ANGLE_GAP and abs(angle - angle_compare) < 180 - ANGLE_GAP):
            return False
    return True

def angle_approximation(lines):
    # find parallel line
    parallel_line_group = dict()
    for line_index in range(0,len(lines)):
        # calculate angle of each line
        line = lines[line_index]
        # deal with division by 0
        if(line[3] - line[1] == 0):
            angle = 90
        else:
             angle = round(math.atan((line[2] - line[0])/(line[3] - line[1])) * 180.0 / np.pi)
        parallel_line_group[(line_index, angle)] = []
        for line_index_compare in range(0, len(lines)):
            if(line_index != line_index_compare):
                # find all parallel line
                line_compare = lines[line_index_compare]
                # deal with division by 0
                if (line_compare[3] - line_compare[1] == 0):
                    angle_compare = 90
                else:
                    angle_compare = round(math.atan((line_compare[2] - line_compare[0]) / (line_compare[3] - line_compare[1])) * 180.0 / np.pi)
                if(detect_angle_difference(angle, angle_compare)):
                    parallel_line_group[(line_index, angle)].append(line_compare)
    # [(line_index, angle) : [line,line,line], ...]
    return parallel_line_group

'''
Find Parallel line distance
'''
def line_distance(line, line_to_compare):
    # find distance between parallel lines
    # pick a point from one line
    (center_x, center_y) = (int((line[0]+line[2])/2.0), int((line[1]+line[3])/2.0))

    # create a perpendicular line toward the compare line
    if (line[3] - line[1] == 0):
        angle = 90
    else:
        angle = round(
            math.atan((line[2] - line[0]) / (line[3] - line[1])) * 180.0 / np.pi)
    slope_line = math.tan((angle+90)*np.pi/180)
    intercept = center_x - slope_line * center_y

    # create the other end of perpendicular line
    center_y1 = center_y + 2
    center_x1 = slope_line*center_y1 + intercept

    (intersectionX, intersectionY, valid) = intersectLines([center_x, center_y, center_x1, center_y1], line_to_compare)
    distance_line = distance(center_x, intersectionX, center_y, intersectionY)

    # print("distance")
    # print(distance_line)
    if(distance_line >= WIDTH_HEIGHT_DIST_MIN and distance_line <= WIDTH_HEIGHT_DIST_MAX):
        return True
    return False

'''
Find Perpendicular line
'''
def perpendicular_approximation(parallel_line_group, line_index_dict, lines, edges):
    perpendicular_line_group = dict()
    for line_set in parallel_line_group:
        line = line_index_dict[line_set[0]]
        angle = line_set[1]
        perpendicular_line_group[line_set] = []
        for line_compare in lines:
            #create a new dict for this line
            if (line_compare[3] - line_compare[1] == 0):
                angle_compare = 90
            else:
                angle_compare = round(math.atan((line_compare[2] - line_compare[0]) / (line_compare[3] - line_compare[1])) * 180.0 / np.pi)
            # edges_BGR = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
            # cv.line(edges_BGR, (line[0], line[1]), (line[2], line[3]),(0, 0, 255), 1, cv.LINE_AA)
            # cv.line(edges_BGR, (line_compare[0], line_compare[1]), (line_compare[2], line_compare[3]),(255, 0, 0), 1, cv.LINE_AA)
            # cv.imshow('hough', edges_BGR)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            # print("angle and angle compare")
            # print(angle, angle_compare)
            threshold_value = abs(angle - angle_compare)
            # print("threshold value")
            # print(threshold_value)
            if(threshold_value > 90 - PERPENDICULAR_THRES and threshold_value < 90 + PERPENDICULAR_THRES):
                (intersectionX, intersectionY, valid) = intersectLines(line,line_compare)
                if(valid):
                    perpendicular_line_group[line_set].append((line_compare,(intersectionX,intersectionY)))
    return perpendicular_line_group

'''
Function used to extend the finding hough line
'''
def extend(l):
    result_line = [0,0,0,0]
    # extend these line to full size across the image
    length = math.sqrt((l[0] - l[2])**2 + (l[1] - l[3])**2)
    result_line[0] = int(l[0] + (l[2] - l[0]) / length * 1000)
    result_line[1] = int(l[1] + (l[3] - l[1]) / length * 1000)
    result_line[2] = int(l[0] + (l[0] - l[2]) / length * 1000)
    result_line[3] = int(l[1] + (l[1] - l[3]) / length * 1000)
    return result_line

"""
This returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)
returns a tuple: (xi, yi, valid, r, s), where
(xi, yi) is the intersection
valid == 0 if there are 0 or inf. intersections (invalid)
valid == 1 if it has a unique intersection ON the segment
"""
def intersectLines(line, line_to_compare):
    DET_TOLERANCE = 0.00000001

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = (line[0], line[1])
    x2, y2 = (line[2], line[3])
    dx1 = x2 - x1
    dy1 = y2 - y1

    # the second line is ptA + s*(ptB-ptA)
    x, y = (line_to_compare[0], line_to_compare[1])
    xB, yB = (line_to_compare[2], line_to_compare[3])
    dx = xB - x
    dy = yB - y

    # we need to find the (typically unique) values of r and s
    # that will satisfy
    #
    # (x1, y1) + r(dx1, dy1) = (x, y) + s(dx, dy)
    #
    # which is the same as
    #
    #    [ dx1  -dx ][ r ] = [ x-x1 ]
    #    [ dy1  -dy ][ s ] = [ y-y1 ]
    #
    # whose solution is
    #
    #    [ r ] = _1_  [  -dy   dx ] [ x-x1 ]
    #    [ s ] = DET  [ -dy1  dx1 ] [ y-y1 ]
    #
    # where DET = (-dx1 * dy + dy1 * dx)
    #
    # if DET is too small, they're parallel
    #
    DET = (-dx1 * dy + dy1 * dx)

    if math.fabs(DET) < DET_TOLERANCE: return (0, 0, 0)

    # now, the determinant should be OK
    DETinv = 1.0 / DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy * (x - x1) + dx * (y - y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x - x1) + dx1 * (y - y1))

    # return the average of the two descriptions
    xi = (x1 + r * dx1 + x + s * dx) / 2.0
    yi = (y1 + r * dy1 + y + s * dy) / 2.0
    return (round(xi), round(yi), 1)
'''
Find Rectangles
'''
def find_common(perpendicular_line, perpendicular_line_compare):
    common = []
    #find common elements between two arrays
    for line_perpendicular_set in perpendicular_line:
        line_perpendicular = line_perpendicular_set[0]
        for line_perpendicular_set_compare in perpendicular_line_compare:
            line_perpendicular_compare = line_perpendicular_set_compare[0]
            if(line_perpendicular == line_perpendicular_compare):
                # add same perpendicular line and two intersection points
                common.append((line_perpendicular,line_perpendicular_set[1],line_perpendicular_set_compare[1]))
    return common

def permute_in_two(common):
    # put all elements in group of two
    permute_common = []
    for line_set in common:
        for line_set_permute in common:
            if(line_set != line_set_permute):
                # get rid of same groups
                if((line_set_permute, line_set) not in permute_common and \
                        (line_set, line_set_permute) not in permute_common):
                    permute_common.append((line_set, line_set_permute))
    return permute_common

def check_parallel(line_1, line_2, parallel_line_group, index1, index2):
    # to check whether line1 and line2 are parallel
    for key in parallel_line_group:
        if(key[0] == index1):
            if(line_2 not in parallel_line_group[key]): return False
        if(key[0] == index2):
            if(line_1 not in parallel_line_group[key]): return False
    return True

def distance(p0, p1, p2, p3):
    return math.sqrt((p0 - p1) ** 2 + (p2 - p3) ** 2)

def check_intersections(intersections):
    # find if the intersections are forming a rectangle
    # sort the array ascending in x
    intersection_X = intersections.copy()
    intersection_X.sort(key=itemgetter(0))
    left = [intersection_X[0], intersection_X[1]]
    right = [intersection_X[2], intersection_X[3]]

    if (left[0][1] > right[0][1] and left[0][1] > right[1][1] and \
            left[1][1] > right[0][1] and left[1][1] > right[1][1]):
        top_left = right[0]
        top_right = right[1]
        bottom_left = left[0]
        bottom_right = left[1]
    elif (left[0][1] < right[0][1] and left[0][1] < right[1][1] and \
          left[1][1] < right[0][1] and left[1][1] < right[1][1]):
        top_left = left[0]
        top_right = left[1]
        bottom_left = right[0]
        bottom_right = right[1]
    else:
        if (left[0][1] > left[1][1]):
            top_left = left[1]
            bottom_left = left[0]
        else:
            top_left = left[0]
            bottom_left = left[1]

        if (right[0][1] > right[1][1]):
            top_right = right[1]
            bottom_right = right[0]
        else:
            top_right = right[0]
            bottom_right = right[1]

    # calculate four side length
    lenght_left = distance(top_left[0],bottom_left[0],top_left[1],bottom_left[1])
    lenght_right = distance(top_right[0], bottom_right[0], top_right[1], bottom_right[1])
    lenght_top = distance(top_left[0], top_right[0], top_left[1], top_right[1])
    lenght_bottom = distance(bottom_left[0], bottom_right[0], bottom_left[1], bottom_right[1])

    # print("check_intersection")
    # print(lenght_left, lenght_right, lenght_top, lenght_bottom)
    # check length
    if (lenght_left <= lenght_top and lenght_right <= lenght_bottom):
        if (lenght_left > WIDTH_MAX or lenght_left < WIDTH_MIN): return False
        if (lenght_top > HEIGHT_MAX or lenght_top < HEIGHT_MIN): return False
        if (lenght_right > WIDTH_MAX or lenght_right < WIDTH_MIN): return False
        if (lenght_bottom > HEIGHT_MAX or lenght_bottom < HEIGHT_MIN): return False
    else:
        if (lenght_left > HEIGHT_MAX or lenght_left < HEIGHT_MIN): return False
        if (lenght_top > WIDTH_MAX or lenght_top < WIDTH_MIN): return False
        if (lenght_right > HEIGHT_MAX or lenght_right < HEIGHT_MIN): return False
        if (lenght_bottom > WIDTH_MAX or lenght_bottom < WIDTH_MIN): return False

    return True

# This function is used to test the distance between two rectangles
# based on the distance of four corners
def check_distance(intersections, intersections_compare):
    distance_total = 0
    for vertex_index in range(0,len(intersections)):
        intersections_vertex = intersections[vertex_index]
        intersections_compare_vertex = intersections_compare[vertex_index]
        distance_total += distance(intersections_vertex[0],intersections_compare_vertex[0], \
                    intersections_vertex[1], intersections_compare_vertex[1])
    if(distance_total/len(intersections) > CENTER_SAME_THRESHOLD):
        return True
    return False

# measure the angle between two lines
def angle(line_1, line_2):
    if (line_1[1][1] - line_1[0][1] == 0):
        angle_line_1 = 90
    else:
        angle_line_1 = round(math.atan((line_1[1][0] - line_1[0][0]) / (line_1[1][1] - line_1[0][1])) * 180.0 / np.pi)

    if (line_2[1][1] - line_2[0][1] == 0):
        angle_line_2 = 90
    else:
        angle_line_2 = round(math.atan((line_2[1][0] - line_2[0][0]) / (line_2[1][1] - line_2[0][1])) * 180.0 / np.pi)

    return abs(angle_line_1 - angle_line_2)

# This function is used to measure the four angles of each rectangle in the group
def measure_angle(intersection_group):
    result = dict()
    for intersection in intersection_group:
        [top_left, bottom_left, top_right, bottom_right] = intersection
        result[(top_left,bottom_left, top_right, bottom_right)] = []
        result[(top_left, bottom_left, top_right, bottom_right)].append(angle([top_right,bottom_left],[bottom_left,bottom_right]))
        result[(top_left, bottom_left, top_right, bottom_right)].append(angle([bottom_left,bottom_right],[bottom_right,top_right]))
        result[(top_left, bottom_left, top_right, bottom_right)].append(angle([bottom_right,top_right],[top_right,top_left]))
        result[(top_left, bottom_left, top_right, bottom_right)].append(angle([top_right,top_left],[top_left,bottom_left]))
    return result

def remove_duplicate(blocks):
    block_dict = dict()
    duplicate_group_index = 0
    for block in blocks:
        # get the current block intersection in given sequence
        new_flag = 1
        intersections = block
        intersections.sort(key=itemgetter(0))
        left = [intersections[0], intersections[1]]
        right = [intersections[2], intersections[3]]

        if (left[0][1] > right[0][1] and left[0][1] > right[1][1] and \
                left[1][1] > right[0][1] and left[1][1] > right[1][1]):
            top_left = right[0]
            top_right = right[1]
            bottom_left = left[0]
            bottom_right = left[1]
        elif (left[0][1] < right[0][1] and left[0][1] < right[1][1] and \
              left[1][1] < right[0][1] and left[1][1] < right[1][1]):
            top_left = left[0]
            top_right = left[1]
            bottom_left = right[0]
            bottom_right = right[1]
        else:
            if (left[0][1] > left[1][1]):
                top_left = left[1]
                bottom_left = left[0]
            else:
                top_left = left[0]
                bottom_left = left[1]

            if (right[0][1] > right[1][1]):
                top_right = right[1]
                bottom_right = right[0]
            else:
                top_right = right[0]
                bottom_right = right[1]

        intersections = [top_left, bottom_left, top_right, bottom_right]
        for center_index in block_dict:
            # compare it with all center in this group
            qualify_flag = 1
            for intersection_compare in block_dict[center_index]:
                if (check_distance(intersections, intersection_compare)):
                    qualify_flag = 0
            if (qualify_flag == 1):
                block_dict[center_index].append(intersections)
                new_flag = 0
        # If no group fit, just create a new group
        if (new_flag == 1):
            block_dict[duplicate_group_index] = [intersections]
            duplicate_group_index += 1
    return block_dict

def merge_duplicate(block_dict, edges):
    result = []
    for center in block_dict:
        intersection_group = block_dict[center]
        intersection = [[0, 0], [0, 0], [0, 0], [0, 0]]
        intersections = [(0, 0), (0, 0), (0, 0), (0, 0)]
        intersection_count = 0
        for intersection_merge in intersection_group:
            intersection[0][0] += intersection_merge[0][0]
            intersection[0][1] += intersection_merge[0][1]
            intersection[1][0] += intersection_merge[1][0]
            intersection[1][1] += intersection_merge[1][1]
            intersection[2][0] += intersection_merge[2][0]
            intersection[2][1] += intersection_merge[2][1]
            intersection[3][0] += intersection_merge[3][0]
            intersection[3][1] += intersection_merge[3][1]
            intersection_count += 1
        if (intersection_count != 0):
            intersections[0] = (
            round(intersection[0][0] / intersection_count), round(intersection[0][1] / intersection_count))
            intersections[1] = (
            round(intersection[1][0] / intersection_count), round(intersection[1][1] / intersection_count))
            intersections[2] = (
            round(intersection[2][0] / intersection_count), round(intersection[2][1] / intersection_count))
            intersections[3] = (
            round(intersection[3][0] / intersection_count), round(intersection[3][1] / intersection_count))
            result.append(intersections)

    return result

# A utility function to calculate
# area of triangle formed by (x1, y1),
# (x2, y2) and (x3, y3)
def area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


# A function to check whether point
# P(x, y) lies inside the rectangle
# formed by A(x1, y1), B(x2, y2),
# C(x3, y3) and D(x4, y4)
def check(x1, y1, x2, y2, x3, y3, x4, y4, x, y):
    # Calculate area of rectangle ABCD
    A = (area(x1, y1, x2, y2, x4, y4) + area(x1, y1, x3, y3, x4, y4))

    # Calculate area of triangle PAB
    A1 = area(x, y, x1, y1, x2, y2)

    # Calculate area of triangle PBC
    A2 = area(x, y, x2, y2, x4, y4)

    # Calculate area of triangle PCD
    A3 = area(x, y, x3, y3, x4, y4)

    # Calculate area of triangle PAD
    A4 = area(x, y, x1, y1, x3, y3)

    # Check if sum of A1, A2, A3
    # and A4 is same as A
    if(abs(A - A1 - A2 - A3 - A4) < 10):
        return True
    return False

def find_rectangle(edges, parallel_line_group, perpendicular_line_group, line_index_dict):
    raw_blocks = []
    blocks = []
    for key in perpendicular_line_group:
        if (len(perpendicular_line_group[key]) > 1):
            for key_compare in perpendicular_line_group:
                if(key_compare != key and len(perpendicular_line_group[key_compare]) > 1):
                    common_perpendicular = find_common(perpendicular_line_group[key], perpendicular_line_group[key_compare])
                    if(len(common_perpendicular) > 1):
                        # find common perpendicular lines of two different line
                        permute_common = permute_in_two(common_perpendicular)
                        # test whether these lines with same perpendiculars are parallel
                        line_1 = line_index_dict[key[0]]
                        line_2 = line_index_dict[key_compare[0]]
                        if(check_parallel(line_1, line_2, parallel_line_group, key[0], key_compare[0])):
                            raw_blocks.append((line_1, line_2, permute_common))
                    else:
                        continue

    # test raw blocks
    # draw_raw_blocks(edges, raw_blocks)

    # filter out not keva block
    for raw_block in raw_blocks:
        # check each perpendicular group
        for block_common_2 in raw_block[2]:
            # four intersections of the rectangle
            intersections = [block_common_2[0][1], block_common_2[0][2], block_common_2[1][1], block_common_2[1][2]]
            # draw_raw_blocks_specific_area(edges, intersections)
            if (check_intersections(intersections)):
                blocks.append(intersections)

    # for intersection in blocks:
    #     draw_raw_blocks_specific_area(edges, intersection)

    # test blocks after filter
    # draw_blocks(edges, blocks)

    # Re-order the intersecitons
    result = []
    for block in blocks:
        # get the current block intersection in given sequence
        intersections = block
        intersections.sort(key=itemgetter(0))
        left = [intersections[0], intersections[1]]
        right = [intersections[2], intersections[3]]

        if (left[0][1] > right[0][1] and left[0][1] > right[1][1] and \
                left[1][1] > right[0][1] and left[1][1] > right[1][1]):
            top_left = right[0]
            top_right = right[1]
            bottom_left = left[0]
            bottom_right = left[1]
        elif (left[0][1] < right[0][1] and left[0][1] < right[1][1] and \
              left[1][1] < right[0][1] and left[1][1] < right[1][1]):
            top_left = left[0]
            top_right = left[1]
            bottom_left = right[0]
            bottom_right = right[1]
        else:
            if (left[0][1] > left[1][1]):
                top_left = left[1]
                bottom_left = left[0]
            else:
                top_left = left[0]
                bottom_left = left[1]

            if (right[0][1] > right[1][1]):
                top_right = right[1]
                bottom_right = right[0]
            else:
                top_right = right[0]
                bottom_right = right[1]

        intersection = [top_left, bottom_left, top_right, bottom_right]
        result.append(intersection)

    return result

'''
Delete fake block

'''
def create_line_arr_height(line, img):
    # extend the line gradually increase x by 1
    # when y1 = y2
    if ((line[1] - line[3]) == 0):
        line_arr = []
        if (line[0] < line[2]):
            for x in range(line[0], line[2]+1):
                if(x >=0 and x < img.shape[1]):
                    line_arr.append((x, line[1]))
        else:
            for x in range(line[0], line[2]-1, -1):
                if (x >= 0 and x < img.shape[1]):
                    line_arr.append((x, line[1]))
        return line_arr
    slope = (line[0] - line[2]) / (line[1] - line[3])
    intercept = line[0] - line[1]*slope
    if (line[0] >= 0 and line[0] < img.shape[1] and line[1] >= 0 and line[1] < img.shape[0]):
        line_arr = [(line[0], line[1])]
    else:
        line_arr = []

    if(line[0] < line[2]):
        for x in range(line[0]+1, line[2]):
            y = round((x - intercept)/slope)
            if (x >= 0 and x < img.shape[1] and y >= 0 and y < img.shape[0]):
                line_arr.append((x,y))
    else:
        for x in range(line[0]+1, line[2], -1):
            y = round((x - intercept) / slope)
            if (x >= 0 and x < img.shape[1] and y >= 0 and y < img.shape[0]):
                line_arr.append((x,y))
    if (line[2] >= 0 and line[2] < img.shape[1] and line[3] >= 0 and line[3] < img.shape[0]):
        line_arr.append((line[2], line[3]))
    return line_arr


def create_line_arr_width(line, img):
    # extend the line gradually increase y by 1
    # when x1 = x2
    if((line[0] - line[2]) == 0):
        line_arr = []
        if(line[1] < line[3]):
            for y in range(line[1], line[3]+1):
                if (y >= 0 and y < img.shape[0]):
                    line_arr.append((line[0], y))
        else:
            for y in range(line[1], line[3]-1, -1):
                if (y >= 0 and y < img.shape[0]):
                    line_arr.append((line[0], y))
        return line_arr

    slope = (line[0] - line[2]) / (line[1] - line[3])

    intercept = line[0] - line[1] * slope
    if (line[0] >= 0 and line[0] < img.shape[1] and line[1] >= 0 and line[1] < img.shape[0]):
        line_arr = [(line[0], line[1])]
    else:
        line_arr = []

    if (line[1] < line[3]):
        for y in range(line[1]+1, line[3]):
            x = round(slope*y + intercept)
            if (x >= 0 and x < img.shape[1] and y >= 0 and y < img.shape[0]):
                line_arr.append((x, y))
    else:
        for y in range(line[1]+1, line[3], -1):
            x = round(slope * y + intercept)
            if (x >= 0 and x < img.shape[1] and y >= 0 and y < img.shape[0]):
                line_arr.append((x, y))
    if (line[2] >= 0 and line[2] < img.shape[1] and line[3] >= 0 and line[3] < img.shape[0]):
        line_arr.append((line[2], line[3]))
    return line_arr

def check_background_or_color(points_1, points_2, img):
    point_similar = 0
    for point_index in range (0,len(points_1)):
        point_1 = points_1[point_index]
        point_2 = points_2[point_index]
        pixel_1 = img[(point_1[1], point_1[0])]
        pixel_2 = img[(point_2[1], point_2[0])]
        average_difference = (abs(int(pixel_1[0]) - int(pixel_2[0])) + abs(int(pixel_1[1]) - int(pixel_2[1])) \
                                            + abs(int(pixel_1[2]) - int(pixel_2[2])))
        if(average_difference < PIXEL_SIMILAR_TRHES):
            point_similar += 1
    if(point_similar/len(points_1) > SIMILARITY):
        return 0
    else:
        return 1

def clear_fake_block(block, img):
    result = []
    length = len(block)
    count = 0
    prev_percentage = 0
    for intersections in block:
        lines = []
        line_test = []
        # create all four side
        # identify four corner
        top_left = intersections[0]
        bottom_left = intersections[1]
        top_right = intersections[2]
        bottom_right = intersections[3]

        # check length (1 -> y+1, 0 -> x+1)
        if (abs(top_left[0] - bottom_left[0]) < abs(top_left[1] - bottom_left[1])):
            lines.append((1, [top_left[0], top_left[1], bottom_left[0], bottom_left[1]]))
            lines.append((0, [bottom_left[0], bottom_left[1], bottom_right[0], bottom_right[1]]))
            lines.append((1, [top_right[0], top_right[1], bottom_right[0], bottom_right[1]]))
            lines.append((0, [top_left[0], top_left[1], top_right[0], top_right[1]]))
        else:
            lines.append((0, [top_left[0], top_left[1], bottom_left[0], bottom_left[1]]))
            lines.append((1, [bottom_left[0], bottom_left[1], bottom_right[0], bottom_right[1]]))
            lines.append((0, [top_right[0], top_right[1], bottom_right[0], bottom_right[1]]))
            lines.append((1, [top_left[0], top_left[1], top_right[0], top_right[1]]))

        img_display = img.copy()
        for qualified_line_set in lines:
            qualified_line = qualified_line_set[1]
            # extend the line by 1
            if(qualified_line_set[0] == 1):
                qualified_line_arr = create_line_arr_width(qualified_line, img)
            else:
                qualified_line_arr = create_line_arr_height(qualified_line, img)

            qualified_point = []

            # avoid division by zero
            if(len(qualified_line_arr) != 0):
                for point in qualified_line_arr:
                    # check for certain directions
                    if (qualified_line_set[0] == 1):
                        # the first direction, gather points
                        points_dir_1_side_1 = []
                        points_dir_1_side_2 = []
                        for dir_1 in range(0,FAKE_RANGE+1):
                            if((point[0] + dir_1) < img.shape[1] and (point[0] - dir_1) >= 0):
                                if((point[0] + dir_1, point[1]) not in points_dir_1_side_1):
                                    points_dir_1_side_1.append((point[0] + dir_1, point[1]))
                                if ((point[0] - dir_1, point[1]) not in points_dir_1_side_2):
                                    points_dir_1_side_2.append((point[0] - dir_1, point[1]))

                        # check the point for this direction
                        # whether it follows one side of backgroud and one side of color
                        dir1_check = check_background_or_color(points_dir_1_side_1, points_dir_1_side_2, img)
                        if (dir1_check == 1): qualified_point.append(point)
                    else:
                        points_dir_2_side_1 = []
                        points_dir_2_side_2 = []

                        for dir_2 in range(0, FAKE_RANGE + 1):
                            if ((point[1] + dir_2) < img.shape[0] and (point[1] - dir_2) >= 0):
                                if ((point[0], point[1] + dir_2) not in points_dir_2_side_1):
                                    points_dir_2_side_1.append((point[0], point[1] + dir_2))
                                if ((point[0], point[1] - dir_2) not in points_dir_2_side_2):
                                    points_dir_2_side_2.append((point[0], point[1] - dir_2))

                        dir2_check = check_background_or_color(points_dir_2_side_1, points_dir_2_side_2, img)
                        if (dir2_check == 1): qualified_point.append(point)
                # print("Qualify rate")
                # print(len(qualified_point)/len(qualified_line_arr))

                if(len(qualified_point)/len(qualified_line_arr) > QUALIFY_RATE):
                    if(qualified_line not in line_test):
                        line_test.append(qualified_line)
        #                 cv.line(img_display, (qualified_line[0], qualified_line[1]), (qualified_line[2], qualified_line[3]),(0, 255, 255), 4, cv.LINE_AA)
        #         else:
        #             cv.line(img_display, (qualified_line[0], qualified_line[1]), (qualified_line[2], qualified_line[3]),(255, 255, 0), 4, cv.LINE_AA)
        # cv.imshow('img', img_display)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        if(len(line_test) == 4): result.append(intersections)
        count += 1
        if (round(count / length * 100) != prev_percentage):
            print(str(round(count / length * 100)) + "%")
        prev_percentage = round(count / length * 100)
    return result

'''
Pipeline to find block
'''
def detect_block(edges, img):
    # find lines using hough lines transform with probability
    linesP = cv.HoughLinesP(edges, 1, np.pi / 180, NUMBER_OF_INTERSECTION, None, MIN_LINE_LENGTH, MAX_LINE_GAP)

    # array that contains all lines
    lines = []

    # A dictionary that maps every line with an unique index
    line_index_dict = dict()
    index_dict = 0
    line_not_extend = []
    extend_line_map_line = dict()

    # Hashmap all lines with its angle
    edges_BGR = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            # l = extend_and_shift(l, rectX, rectY)
            l_extend = extend(l)
            lines.append(list(l_extend))
            line_not_extend.append(l)
            line_index_dict[index_dict] = l_extend
            extend_line_map_line[index_dict] = l
            index_dict += 1
            cv.line(edges_BGR, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 1, cv.LINE_AA)
    cv.imshow('hough', edges_BGR)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # [(line_index, angle) : [line,line,line], ...]
    parallel_line_group = angle_approximation(lines)

    # Delete parallel lines that not form keva block
    for line_set in parallel_line_group:
        group = parallel_line_group[line_set]
        line = line_index_dict[line_set[0]]
        if (len(group) >= 1):
            line_parallel_index = 0
            while(line_parallel_index < len(group)):
                # edges_BGR = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
                # cv.line(edges_BGR, (line[0], line[1]), (line[2], line[3]),(255, 255, 0), 1, cv.LINE_AA)
                line_parallel = group[line_parallel_index]
                # cv.line(edges_BGR, (line_parallel[0], line_parallel[1]), (line_parallel[2], line_parallel[3]),(255, 255, 0), 1, cv.LINE_AA)
                # cv.imshow('line_distance', edges_BGR)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
                # get the non_extend line to test distance between parallel
                line_check_distance = extend_line_map_line[line_set[0]]
                if (not line_distance(line_check_distance, line_parallel)):
                    # not witin the line distance we delete it
                    group.remove(line_parallel)
                else:
                    line_parallel_index += 1


    # Delete alone parallel line or empty group after above operation
    for line_set in list(parallel_line_group.keys()):
        if(len(parallel_line_group[line_set]) < 1):
            del parallel_line_group[line_set]

    perpendicular_line_group = perpendicular_approximation(parallel_line_group, line_index_dict, lines, edges)

    # Draw functions used to debug
    # draw_parallel_line(edges, parallel_line_group, line_index_dict)
    # draw_perpendicular_line(edges, perpendicular_line_group, line_index_dict)
    # draw_parallel_line_specific_area(edges, parallel_line_group, line_index_dict, extend_line_map_line)
    # draw_perpendicular_line_specific_area(edges, perpendicular_line_group, line_index_dict, extend_line_map_line)

    block = find_rectangle(edges, parallel_line_group, perpendicular_line_group, line_index_dict)

    # Clear fake block
    block = clear_fake_block(block, img)

    # remove duplicate
    block_dict = remove_duplicate(block)
    # merge duplicate
    result = merge_duplicate(block_dict, edges)

    blocks = block.copy()
    # double check for duplication
    while (len(blocks) > len(result)):
        blocks = result.copy()
        block_dict = remove_duplicate(blocks)
        result = merge_duplicate(block_dict, edges)
        print("block: " + str(len(blocks)) + "result: " + str(len(result)))

    return result

'''
Draw functions
'''
def show_edge(img, threshold, blur, edges):
    edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    imstack_horizontal1 = np.hstack((img, threshold))
    imstack_horizontal2 = np.hstack((blur, edges))
    imstack = np.vstack((imstack_horizontal1,imstack_horizontal2))
    cv.imshow('stack', imstack)
    cv.waitKey(0)
    cv.destroyAllWindows()

def draw_parallel_line(edges, parallel_line_group, line_index_dict):
    for key in parallel_line_group:
        edges_BGR_modified = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        line = line_index_dict[key[0]]
        print(line)
        print("angle:" + str(key[1]))
        cv.line(edges_BGR_modified, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 5, cv.LINE_AA)
        group = parallel_line_group[key]
        if(len(group) >= 1):
            for line_parallel in group:
                print(line_parallel)
                if (line_parallel[3] - line_parallel[1] == 0):
                    angle_compare = 90
                else:
                    angle_compare = round(math.atan((line_parallel[2] - line_parallel[0]) / (line_parallel[3] - line_parallel[1])) * 180.0 / np.pi)
                print("angle_compare:" + str(angle_compare))
                cv.line(edges_BGR_modified, (line_parallel[0], line_parallel[1]), (line_parallel[2], line_parallel[3]), (0, 255, 0), 1, cv.LINE_AA)
            cv.imshow('parallel', edges_BGR_modified)
            cv.waitKey(0)
            cv.destroyAllWindows()

def draw_perpendicular_line(edges, perpendicular_line_group, line_index_dict):
    edges_BGR_modified = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    for key in perpendicular_line_group:
        if(len(perpendicular_line_group[key]) > 0):
            line = line_index_dict[key[0]]
            cv.line(edges_BGR_modified, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 1, cv.LINE_AA)
            for line_perpendicular_set in perpendicular_line_group[key]:
                line_perpendicular = line_perpendicular_set[0]
                intersection = line_perpendicular_set[1]
                cv.line(edges_BGR_modified, (line_perpendicular[0], line_perpendicular[1]),
                        (line_perpendicular[2], line_perpendicular[3]), (0, 255, 0), 1, cv.LINE_AA)
                cv.circle(edges_BGR_modified, intersection, 2, (255, 0, 0), 10)
            cv.imshow('perpendicular', edges_BGR_modified)
            cv.waitKey(0)
            cv.destroyAllWindows()

def draw_parallel_line_specific_area(edges, parallel_line_group, line_index_dict, extend_line_map_line):
    [x1, y1, x2, y2, x3, y3, x4, y4] = SPECIFIC_AREA
    for key in parallel_line_group:
        edges_BGR_modified = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        line = line_index_dict[key[0]]
        print(line)
        print("angle:" + str(key[1]))
        line_to_check = extend_line_map_line[key[0]]
        check1 = check(x1, y1, x2, y2, x3, y3, x4, y4, line_to_check[0], line_to_check[1])
        check2 = check(x1, y1, x2, y2, x3, y3, x4, y4, line_to_check[2], line_to_check[3])
        if (check1 or check2):
            cv.line(edges_BGR_modified, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 5, cv.LINE_AA)
            group = parallel_line_group[key]
            if (len(group) >= 1):
                for line_parallel in group:
                    print(line_parallel)
                    if (line_parallel[3] - line_parallel[1] == 0):
                        angle_compare = 90
                    else:
                        angle_compare = round(math.atan(
                            (line_parallel[2] - line_parallel[0]) / (line_parallel[3] - line_parallel[1])) * 180.0 / np.pi)
                    print("angle_compare:" + str(angle_compare))
                    cv.line(edges_BGR_modified, (line_parallel[0], line_parallel[1]), (line_parallel[2], line_parallel[3]),
                            (0, 255, 0), 1, cv.LINE_AA)
                cv.imshow('parallel', edges_BGR_modified)
                cv.waitKey(0)
                cv.destroyAllWindows()

def draw_perpendicular_line_specific_area(edges, perpendicular_line_group, line_index_dict, extend_line_map_line):
    [x1, y1, x2, y2, x3, y3, x4, y4] = SPECIFIC_AREA
    edges_BGR_modified = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    for key in perpendicular_line_group:
        if (len(perpendicular_line_group[key]) > 0):
            line = line_index_dict[key[0]]
            line_to_check = extend_line_map_line[key[0]]
            check1 = check(x1, y1, x2, y2, x3, y3, x4, y4, line_to_check[0], line_to_check[1])
            check2 = check(x1, y1, x2, y2, x3, y3, x4, y4, line_to_check[2], line_to_check[3])
            if (check1 or check2):
                cv.line(edges_BGR_modified, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 1, cv.LINE_AA)
                for line_perpendicular_set in perpendicular_line_group[key]:
                    line_perpendicular = line_perpendicular_set[0]
                    intersection = line_perpendicular_set[1]
                    cv.line(edges_BGR_modified, (line_perpendicular[0], line_perpendicular[1]),
                            (line_perpendicular[2], line_perpendicular[3]), (0, 255, 0), 1, cv.LINE_AA)
                    cv.circle(edges_BGR_modified, intersection, 2, (255, 0, 0), 10)
                cv.imshow('perpendicular', edges_BGR_modified)
                cv.waitKey(0)
                cv.destroyAllWindows()
                edges_BGR_modified = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

def draw_rectangular(edges, intersections):
        edges_BGR_modified = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        cv.circle(edges_BGR_modified, intersections[0], 2, (255, 0, 0), 10)
        cv.circle(edges_BGR_modified, intersections[1], 2, (255, 0, 0), 10)
        cv.circle(edges_BGR_modified, intersections[2], 2, (255, 0, 0), 10)
        cv.circle(edges_BGR_modified, intersections[3], 2, (255, 0, 0), 10)
        cv.imshow('draw_rectangular', edges_BGR_modified)
        cv.waitKey(0)
        cv.destroyAllWindows()

def draw_rectangular_specific_area(edges, intersections):
    [x1, y1, x2, y2, x3, y3, x4, y4] = SPECIFIC_AREA
    check1 = check(x1, y1, x2, y2, x3, y3, x4, y4, intersections[0][0], intersections[0][1])
    check2 = check(x1, y1, x2, y2, x3, y3, x4, y4, intersections[1][0], intersections[1][1])
    check3 = check(x1, y1, x2, y2, x3, y3, x4, y4, intersections[2][0], intersections[2][1])
    check4 = check(x1, y1, x2, y2, x3, y3, x4, y4, intersections[3][0], intersections[3][1])
    if (check1 and check2 and check3 and check4):
        draw_rectangular(edges, intersections)

def draw_raw_blocks_specific_area(edges, intersections):
    [x1, y1, x2, y2, x3, y3, x4, y4] = SPECIFIC_AREA
    check1 = check(x1, y1, x2, y2, x3, y3, x4, y4, intersections[0][0], intersections[0][1])
    check2 = check(x1, y1, x2, y2, x3, y3, x4, y4, intersections[1][0], intersections[1][1])
    check3 = check(x1, y1, x2, y2, x3, y3, x4, y4, intersections[2][0], intersections[2][1])
    check4 = check(x1, y1, x2, y2, x3, y3, x4, y4, intersections[3][0], intersections[3][1])
    if (check1 and check2 and check3 and check4):
        edges_BGR = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        cv.circle(edges_BGR, intersections[0], 2, (255, 0, 0), 10)
        cv.circle(edges_BGR, intersections[1], 2, (255, 0, 0), 10)
        cv.circle(edges_BGR, intersections[2], 2, (255, 0, 0), 10)
        cv.circle(edges_BGR, intersections[3], 2, (255, 0, 0), 10)
        cv.imshow('draw_raw_blocks', edges_BGR)
        cv.waitKey(0)
        cv.destroyAllWindows()

def  draw_raw_blocks(edges, raw_blocks):
    for blockset in raw_blocks:
        line1 = blockset[0]
        line2 = blockset[1]
        permute_common = blockset[2]
        for common in permute_common:
            intersections = []
            line3 = common[0][0]
            intersections.append(common[0][1])
            intersections.append(common[0][2])
            line4 = common[1][0]
            intersections.append(common[1][1])
            intersections.append(common[1][2])
            edges_BGR_modified = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
            cv.line(edges_BGR_modified, (line1[0], line1[1]), (line1[2], line1[3]), (255, 0, 0), 1, cv.LINE_AA)
            cv.line(edges_BGR_modified, (line2[0], line2[1]), (line2[2], line2[3]), (255, 0, 0), 1, cv.LINE_AA)
            cv.line(edges_BGR_modified, (line3[0], line3[1]), (line3[2], line3[3]), (0, 255, 0), 1, cv.LINE_AA)
            cv.line(edges_BGR_modified, (line4[0], line4[1]), (line4[2], line4[3]), (0, 255, 0), 1, cv.LINE_AA)
            cv.circle(edges_BGR_modified, intersections[0], 2, (255, 0, 0), 10)
            cv.circle(edges_BGR_modified, intersections[1], 2, (255, 0, 0), 10)
            cv.circle(edges_BGR_modified, intersections[2], 2, (255, 0, 0), 10)
            cv.circle(edges_BGR_modified, intersections[3], 2, (255, 0, 0), 10)
            cv.imshow('draw_raw_blocks', edges_BGR_modified)
            k = cv.waitKey(1000)
            cv.destroyAllWindows()

def draw_blocks(edges, blocks):
    for blockset in blocks:
        intersections = blockset
        edges_BGR_modified = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        cv.circle(edges_BGR_modified, intersections[0], 2, (255, 0, 0), 10)
        cv.circle(edges_BGR_modified, intersections[1], 2, (255, 0, 0), 10)
        cv.circle(edges_BGR_modified, intersections[2], 2, (255, 0, 0), 10)
        cv.circle(edges_BGR_modified, intersections[3], 2, (255, 0, 0), 10)
        cv.imshow('draw_blocks', edges_BGR_modified)
        k = cv.waitKey(1000)
        cv.destroyAllWindows()

def draw_result(img, block):
    for intersections in block:
        top_left = intersections[0]
        bottom_left = intersections[1]
        top_right = intersections[2]
        bottom_right = intersections[3]
        cv.line(img, (top_left[0], top_left[1]), (top_right[0], top_right[1]), (0, 255, 255), 4, cv.LINE_AA)
        cv.line(img, (top_left[0], top_left[1]), (bottom_left[0], bottom_left[1]), (0, 255, 255), 4, cv.LINE_AA)
        cv.line(img, (top_right[0], top_right[1]), (bottom_right[0], bottom_right[1]), (0, 255, 255), 4, cv.LINE_AA)
        cv.line(img, (bottom_left[0], bottom_left[1]), (bottom_right[0], bottom_right[1]), (0, 255, 255), 4, cv.LINE_AA)
        cv.circle(img, intersections[0], 2, (255, 0, 0), 10)
        cv.circle(img, intersections[1], 2, (255, 0, 0), 10)
        cv.circle(img, intersections[2], 2, (255, 0, 0), 10)
        cv.circle(img, intersections[3], 2, (255, 0, 0), 10)
    cv.imshow('draw_result', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


'''
Pipeline
'''
def output_coordinates(block, save_path):
    result = dict()
    index = 0
    for intersections in block:
        center = (round(sum(x for x, y in intersections) / 4.0), round(sum(y for x, y in intersections) / 4.0))
        top_left = intersections[0]
        bottom_left = intersections[1]
        top_right = intersections[2]
        bottom_right = intersections[3]

        width = distance(top_left[0], bottom_left[0], top_left[1], bottom_left[1])
        height = distance(top_left[0], top_right[0], top_left[1], top_right[1])

        if (width > height):
            if (bottom_left[1] - top_left[1] == 0):
                roll = 90
            else:
                roll = round(math.atan((bottom_left[0] - top_left[0]) / (bottom_left[1] - top_left[1])) * 180.0 / np.pi)
        else:
            if (top_left[1] - top_right[1] == 0):
                roll = 90
            else:
                roll = round(math.atan((top_left[0] - top_right[0]) / (top_left[1] - top_right[1])) * 180.0 / np.pi)
        pitch = 0
        yaw = 0
        height = 0
        result[index] = [center, height, roll, pitch, yaw]
        index += 1
    save_file = open(save_path, "w")
    for key in result:
        save_file.write(str(key) + ":" + ','.join(map(str, result[key])) + "\n")
    return result

'''
Main function
'''
def main():
    image_name = sys.argv[1]

    #find and show edges
    [img, threshold, blur, edges] = find_edge(image_name)
    # show_edge(img, threshold, blur, edges)

    #detect block and find their places
    block = detect_block(edges, img)
    draw_result(img, block)
    cv.imwrite("test_3.jpg", img)
    # center, height, roll, pitch, yaw
    output_coordinates(block,"test_3.txt")

main()
