import numpy as np
import cv2 as cv
import math
import sys
from operator import itemgetter

'''
Define Macro and Tunable Variable
'''
# Params to find edges
MIN_LINE_LENGTH = 15
MAX_LINE_GAP = 30
NUMBER_OF_INTERSECTION = 25# The probability value for the hough line transform

ANGLE_GAP = 5# Degree difference, we make angles same wihtin these gap
PERPENDICULAR_THRES = 7# Threshold tolerance for perpendiculars

MAX_LINE_LENGTH = 300
WIDTH_HEIGHT_DIST_MIN = 60.0
WIDTH_HEIGHT_DIST_MAX = 400.0

# Params to check block length
WIDTH_MIN = 60.0
WIDTH_MAX = 80.0
HEIGHT_MIN = 330.0
HEIGHT_MAX = 380.0

# Params to filter the line
FAKE_RANGE = 250
FAKE_PARALLEL_GAP = 10

# Same Center threshold
CENTER_SAME_THRESHOLD = 10

# Check fake length
FAKE_LENGTH_THRE = 25
SIMILARITY = 0.8

# Threshold used to get edges
THRES = 1.2
BLUR_LEVEL = 3
CANNY_EDGE_LOWER_THRES = 100
CANNY_EDGE_UPPER_THRES = 120

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
    line_index = 0
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

    #print(distance_line)
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
            # print(angle, angle_compare)
            threshold_value = abs(angle - angle_compare)
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
    dx1 = x2 - x1;
    dy1 = y2 - y1

    # the second line is ptA + s*(ptB-ptA)
    x, y = (line_to_compare[0], line_to_compare[1]);
    xB, yB = (line_to_compare[2], line_to_compare[3]);
    dx = xB - x;
    dy = yB - y;

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

    if(left[0][1] > left[1][1]):
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
    # print(lenght_left, lenght_right, lenght_top, lenght_bottom)
    # check length
    if(lenght_left <= lenght_top and lenght_right <= lenght_bottom):
        if(lenght_left > WIDTH_MAX or lenght_left < WIDTH_MIN ): return False
        if(lenght_top > HEIGHT_MAX or lenght_top < HEIGHT_MIN): return False
        if(lenght_right > WIDTH_MAX or lenght_right < WIDTH_MIN): return False
        if(lenght_bottom > HEIGHT_MAX or lenght_bottom < HEIGHT_MIN): return False
    else:
        if(lenght_left > HEIGHT_MAX or lenght_left < HEIGHT_MIN ): return False
        if(lenght_top > WIDTH_MAX or lenght_top < WIDTH_MIN): return False
        if(lenght_right > HEIGHT_MAX or lenght_right < HEIGHT_MIN): return False
        if(lenght_bottom > WIDTH_MAX or lenght_bottom < WIDTH_MIN): return False

    return True


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
    #draw_raw_blocks(edges, raw_blocks)

    # filter out not keva block
    for raw_block in raw_blocks:
        # check each perpendicular group
        intersection = [[0,0],[0,0],[0,0],[0,0]]
        intersection_count = 0
        for block_common_2 in raw_block[2]:
            # four intersections of the rectangle
            intersections = [block_common_2[0][1], block_common_2[0][2], block_common_2[1][1], block_common_2[1][2]]
            # edges_BGR_modified = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
            # cv.circle(edges_BGR_modified, intersections[0], 2, (255, 0, 0), 2)
            # cv.circle(edges_BGR_modified, intersections[1], 2, (255, 0, 0), 2)
            # cv.circle(edges_BGR_modified, intersections[2], 2, (255, 0, 0), 2)
            # cv.circle(edges_BGR_modified, intersections[3], 2, (255, 0, 0), 2)
            # cv.imshow('perpendicular', edges_BGR_modified)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            if(check_intersections(intersections)):
               blocks.append((raw_block[0], raw_block[1], intersections))

    # test blocks after filter
    # draw_blocks(edges, blocks)

    # remove duplicate
    block_dict = dict()
    center_group_index = 0
    for block in blocks:
        # get the current block center
        new_flag = 1
        intersections = block[2]
        center = (round(sum(x for x, y in intersections)/4.0), round(sum(y for x, y in intersections)/4.0))
        for center_index in block_dict:
            # compare it with all center in this group
            qualify_flag = 1
            for intersection_compare in block_dict[center_index]:
                center_compare = (round(sum(x for x, y in intersection_compare)/4.0), round(sum(y for x, y in intersection_compare)/4.0))
                if(distance(center[0], center_compare[0], center[1], center_compare[1]) > CENTER_SAME_THRESHOLD):
                    qualify_flag = 0
            if(qualify_flag == 1):
                block_dict[center_index].append(intersections)
                new_flag = 0
        # If no group fit, just create a new group
        if(new_flag == 1):
            block_dict[center_group_index]  = [intersections]
            center_group_index += 1

    # edges_BGR_modified = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    # for center in block_dict:
    #     intersections = block_dict[center]
    #     edges_BGR_modified = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    #     for intersection in intersections:
    #         center_show = (round(sum(x for x, y in intersection) / 4.0), round(sum(y for x, y in intersection) / 4.0))
    #         cv.circle(edges_BGR_modified, center_show, 2, (255, 0, 0), 10)
    #     cv.imshow('perpendicular', edges_BGR_modified)
    #     cv.waitKey(0)
    #     cv.destroyAllWindows()

    # merge duplicate
    result = []
    for center in block_dict:
        intersection_group = block_dict[center]
        intersection = [[0, 0], [0, 0], [0, 0], [0, 0]]
        intersections = [(0,0),(0,0),(0,0),(0,0)]
        intersection_count = 0

        # edges_BGR_modified = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        for intersection_merge in intersection_group:
            # cv.circle(edges_BGR_modified, intersection_merge[0], 2, (255, 0, 0), 2)
            # cv.circle(edges_BGR_modified, intersection_merge[1], 2, (255, 0, 0), 2)
            # cv.circle(edges_BGR_modified, intersection_merge[2], 2, (255, 0, 0), 2)
            # cv.circle(edges_BGR_modified, intersection_merge[3], 2, (255, 0, 0), 2)
        # cv.imshow('perpendicular', edges_BGR_modified)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
            intersection_merge.sort(key=itemgetter(0))
            left = [intersection_merge[0], intersection_merge[1]]
            right = [intersection_merge[2], intersection_merge[3]]

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

            intersection[0][0] += top_left[0]
            intersection[0][1] += top_left[1]
            intersection[1][0] += bottom_left[0]
            intersection[1][1] += bottom_left[1]
            intersection[2][0] += top_right[0]
            intersection[2][1] += top_right[1]
            intersection[3][0] += bottom_right[0]
            intersection[3][1] += bottom_right[1]

            intersection_count += 1
        if (intersection_count != 0):
            intersections[0] = (round(intersection[0][0] / intersection_count), round(intersection[0][1] / intersection_count))
            intersections[1] = (round(intersection[1][0] / intersection_count), round(intersection[1][1] / intersection_count))
            intersections[2] = (round(intersection[2][0] / intersection_count), round(intersection[2][1] / intersection_count))
            intersections[3] = (round(intersection[3][0] / intersection_count), round(intersection[3][1] / intersection_count))
            result.append(intersections)

    # for intersection in result:
    #     draw_rectangular(edges, intersection)
    return result

'''
Delete fake block

'''
def create_line_arr_height(line):
    # extend the line gradually increase x by 1
    # when y1 = y2
    if ((line[1] - line[3]) == 0):
        line_arr = []
        if (line[0] < line[2]):
            for x in range(line[0], line[2]+1):
                line_arr.append((x, line[1]))
        else:
            for x in range(line[0], line[2]-1, -1):
                line_arr.append((x, line[1]))
        return line_arr

    slope = (line[0] - line[2]) / (line[1] - line[3])

    intercept = line[0] - line[1]*slope
    line_arr = [(line[0], line[1])]
    if(line[0] < line[2]):
        for x in range(line[0]+1, line[2]):
            y = round((x - intercept)/slope)
            line_arr.append((x,y))
        line_arr.append((line[2],line[3]))
    else:
        for x in range(line[0]+1, line[2], -1):
            y = round((x - intercept) / slope)
            line_arr.append((x,y))
    line_arr.append((line[2], line[3]))
    return line_arr


def create_line_arr_width(line):
    # extend the line gradually increase y by 1
    # when x1 = x2
    if((line[0] - line[2]) == 0):
        line_arr = []
        if(line[1] < line[3]):
            for y in range(line[1], line[3]+1):
                line_arr.append((line[0], y))
        else:
            for y in range(line[1], line[3]-1, -1):
                line_arr.append((line[0], y))
        return line_arr

    slope = (line[0] - line[2]) / (line[1] - line[3])

    intercept = line[0] - line[1] * slope
    line_arr = [(line[0], line[1])]
    if (line[1] < line[3]):
        for y in range(line[1]+1, line[3]):
            x = round(slope*y + intercept)
            line_arr.append((x,y))
        line_arr.append((line[2], line[3]))
    else:
        for y in range(line[1]+1, line[3], -1):
            x = round(slope * y + intercept)
            line_arr.append((x,y))
    line_arr.append((line[2], line[3]))
    return line_arr

def parallel_check(qualified_line,line_compare):
    # deal with division by 0
    if (qualified_line[3] - qualified_line[1] == 0):
        angle = 90
    else:
        angle = round(math.atan((qualified_line[2] - qualified_line[0]) / (qualified_line[3] - qualified_line[1])) * 180.0 / np.pi)
    if (line_compare[3] - line_compare[1] == 0):
        angle_compare = 90
    else:
        angle_compare = round(math.atan(
            (line_compare[2] - line_compare[0]) / (line_compare[3] - line_compare[1])) * 180.0 / np.pi)
    if (abs(angle - angle_compare) > FAKE_PARALLEL_GAP and abs(angle - angle_compare) < 180 - FAKE_PARALLEL_GAP):
        return False
    return True

def clear_fake_block(block,line_not_extend, edges):
    result = []
    length = len(block)
    count = 0
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

        center = (round(sum(x for x, y in intersections)/4.0), round(sum(y for x, y in intersections)/4.0))
        lines_compare = []

        for line in line_not_extend:
            flag = 1
            center_x = center[0]
            center_y = center[1]
            center_x_min = center_x - FAKE_RANGE
            center_x_max = center_x + FAKE_RANGE
            center_y_min = center_y - FAKE_RANGE
            center_y_max = center_y + FAKE_RANGE
            if(line[0] < center_x_min or line[0] > center_x_max): flag = 0
            if(line[1] < center_y_min or line[1] > center_y_max): flag = 0
            if(line[2] < center_x_min or line[2] > center_x_max): flag = 0
            if(line[3] < center_y_min or line[3] > center_y_max): flag = 0
            if(flag == 1):
                lines_compare.append(line)
                # cv.line(edges_BGR_modified, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 1, cv.LINE_AA)
        # cv.circle(edges_BGR_modified, intersections[0], 2, (255, 0, 0), 2)
        # cv.circle(edges_BGR_modified, intersections[1], 2, (255, 0, 0), 2)
        # cv.circle(edges_BGR_modified, intersections[2], 2, (255, 0, 0), 2)
        # cv.circle(edges_BGR_modified, intersections[3], 2, (255, 0, 0), 2)

        for line_compare in lines_compare:
            for qualified_line_set in lines:
                # edges_BGR_modified = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
                qualified_line = qualified_line_set[1]
                if(parallel_check(qualified_line,line_compare)):
                    # extend the line by 1
                    if(qualified_line_set[0] == 1):
                        qualified_line_arr = create_line_arr_width(qualified_line)
                    else:
                        qualified_line_arr = create_line_arr_height(qualified_line)
                    if(abs(line_compare[0]-line_compare[2]) < abs(line_compare[1] - line_compare[3])):
                        line_compare_arr = create_line_arr_width(line_compare)
                    else:
                        line_compare_arr = create_line_arr_height(line_compare)

                    # find similarity
                    len_qualified = len(qualified_line_arr)
                    qualified_similar = set()
                    len_compare = len(line_compare_arr)
                    compare_similar = set()
                    for qualified_line_element in qualified_line_arr:
                        for line_compare_element in line_compare_arr:
                            if(distance(qualified_line_element[0],line_compare_element[0],\
                                        qualified_line_element[1], line_compare_element[1]) < FAKE_LENGTH_THRE):
                                qualified_similar.add(qualified_line_element)
                                compare_similar.add(line_compare_element)
                        if(len(qualified_similar)/len_qualified > SIMILARITY or len(compare_similar)/len_compare > SIMILARITY):
                           if(qualified_line not in line_test):
                               line_test.append(qualified_line)
                    # print(len(qualified_similar)/len_qualified, len(compare_similar)/len_compare)
                    # cv.line(edges_BGR_modified, (qualified_line[0], qualified_line[1]),(qualified_line[2], qualified_line[3]), (255, 0, 0), 1, cv.LINE_AA)
                    # cv.line(edges_BGR_modified, (line_compare[0], line_compare[1]), (line_compare[2], line_compare[3]), (0, 0, 255), 1, cv.LINE_AA)
                    # cv.imshow('perpendicular', edges_BGR_modified)
                    # cv.waitKey(0)
                    # cv.destroyAllWindows()
            # print(line_test)
            # get rid of qualified lines
            for line_no_need in line_test:
                if((0,line_no_need) in lines):
                    lines.remove((0,line_no_need))
                if((1,line_no_need) in lines):
                    lines.remove((1,line_no_need))
            # print(len(lines))
        if(len(line_test) == 4): result.append(intersections)
        count += 1
        print(str(round(count/length*100)) + "%")
    return result
'''
Pipeline to find block
'''
def detect_block(edges):
    # edge with three channel used to display
    edges_BGR = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    edges_BGR_modified = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    width = edges.shape[0]
    height = edges.shape[1]
    block= []

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
            # edges_BGR = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
            cv.line(edges_BGR, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 1, cv.LINE_AA)
            # cv.imshow('hough', edges_BGR)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

    # [(line_index, angle) : [line,line,line], ...]
    parallel_line_group = angle_approximation(lines)

    # Delete parallel lines that not form keva block
    for line_set in parallel_line_group:
        line = line_index_dict[line_set[0]]
        group = parallel_line_group[line_set]
        if (len(group) > 1):
            line_parallel_index = 0
            while(line_parallel_index < len(group)):
                # edges_BGR = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
                # cv.line(edges_BGR, (line[0], line[1]), (line[2], line[3]),(255, 255, 0), 1, cv.LINE_AA)
                line_parallel = group[line_parallel_index]
                # cv.line(edges_BGR, (line_parallel[0], line_parallel[1]), (line_parallel[2], line_parallel[3]),(255, 255, 0), 1, cv.LINE_AA)
                # cv.imshow('hough', edges_BGR)
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
    index_remove_group = 0
    for line_set in list(parallel_line_group.keys()):
        if(len(parallel_line_group[line_set]) <= 1):
            del parallel_line_group[line_set]

    perpendicular_line_group = perpendicular_approximation(parallel_line_group, line_index_dict, lines, edges)

    # Draw functions used to debug
    # draw_parallel_line(edges, parallel_line_group, line_index_dict)
    # draw_perpendicular_line(edges, perpendicular_line_group, line_index_dict)

    block = find_rectangle(edges, parallel_line_group, perpendicular_line_group, line_index_dict)

    # Clear fake block
    block = clear_fake_block(block,line_not_extend, edges)

    # Draw all the blocks found
    for intersections in block:
        cv.circle(edges_BGR_modified, intersections[0], 2, (255, 0, 0), 10)
        cv.circle(edges_BGR_modified, intersections[1], 2, (255, 0, 0), 10)
        cv.circle(edges_BGR_modified, intersections[2], 2, (255, 0, 0), 10)
        cv.circle(edges_BGR_modified, intersections[3], 2, (255, 0, 0), 10)
        top_left = intersections[0]
        bottom_left = intersections[1]
        top_right = intersections[2]
        bottom_right = intersections[3]
        cv.line(edges_BGR_modified, (top_left[0], top_left[1]), (top_right[0], top_right[1]), (0, 0, 255), 1, cv.LINE_AA)
        cv.line(edges_BGR_modified, (top_left[0], top_left[1]), (bottom_left[0], bottom_left[1]), (0, 0, 255), 1, cv.LINE_AA)
        cv.line(edges_BGR_modified, (top_right[0], top_right[1]), (bottom_right[0], bottom_right[1]), (0, 0, 255), 1, cv.LINE_AA)
        cv.line(edges_BGR_modified, (bottom_left[0], bottom_left[1]), (bottom_right[0], bottom_right[1]), (0, 0, 255), 1, cv.LINE_AA)
    imstack = np.hstack((edges_BGR_modified, edges_BGR))
    cv.imshow('stack', imstack)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return

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
        if(len(group) > 1):
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
            edges_BGR_modified = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

def draw_rectangular(edges, intersections):
        edges_BGR_modified = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        cv.circle(edges_BGR_modified, intersections[0], 2, (255, 0, 0), 10)
        cv.circle(edges_BGR_modified, intersections[1], 2, (255, 0, 0), 10)
        cv.circle(edges_BGR_modified, intersections[2], 2, (255, 0, 0), 10)
        cv.circle(edges_BGR_modified, intersections[3], 2, (255, 0, 0), 10)
        cv.imshow('perpendicular', edges_BGR_modified)
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
            cv.imshow('perpendicular', edges_BGR_modified)
            k = cv.waitKey(1000)
            cv.destroyAllWindows()

def draw_blocks(edges, blocks):
    for blockset in blocks:
        line1 = blockset[0]
        line2 = blockset[1]
        intersections = blockset[2]
        edges_BGR_modified = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        cv.line(edges_BGR_modified, (line1[0], line1[1]), (line1[2], line1[3]), (255, 0, 0), 1, cv.LINE_AA)
        cv.line(edges_BGR_modified, (line2[0], line2[1]), (line2[2], line2[3]), (255, 0, 0), 1, cv.LINE_AA)
        cv.circle(edges_BGR_modified, intersections[0], 2, (255, 0, 0), 10)
        cv.circle(edges_BGR_modified, intersections[1], 2, (255, 0, 0), 10)
        cv.circle(edges_BGR_modified, intersections[2], 2, (255, 0, 0), 10)
        cv.circle(edges_BGR_modified, intersections[3], 2, (255, 0, 0), 10)
        cv.imshow('perpendicular', edges_BGR_modified)
        k = cv.waitKey(1000)
        cv.destroyAllWindows()


'''
Main function
'''
def main():
    if __name__ == '__main__':
        image_name = sys.argv[1]

    #find and show edges
    [img, threshold, blur, edges] = find_edge(image_name)
    #show_edge(img, threshold, blur, edges)

    #detect block and find their places
    detect_block(edges)

main()

