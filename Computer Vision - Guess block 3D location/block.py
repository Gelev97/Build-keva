import numpy as np
import cv2 as cv
import math
import sys

'''
Define Macro and Tunable Variable
'''
# Params to find edges
MIN_LINE_LENGTH = 15
MAX_LINE_GAP = 30
NUMBER_OF_INTERSECTION = 20# The probability value for the hough line transform

ANGLE_GAP = 5# Degree difference, we make angles same wihtin these gap
PERPENDICULAR_THRES = 0.5# Threshold tolerance for perpendiculars
INTERSECT_THRES = 20# Distance threshqqqqqqqqqqqqqqqqold to determine if two line intersect

MAX_LINE_LENGTH = 300
WIDTH_HEIGHT_DIST_MIN = 20.0
WIDTH_HEIGHT_DIST_MAX = 400.0

D_MAX = 230 # search radius

WIDTH_MIN = 65
WIDTH_MAX = 75
HEIGHT_MIN = 330
HEIGHT_MAX = 340

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
Find Perpendicular line
'''
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
                (intersectionX, intersectionY, valid) = intersectLines(line,line_to_compare)
                if(valid):
                   perpendicular_line_group[unique_index].append((line_to_compare,(intersectionX,intersectionY)))
    return perpendicular_line_group

'''
Function used to extend and shift the finding hough line
'''
def extend_and_shift(l, rectX, rectY):
    result_line = [0,0,0,0]
    # shift the line based on crop center
    l[0] = l[0] + rectY
    l[1] = l[1] + rectX
    l[2] = l[2] + rectY
    l[3] = l[3] + rectX

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

def check_parallel(line1, line2, parallel_line_group):
    # to check whether line1 and line2 are parallel
    for group in parallel_line_group:
        if (len(group) > 1):
            if(line1 in group and line2 in group):
                return True
    return False

def distance(p0, p1, p2, p3):
    return math.sqrt((p0 - p1) ** 2 + (p2 - p3) ** 2)

def check_intersections(intersections):
    # find if the intersections are forming a rectangle
    top_left = intersections[0]
    bottom_left = intersections[1]
    top_right = intersections[2]
    bottom_right = intersections[3]

    # calculate four side length
    lenght_left = distance(top_left[0],bottom_left[0],top_left[1],bottom_left[1])
    lenght_right = distance(top_right[0], bottom_right[0], top_right[1], bottom_right[1])
    lenght_top = distance(top_left[0], top_right[0], top_left[1], top_right[1])
    lenght_bottom = distance(bottom_left[0], bottom_right[0], bottom_left[1], bottom_right[1])

    # check length
    width = round((lenght_left + lenght_right)/2.0)
    height = round((lenght_top + lenght_bottom) / 2.0)
    if(width > WIDTH_MAX or width < WIDTH_MIN ): return False
    if(height > HEIGHT_MAX or height < HEIGHT_MIN): return False
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
                        line_1 = line_index_dict[key]
                        angle_1 = round(math.atan2(line_1[3] - line_1[1], line_1[2] - line_1[0]) * 180.0 / np.pi)
                        line1 = (line_1, angle_1)
                        line_2 = line_index_dict[key_compare]
                        angle_2 = round(math.atan2(line_2[3] - line_2[1], line_2[2] - line_2[0]) * 180.0 / np.pi)
                        line2 = (line_2, angle_2)
                        if(check_parallel(line1, line2, parallel_line_group)):
                            raw_blocks.append((line1, line2, permute_common))
                    else:
                        continue

    # filter out not keva block
    for raw_block in raw_blocks:
        # check each perpendicular group
        for block_common_2 in raw_block[2]:
            # four intersections of the rectangle
            intersections = [block_common_2[0][1], block_common_2[0][2], block_common_2[1][1], block_common_2[1][2]]
            if(check_intersections(intersections)):
                #draw_rectangular(edges, intersections)
                blocks.append((raw_block[0], raw_block[1], intersections))

    for block in blocks:
        print(block)

    # # get two parallel line of the block
    # line_1 = block[0]
    # line_2 = block[1]
    return blocks
'''
Pipeline to find block
'''
def detect_block(edges, blur):
    # edge with three channel used to display
    edges_BGR = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    edges_BGR_modified = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    width = edges.shape[0]
    height = edges.shape[1]
    (x, y) = (width / 2.0, height / 2.0)
    (x, y) = (int(x), int(y))
    center = (x, y)

    # Hough Transform
    rho_resolution = 3 / 4
    theta_resolution = (3 * np.pi) / (4 * D_MAX)

    rectX = (x - D_MAX)
    rectY = (y - D_MAX)
    crop_img = blur[rectX:(rectX + 2 * D_MAX), rectY:(rectY + 2 * D_MAX)]
    # Detect edges using canny method and show the result
    edges_crop = cv.Canny(crop_img, CANNY_EDGE_LOWER_THRES, CANNY_EDGE_UPPER_THRES)

    # find lines using hough lines transform with probability
    linesP = cv.HoughLinesP(edges_crop, 1, np.pi / 180, NUMBER_OF_INTERSECTION, None, MIN_LINE_LENGTH, MAX_LINE_GAP)

    # array that contains all lines
    lines = []

    # A dictionary that maps every line with an unique index
    line_index_dict = dict()
    index_dict = 0

    # Hashmap all lines with its angle
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            l = extend_and_shift(l, rectX, rectY)
            lines.append(list(l))
            line_index_dict[index_dict] = l
            index_dict += 1
            cv.line(edges_BGR, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 1, cv.LINE_AA)

    # [[(line, angle), (line, angle)], [(line, angle), (line, angle)], ...]
    parallel_line_group = angle_approximation(lines)

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

    # Delete alone parallel line or empty group after above operation
    index_remove_group = 0
    while(index_remove_group != len(parallel_line_group)):
        group = parallel_line_group[index_remove_group]
        if (len(group) <= 1):
            parallel_line_group.remove(group)
        else:
            index_remove_group += 1

    perpendicular_line_group = perpendicular_approximation(line_index_dict, lines)

    # Draw functions used to debug
    #draw_parallel_line(edges, parallel_line_group)
    #draw_perpendicular_line(edges, perpendicular_line_group, line_index_dict)

    block_group = find_rectangle(edges, parallel_line_group, perpendicular_line_group, line_index_dict)
    #
    # #draw lines
    # for block in block_group:
    #     for pair in block:
    #         line = pair[0]
    #         cv.line(edges_BGR_modified, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 1, cv.LINE_AA)
    cv.rectangle(edges_BGR, (rectY, rectX), (rectY + 2*D_MAX, rectX + 2*D_MAX), (255,0,255))
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
    detect_block(edges, blur)

main()

