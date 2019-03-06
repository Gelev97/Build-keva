import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math

D_MIN = 30 # search internal radius
D_MAX = 230 #search external radius
'''
Helper Function for search in circle
'''
def is_in_bound(center, direction_radius, width, height):
    check_x = center[0]+direction_radius[0]
    check_y = center[1]+direction_radius[1]
    if(check_x < 0 or check_x > width): return False
    if(check_y < 0 or check_y > height): return False
    return True

#horizontal y, vertical x
def create_search_region(center, width, height):
    region = []
    directions = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]
    for radius in range(D_MIN, D_MAX+1):
        for direction in directions:
            direction_radius = [direction[0]*radius, direction[1]*radius]
            if(is_in_bound(center, direction_radius, width, height)):
                region.append([center[0]+direction_radius[0], center[1]+direction_radius[1]])
    return region

def draw_search_region(img, center, D_MIN, D_MAX):
    cv.circle(img, center, D_MIN, (0,255,0), thickness=1, lineType=8, shift=0)
    cv.circle(img, center, D_MAX, (0, 255, 0), thickness=1, lineType=8, shift=0)


'''
Helper function for Hough Transform
'''


# This is the function that will build the Hough Accumulator for the given image
def hough_lines_acc(search_region, rho_resolution, theta_resolution, height, width):
    ''' A function for creating a Hough Accumulator for lines in an image. '''
    img_diagonal = np.ceil(np.sqrt(height ** 2 + width ** 2))  # a**2 + b**2 = c**2
    rhos = np.arange(-img_diagonal, img_diagonal + 1, rho_resolution)
    thetas = np.deg2rad(np.arange(-90, 90, theta_resolution))
    # create the empty Hough Accumulator with dimensions equal to the size of
    # rhos and thetas
    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

    for i in range(len(search_region)):  # cycle through edge points
        x = search_region[i][0]
        y = search_region[i][1]

        for j in range(len(thetas)):  # cycle through thetas and calc rho
            rho = int(x * np.cos(thetas[j]) + y * np.sin(thetas[j]) + img_diagonal)
            H[rho, j] += 1

    return H, rhos, thetas

# This more advance Hough peaks funciton has threshold and nhood_size arguments
# threshold will threshold the peak values to be above this value if supplied,
# where as nhood_size will surpress the surrounding pixels centered around
# the local maximum after that value has been assigned as a peak.  This will
# force the algorithm to look eslwhere after it's already selected a point from
# a 'pocket' of local maxima.
def hough_peaks(H, num_peaks, threshold=0, nhood_size=3):
    ''' A function that returns the indicies of the accumulator array H that
        correspond to a local maxima.  If threshold is active all values less
        than this value will be ignored, if neighborhood_size is greater than
        (1, 1) this number of indicies around the maximum will be surpessed. '''
    # loop through number of peaks to identify
    indicies = []
    H1 = np.copy(H)
    for i in range(num_peaks):
        idx = np.argmax(H1)  # find argmax in flattened array
        H1_idx = np.unravel_index(idx, H1.shape)  # remap to shape of H
        indicies.append(H1_idx)

        # surpess indicies in neighborhood
        idx_y, idx_x = H1_idx  # first separate x, y indexes from argmax(H)
        # if idx_x is too close to the edges choose appropriate values
        if (idx_x - (nhood_size / 2)) < 0:
            min_x = 0
        else:
            min_x = idx_x - (nhood_size / 2)
        if ((idx_x + (nhood_size / 2) + 1) > H.shape[1]):
            max_x = H.shape[1]
        else:
            max_x = idx_x + (nhood_size / 2) + 1

        # if idx_y is too close to the edges choose appropriate values
        if (idx_y - (nhood_size / 2)) < 0:
            min_y = 0
        else:
            min_y = idx_y - (nhood_size / 2)
        if ((idx_y + (nhood_size / 2) + 1) > H.shape[0]):
            max_y = H.shape[0]
        else:
            max_y = idx_y + (nhood_size / 2) + 1

        # bound each index by the neighborhood size and set all values to 0
        for x in range(int((min_x), int(max_x))):
            for y in range(int(min_y), int(max_y)):
                # remove neighborhoods in H1
                H1[y, x] = 0

                # highlight peaks in original H
                if (x == min_x or x == (max_x - 1)):
                    H[y, x] = 255
                if (y == min_y or y == (max_y - 1)):
                    H[y, x] = 255

    # return the indicies and the original Hough space with selected points
    return indicies, H


# a simple funciton used to plot a Hough Accumulator
def plot_hough_acc(H, plot_title='Hough Accumulator Plot'):
    ''' A function that plot a Hough Space using Matplotlib. '''
    fig = plt.figure(figsize=(10, 10))
    fig.canvas.set_window_title(plot_title)

    plt.imshow(H, cmap='jet')

    plt.xlabel('Theta Direction'), plt.ylabel('Rho Direction')
    plt.tight_layout()
    plt.show()

# drawing the lines from the Hough Accumulatorlines using OpevCV cv2.line
def hough_lines_draw(img, indicies, rhos, thetas):
    ''' A function that takes indicies a rhos table and thetas table and draws
        lines on the input images that correspond to these values. '''
    for i in range(len(indicies)):
        # reverse engineer lines from rhos and thetas
        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # these are then scaled so that the lines go off the edges of the image
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

'''
Main Function
'''
def detect_block(edges):
    # edge with three channel used to display
    edges_BGR = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    edges_BGR_modified = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    width = edges.shape[0]
    height = edges.shape[1]

    (x,y) = (height/2.0, width/2.0)
    (x,y) = (int(x), int(y))
    center = (x,y)
    draw_search_region(edges_BGR, center,  D_MIN, D_MAX)
    search_region = create_search_region(center, width, height)
    print(search_region)

    #Hough Transform
    rho_resolution = 3/4
    theta_resolution = (3*np.pi)/(4*D_MAX)
    # H, rhos, thetas = hough_lines_acc(search_region, rho_resolution, theta_resolution, height, width)
    # # indicies, H = hough_peaks(H, 3, nhood_size=11)  # find peaks
    # plot_hough_acc(H)  # plot hough space, brighter spots have higher votes
    # # hough_lines_draw(edges_BGR, indicies, rhos, thetas)
    #
    # imstack = np.hstack((edges_BGR_modified, edges_BGR))


    rectX = (x - D_MAX)
    rectY = (y - D_MAX)
    crop_img = edges[rectX:(rectX + 2 * D_MAX), rectY:(rectY + 2 * D_MAX)]

    lines = cv.HoughLines(crop_img, rho_resolution, theta_resolution, 100)
    for line in lines:
        (rho, theta) = (line[0][0], line[0][1])
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv.line(edges_BGR_modified, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.rectangle(edges_BGR_modified,(rectX,rectY),(rectX+2*D_MAX, rectY+2*D_MAX),(0,255,0))
    crop_img_BGR = cv.cvtColor(crop_img, cv.COLOR_GRAY2BGR)
    cv.imshow('edges_BGR_modified', edges_BGR_modified)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return
