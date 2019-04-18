import numpy as np
from scipy import interpolate
from scipy import optimize

import math

# All units in cm
BASE_HEIGHT = 6.3
LINK_1 = 3.1
LINK_2 = 42.7
LINK_3 = 45
LINK_4 = 10

# Helper Rotation Matrix
def trans_x(x):
    return np.around(np.array([[1, 0, 0, x],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),decimals = 3)

def trans_y(y):
    return np.around(np.array([[1, 0, 0, 0],
                     [0, 1, 0, y],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),decimals = 3)

def trans_z(z):
    return np.around(np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]]),decimals = 3)

def rotation_x(theta):
    return np.around(np.array([[1,0,0,0],
                     [0,math.cos(theta),-math.sin(theta),0],
                     [0,math.sin(theta),math.cos(theta),0],
                     [0,0,0,1]]),decimals = 3)

def rotation_y(theta):
    return np.around(np.array([[math.cos(theta), 0, math.sin(theta),0],
                     [0, 1, 0, 0],
                     [-math.sin(theta), 0, math.cos(theta),0],
                     [0,0,0,1]]),decimals = 3)

def rotation_z(theta):
    return np.around(np.array([[math.cos(theta), -math.sin(theta), 0, 0],
                     [math.sin(theta), math.cos(theta), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),decimals = 3)


# Foward Kinematics - return frames
def forward_kinematics(thetas):
    frames = np.zeros((6,4,4))

    # Calculate the first frame
    theta = thetas[0]
    frames[0] = rotation_z(theta)

    # Calculate the Second frame
    theta = thetas[1]
    frames[1] =  np.matmul(np.matmul(frames[0], trans_z(LINK_1+BASE_HEIGHT)), rotation_y(-theta))

    # Calculate the third frame
    theta = thetas[2]
    frames[2] = np.matmul(np.matmul(frames[1], trans_x(LINK_2)), rotation_y(theta))

    # Calculate the forth frame
    theta = thetas[3]
    frames[3] = np.matmul(np.matmul(frames[2], trans_x(LINK_3)), rotation_y(theta))

    # Calculate the fifth frame
    theta = thetas[4]
    frames[4] = np.matmul(frames[3], rotation_x(theta))

    # Calculate the sixth frame
    frames[5] = np.matmul(frames[4], trans_x(LINK_4))

    #print(frames)
    return frames

# Inverser Kinmatics
def inverse_kinematics(x,y,z,roll):
    def error(theta, x, y, z, roll):
        goal_position = [x, y, z, roll]
        actual_pos = end_effector(theta)
        actual_position = actual_pos[0:4]
        # print(goal_position)
        # err = np.sqrt(sum((np.array(goal_position) - np.array(actual_position)) ** 2))
        err = np.sqrt(sum(((np.array(goal_position) - np.array(actual_position)) ** 2)))
        print(err)
        return math.fabs(err)

    bound = [[0, math.pi], [0, math.pi], [0, math.pi] \
        , [0, math.pi / 2], [-math.pi / 4, math.pi / 4]]

    res = optimize.differential_evolution(func=error, args=(x, y, z, roll), bounds=bound, disp=False)
    result = res.x
    return np.around(np.array(result), decimals=3)

# Returns [x; y; theta] for the end effector given a set of joint angles.
def end_effector(thetas):
    # Find the transform to the end-effector frame.
    frames = forward_kinematics(thetas)
    H_0_ee = frames[-1]

    # Extract the components of the end_effector position and
    # orientation.
    x = H_0_ee[0,3]
    y = H_0_ee[1,3]
    z = H_0_ee[2,3]
    roll = round(math.atan2(H_0_ee[1, 0], H_0_ee[0, 0]),3)

    # Pack them up nicely.
    ee = [x, y, z, roll]

    # pitch = -round(math.asin(H_0_ee[2, 0]),3)
    # yaw = round(math.atan2(H_0_ee[2, 1], H_0_ee[2, 2]),3)
    # ee = [x,y,z,roll,pitch,yaw]
    return ee

# Spline trajectory
def create_trajecotry(current_position, goal_position):
    x = np.array([current_position[0], goal_position[0]])
    y = np.array([current_position[1], goal_position[1]])
    z = np.array([current_position[2], goal_position[2]])
    current_position = [x,y,z]

    result = interpolate.splprep(current_position, s=0, k = 1)
    x_i,y_i,z_i = interpolate.splev(np.linspace(0,1,10),result[0])
    trajectory = np.zeros((len(x_i),3))
    for index in range(0, len(x_i)):
        trajectory[index] = [x_i[index], y_i[index], z_i[index]]
    print(trajectory)
    return trajectory

# print(end_effector([0,math.pi/2,math.pi/2,0,0]))
# result = inverse_kinematics(50.0,20.0,52.1,0)
# print(result)
# print(end_effector(result))
trajecotry = create_trajecotry([0,0,0], [1,1,1])
angle_group = []
for waypoint in trajecotry:
    result = inverse_kinematics(waypoint[0], waypoint[1], waypoint[2], 0)
    angle_group.append(result)
for angle in angle_group:
    print(end_effector(angle))


