import numpy as np
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
    frames[1] =  np.matmul(np.matmul(frames[0], trans_z(LINK_1+BASE_HEIGHT)), rotation_y(theta))

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

    print(frames)
    return frames

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
    pitch = -round(math.asin(H_0_ee[2, 0]),3)
    yaw = round(math.atan2(H_0_ee[2, 1], H_0_ee[2, 2]),3)
    # Pack them up nicely.
    ee = [x,y,z,roll,pitch,yaw]
    return ee

print(end_effector([math.pi/2,0,-math.pi/2,0,0]))