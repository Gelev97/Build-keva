import numpy as np
import math

# All units in cm
BASE_HEIGHT = 6.3
LINK_1 = 3.1
LINK_2 = 42.7
LINK_3 = 45
LINK_4 = 10

DOF = 5

# DH PARAM
a_arr = [0,LINK_2,LINK_3,0,0]
alpha_arr = [-90,0,0,-90,0]
d_arr = [LINK_1,0,0,0,LINK_4]
theta_arr = [0,0,0,0,0]

# Foward Kinematics - return frames
def forward_kinematics(thetas):
    frames = np.zeros((DOF,4,4))
    a = a_arr[0]
    alpha = alpha_arr[0]
    d = d_arr[0]
    theta = theta_arr[0] + thetas[0]
    
    # Calculate the first frame
    frames[0] = np.array([[math.cos(theta),-math.sin(theta)*math.cos(alpha),math.sin(theta)*math.sin(alpha),a*math.cos(theta)],
                         [math.sin(theta),math.cos(theta)*math.cos(alpha),-math.cos(theta)*math.sin(alpha),a*math.sin(theta)],
                         [0,math.sin(alpha),math.cos(alpha),d],
                         [0,0,0,1]])
    
    # Calculate the rest using the first frame
    for frame_index in range(1,DOF):
        a = a_arr[frame_index]
        alpha = alpha_arr[frame_index]
        d = d_arr[frame_index]
        theta = theta_arr[frame_index] + thetas[frame_index]
    
        frames[frame_index] = np.matmul(frames[frame_index-1],\
                              np.array([[math.cos(theta),-math.sin(theta)*math.cos(alpha),math.sin(theta)*math.sin(alpha),a*math.cos(theta)],
                              [math.sin(theta),math.cos(theta)*math.cos(alpha),-math.cos(theta)*math.sin(alpha),a*math.sin(theta)],
                              [0,math.sin(alpha),math.cos(alpha),d],
                              [0,0,0,1]]))
        print(frames[frame_index-1])
        print(np.array([[math.cos(theta),-math.sin(theta)*math.cos(alpha),math.sin(theta)*math.sin(alpha),a*math.cos(theta)],
                                  [math.sin(theta),math.cos(theta)*math.cos(alpha),-math.cos(theta)*math.sin(alpha),a*math.sin(theta)],
                                  [0,math.sin(alpha),math.cos(alpha),d],
                                  [0,0,0,1]]))
    print(frames)
    return frames

# Returns [x; y; theta] for the end effector given a set of joint angles. 
def end_effector(thetas):
    # Find the transform to the end-effector frame.
    frames = forward_kinematics(thetas)
    H_0_ee = frames[DOF-1]
    
    # Extract the components of the end_effector position and
    # orientation.
    x = H_0_ee[0,3]
    y = H_0_ee[1,3]
    z = H_0_ee[2,3]
    roll = math.atan2(H_0_ee[1, 0], H_0_ee[0, 0])
    pitch = -math.asin(H_0_ee[2, 0])
    yaw = math.atan2(H_0_ee[2, 1], H_0_ee[2, 2])
    # Pack them up nicely.
    ee = [x,y,z,roll,pitch,yaw]
    return ee

print(end_effector([0,0,0,0,0]))