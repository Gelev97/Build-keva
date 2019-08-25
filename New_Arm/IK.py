import numpy as np
from scipy import interpolate
import math
import socket
from pyrobot import Robot
import time

GRAB_HEIGHT = 22
GRAB_ADJUST_HEIGHT = 35

# Spline trajectory
def create_trajectory(current_position, goal_position):
    x = np.array([current_position[0], goal_position[0]])
    y = np.array([current_position[1], goal_position[1]])
    z = np.array([current_position[2], goal_position[2]])
    current_position = [x, y, z]

    result = interpolate.splprep(current_position, s=0, k=1)
    x_i, y_i, z_i = interpolate.splev(np.linspace(0, 1, 10), result[0])
    trajectory = np.zeros((len(x_i), 3))
    for index in range(0, len(x_i)):
        trajectory[index] = [x_i[index], y_i[index], z_i[index]]
    print(trajectory)
    return trajectory

# Pipeline RPC
def pipline_position_encoder(start_position, end_position, s, adjust=0, fifth_angle=0):
    # create trajectory
    trajectory = create_trajectory(start_position, end_position)

    # Calculate Inverse Kinematics
    angle_group = []
    previous_theta = [0,0,0,0,0]
    for waypoint in trajectory:
        tx_data = '%f,%f,%f,%d,%d,%d' % (waypoint[0], waypoint[1], waypoint[2], previous_theta[0], \
                                         previous_theta[1], previous_theta[2])
        s.send(tx_data.encode())

        # Receive from socket (wait here)
        rx_data = s.recv(1024).decode()
        if rx_data == '':  # terminate
            print("Connection dropped.")
            break

        print(rx_data)  # print-out
        string_theta = rx_data.split(",")
        theta = [float(string_theta[0]),float(string_theta[1]),float(string_theta[2])]
        previous_theta = theta

        if(adjust):			
            theta.append(math.pi/2 - theta[1] - theta[2])
            theta.append(fifth_angle)
        else:
            theta.append(0)
            theta.append(0)

        angle_group.append(theta)

    # Change to encoder value
    return angle_group

def grab(robot, s, grab_position, roll):
    # Extract current position
    x = robot.arm.pose_ee[0][0][0] * 100
    y = robot.arm.pose_ee[0][1][0] * 100
    z = robot.arm.pose_ee[0][2][0] * 100

    # Go to Adjust Height
    trajectory_adjust = pipline_position_encoder([x,y,z], [grab_position[0],grab_position[1],GRAB_ADJUST_HEIGHT], s)
    trajectory_grab = pipline_position_encoder([grab_position[0],grab_position[1],GRAB_ADJUST_HEIGHT], [grab_position[0],grab_position[1],GRAB_HEIGHT], s, 1, roll)

    for joint in trajectory_adjust:
        robot.arm.set_joint_positions(joint, plan=False)
        time.sleep(0.5)
    
    robot.gripper.open()

    for joint in trajectory_grab:
        robot.arm.set_joint_positions(joint, plan=False)
        time.sleep(0.5)
    
    robot.gripper.close()
    robot.arm.go_home()

def put(robot, s, put_position, release_height, roll):
    x = robot.arm.pose_ee[0][0][0] * 100
    y = robot.arm.pose_ee[0][1][0] * 100
    z = robot.arm.pose_ee[0][2][0] * 100

    # Go to Adjust Height
    trajectory_adjust = pipline_position_encoder([x,y,z], put_position, s)
    trajectory_put = pipline_position_encoder([put_position[0],put_position[1],GRAB_ADJUST_HEIGHT], [put_position[0],put_position[1], release_height], s, 1, roll)

    for joint in trajectory_adjust:
        print(joint)
        robot.arm.set_joint_positions(joint, plan=False)
        time.sleep(0.5)
    
    for joint in trajectory_put:
        print(joint)
        robot.arm.set_joint_positions(joint, plan=False)
        time.sleep(0.5)
    
    robot.gripper.open()
    robot.arm.go_home()

    
