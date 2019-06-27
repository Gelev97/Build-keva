from pyrobot import Robot
import numpy as np
import time
robot = Robot('locobot')

target_joints = [
        [0, 0.6, -0.5, 1.3, 0.920],
        [0, 0.6, -0, 0.9, 0.920]
    ]
robot.arm.go_home()
robot.gripper.open()

for joint in target_joints:
    robot.arm.set_joint_positions(joint, plan=False)
    time.sleep(1)
 
robot.gripper.close()
robot.arm.go_home()

target_joints = [
        [-0.675, 0.9, -0.8, 1.3, 0.320],
    ]
robot.arm.go_home()

for joint in target_joints:
    robot.arm.set_joint_positions(joint, plan=False)
    time.sleep(1)
 
robot.gripper.open()
robot.arm.go_home()
