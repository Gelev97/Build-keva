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

    #print(frames)
    return frames

# Inverser Kinmatics
def inverse_kinematics(goal_position, initial_theta):
    def error_function(theta):
        actual_pos = end_effector(theta)
        actual_position = actual_pos[0:3]
        err = sum((np.array(goal_position) - np.array(actual_position))**2)
        print(err)
        return -err

    def eqcon(theta):
        actual_pos = end_effector(theta)
        actual_position = actual_pos[0:3]
        err = sum((np.array(goal_position) - np.array(actual_position))**2)
        return np.array([err])

    bound = [[-math.pi*2,math.pi*2],[-math.pi*2,math.pi*2],[-math.pi*2,math.pi*2]\
             ,[-math.pi*2,math.pi*2],[-math.pi*2,math.pi*2]]

    res = optimize.fmin_slsqp(error_function, x0 = initial_theta)
    print(res)

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

# print(end_effector([math.pi/2,0,-math.pi/2,0,0]))
# print(inverse_kinematics([1,1,0], [0,0,0,0,0]))

def transform(theta):
    theta_c = np.matmul(theta, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]))
    theta_c = np.matmul(theta_c, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
    theta_c = np.matmul(theta_c, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
    theta_c = np.matmul(theta_c, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
    return theta_c
def ha(theta,x,y,z):
    def fun(theta,x,y,z):
        """
        Arguments:
        d     - A list of two elements, where d[0] represents x and d[1] represents y
                 in the following equation.
        sign - A multiplier for f.  Since we want to optimize it, and the scipy
               optimizers can only minimize functions, we need to multiply it by
               -1 to achieve the desired solution
        Returns:
        2*x*y + 2*x - x**2 - 2*y**2

        """
        # x = d[0]
        # y = d[1]
        # print(d)
        # # print(2 * x * y + 2 * x - x ** 2 - 2 * y ** 2)
        # return -1*(2 * x * y + 2 * x - x ** 2 - 2 * y ** 2)
        # print(theta)
        goal_position = [x,y,z]
        theta_c = transform(theta)
        # print(theta_c)
        # actual_pos = end_effector(theta)
        # actual_position = actual_pos[0:3]
        # print(goal_position)
        # err = np.sqrt(sum((np.array(goal_position) - np.array(actual_position)) ** 2))
        err = np.sqrt(sum((np.array(goal_position) - np.array(theta_c)) ** 2))
        print(err)
        print(theta)
        return math.fabs(err)

    res = optimize.fmin_slsqp(func = fun, x0 = theta, args = (x,y,z))
    # print(res)
    # print(res[0])
# print(end_effector([1,1,1,1,1]))
print(ha([0,0,0,3,4],2,4,10))
# print(ha([-1.0,1.0]))

# def inv_kin(self, xy):
#     def distance_to_default(q, *args):
#         # weights found with trial and error, get some wrist bend, but not much
#         weight = [1, 1, 1.3]
#         return np.sqrt(np.sum([(qi - q0i) ** 2 * wi
#                                for qi, q0i, wi in zip(q, self.q0, weight)]))
#
#     def x_constraint(q, xy):
#         x = (self.L[0] * np.cos(q[0]) + self.L[1] * np.cos(q[0] + q[1]) +
#              self.L[2] * np.cos(np.sum(q))) - xy[0]
#         return x
#
#     def y_constraint(q, xy):
#         y = (self.L[0] * np.sin(q[0]) + self.L[1] * np.sin(q[0] + q[1]) +
#              self.L[2] * np.sin(np.sum(q))) - xy[1]
#         return y
#
#     return scipy.optimize.fmin_slsqp(func=distance_to_default,
#                                      x0=self.q, eqcons=[x_constraint, y_constraint],
#                                      args=[xy], iprint=0)  # iprint=0 suppresses output

