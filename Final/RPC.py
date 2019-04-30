
import numpy as np
from scipy import interpolate
import math

# Spline trajectory
def create_trajecotry(current_position, goal_position):
    x = np.array([current_position[0], goal_position[0]])
    y = np.array([current_position[1], goal_position[1]])
    z = np.array([current_position[2], goal_position[2]])
    current_position = [x, y, z]

    result = interpolate.splprep(current_position, s=0, k=1)
    x_i, y_i, z_i = interpolate.splev(np.linspace(0, 1, 5), result[0])
    trajectory = np.zeros((len(x_i), 3))
    for index in range(0, len(x_i)):
        trajectory[index] = [x_i[index], y_i[index], z_i[index]]
    print(trajectory)
    return trajectory


# Encoder transform
def encoder_transform(angle_group):
    result = []
    servo_1 = 0
    servo_3 = 0
    for angle in angle_group:
        tmp = []
        tmp.append(int((angle[0] / math.pi * 180 + 90) * 4.16 + 15))
        tmp.append(860 - int((angle[1] / math.pi * 180) * 4))
        tmp.append(int((angle[2] / math.pi * 180) * 4) + 162)

        if(angle[3] != -1):
            servo_1 = 6000 - int((angle[3] / math.pi * 180) * 40)
            if(servo_1 < 4000):
                tmp.append(4000)
            else:
                tmp.append(servo_1)
        else:
            tmp.append(-1)

        print("servo4" + str(angle[4]))
        if (angle[4] != -1):
            servo_3 = 4000 + int(((angle[4]+math.pi/2) / math.pi * 180) * 40)
            if (servo_3 < 4000):
                tmp.append(4000)
            elif (servo_3 > 8000):
                tmp.append(8000)
            else:
                tmp.append(servo_3)
        else:
            tmp.append(-1)

        tmp.append(-1)

        result.append(tmp)
    return result


# Pipeline
def pipline_position_encoder(start_position, end_position, s):
    # create trajectory
    trajecotry = create_trajecotry(start_position, end_position)

    # Calculate Inverse Kinematics
    angle_group = []
    previous_theta = [0,0,0,0,0]
    for waypoint in trajecotry:
        tx_data = '%f,%f,%f,%d,%d,%d' % (waypoint[0], waypoint[1], waypoint[2], previous_theta[0], \
                                         previous_theta[1], previous_theta[2])
        s.send(tx_data.encode())

        # Recieve from socket (wait here)
        rx_data = s.recv(1024).decode()
        if rx_data == '':  # terminate
            print("Connection dropped.")
            break

        print(rx_data)  # print-out
        string_theta = rx_data.split(",")
        theta = [float(string_theta[0]),float(string_theta[1]),float(string_theta[2])]
        previous_theta = theta
        theta.append(-1)
        theta.append(-1)
        theta.append(-1)
        print(theta)
        angle_group.append(theta)


    # Change to encoder value
    return encoder_transform(angle_group)

def pipline_position_encoder_roll(start_position, end_position, roll, s):
    # create trajectory
    trajecotry = create_trajecotry(start_position, end_position)

    # Calculate Inverse Kinematics
    angle_group = []
    previous_theta = [0,0,0]
    for waypoint in trajecotry:
        tx_data = '%f,%f,%f,%d,%d,%d' % (waypoint[0], waypoint[1], waypoint[2], previous_theta[0], \
                                                  previous_theta[1], previous_theta[2])
        s.send(tx_data.encode())

        # Recieve from socket (wait here)
        rx_data = s.recv(1024).decode()
        if rx_data == '':  # terminate
            print("Connection dropped.")
            break

        print(rx_data)  # print-out
        string_theta = rx_data.split(",")
        theta = [float(string_theta[0]),float(string_theta[1]),float(string_theta[2])]
        previous_theta = theta
        theta.append(float(string_theta[1]) + math.pi/2 - float(string_theta[2]))
        theta.append(roll-float(string_theta[0]))
        angle_group.append(theta)

    # Change to encoder value
    return encoder_transform(angle_group)
