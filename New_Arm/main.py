from IK import *
import math

GRAB_POSITION = [30,-20]
GRAB_ROLL = 0.4
def main():
    TCP_IP = '192.168.0.127'
    TCP_PORT = 2002
    print('Socket Information: %s:%d' % (TCP_IP, TCP_PORT))
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))

    robot = Robot('locobot')
    robot.arm.go_home()

    grab(robot, s, GRAB_POSITION, GRAB_ROLL)
    put(robot, s, [30,-5,35], 22.5, math.pi/2)

    grab(robot, s, GRAB_POSITION, GRAB_ROLL)
    put(robot, s, [35,-5,35], 22.5, math.pi/2)

    grab(robot, s, GRAB_POSITION, GRAB_ROLL)
    put(robot, s, [32.5,0,35], 23.5, 0)

    grab(robot, s, GRAB_POSITION, GRAB_ROLL)
    put(robot, s, [32.5,-8,35], 23.5, 0)

    grab(robot, s, GRAB_POSITION, GRAB_ROLL)
    put(robot, s, [30,-5,35], 24.5, math.pi/2)

    grab(robot, s, GRAB_POSITION, GRAB_ROLL)
    put(robot, s, [35,5,35], 24, math.pi/2)

    grab(robot, s, GRAB_POSITION, GRAB_ROLL)
    put(robot, s, [32.5,0,35], 25.5, 0)

    grab(robot, s, GRAB_POSITION, GRAB_ROLL)
    put(robot, s, [32.5,8,35], 25.5, 0)

    grab(robot, s, GRAB_POSITION, GRAB_ROLL)
    put(robot, s, [30,-5,35], 26.5, math.pi/2)

    grab(robot, s, GRAB_POSITION, GRAB_ROLL)
    put(robot, s, [35,-5,35], 26.5, math.pi/2)

main()