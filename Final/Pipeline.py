import RPC
import subprocess
import math

HOME_POSITION = [20,-15,20]
LOWER_HEIGHT_GRAB = 12
LOWER_HEIGHT_PUT = 18
HEIGHT_ADJUST = 20
HIGHER_HEIGHT_PUT = 30

# Go back to Home
def back_home(current_position, s):
    return RPC.pipline_position_encoder(current_position, HOME_POSITION, s)

# Start from Home
def start_home(goal_position, s):
    return RPC.pipline_position_encoder(HOME_POSITION, goal_position, s)

# Go GRAB the block
def grab(current_position, s, roll):
    trajectory_grab = RPC.pipline_position_encoder_roll(current_position, [current_position[0],current_position[1],LOWER_HEIGHT_GRAB], roll, s)
    open_gripper = [-1,-1,-1,-1,-1,4000]
    close_gripper = [-1,-1,-1,-1,-1,7000]
    return [open_gripper] + trajectory_grab + [close_gripper]
    
def up(grab_position, s):
    open_gripper = [-1,-1,-1,-1,-1,4000]
    trajectory_up= RPC.pipline_position_encoder([grab_position[0],grab_position[1],LOWER_HEIGHT_GRAB], grab_position, s)
    return [open_gripper] + trajectory_up

# Go PUT the block
def put(current_position, s, roll):
    trajectory_put = RPC.pipline_position_encoder_roll(current_position, [current_position[0],current_position[1],LOWER_HEIGHT_PUT], roll, s)
    open_gripper = [-1,-1,-1,-1,-1,4000]
    return trajectory_put + [open_gripper]

def adjust(grab_trajectory, adj, angle):
    last = grab_trajectory[-2][0]
    print("last_encoder")
    print(last)
    last = ((last - 15)/4.16 - 90)/180*math.pi
    print("last")
    print(last)
    roll = angle +adj-float(last)
    print("roll")
    print(roll)
    servo_3 = 4000 + int(((roll+math.pi/2) / math.pi * 180) * 40)
    if (servo_3 < 4000):
        servo_3 = 4000
    elif (servo_3 > 8000):
        servo_3 = 8000
    return [[-1,-1,-1,-1,servo_3,-1]]    

# Classical Combination
# Home - Move above - Grab - Home - Move above - Put - Home
def classical_combi(grab_position, put_position, roll_grab, roll_put, s):
    grab_position = [grab_position[0], grab_position[1], grab_position[2]+HEIGHT_ADJUST]
    put_position = [put_position[0], put_position[1], HIGHER_HEIGHT_PUT]

    home_1 = start_home([grab_position[0],grab_position[1],grab_position[2]], s)
    grab_trajectory = grab(grab_position, s, roll_grab)
    grab_back = RPC.pipline_position_encoder([grab_position[0],grab_position[1],LOWER_HEIGHT_GRAB], grab_position, s)
    grab_put = RPC.pipline_position_encoder(grab_position, put_position, s)
    put_trajectory = put(put_position, s, roll_put)
    put_back = RPC.pipline_position_encoder([put_position[0],put_position[1],LOWER_HEIGHT_PUT],put_position, s)
    home_2 = back_home(put_position, s)

    result = (home_1, grab_trajectory, grab_back, grab_put, put_trajectory, put_back, home_2)
    return result

def Adjust(the_block, adj, s):
    grab_position = [the_block[0], the_block[1], the_block[2]+HEIGHT_ADJUST]
    
    home_1 = RPC.pipline_position_encoder([40,10,30], grab_position, s)
    grab_trajectory = grab(grab_position, s, the_block[3])
    adjust_traj = adjust(grab_trajectory, adj, the_block[3])
    grab_position = [the_block[0], the_block[1], 30]
    up_traj = up(grab_position, s)
    home_2 = RPC.pipline_position_encoder(grab_position,[40,10,30], s)

    result = (home_1, grab_trajectory, adjust_traj, up_traj, home_2)
    return result

# Make Command in C
def C_execute(commands):
    for command in commands:
        final = []
        for element_group in command:
            for element in element_group:
                final.append(str(element))
        print(final)
        subprocess.call(["sudo","./xarm2"] + final)

