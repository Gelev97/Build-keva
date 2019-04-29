import RPC
import subprocess


HOME_POSITION = [45,20,10]
LOWER_HEIGHT_GRAB = 3
LOWER_HEIGHT_PUT = 5
HEIGHT_ADJUST = 9

# Go back to Home
def back_home(current_position, s):
    return RPC.pipline_position_encoder(current_position, HOME_POSITION, s)

# Start from Home
def start_home(goal_position, s):
    return RPC.pipline_position_encoder(HOME_POSITION, goal_position, s)

# Go GRAB the block
def grab(current_position, s):
    trajectory_grab = RPC.pipline_position_encoder_roll(current_position, [current_position[0],current_position[1],LOWER_HEIGHT_GRAB, current_position[3]], s)
    open_gripper = [-1,-1,-1,-1,-1,4000]
    close_gripper = [-1,-1,-1,-1,-1,6000]
    return [open_gripper] + trajectory_grab + [close_gripper]

# Go PUT the block
def put(current_position, s):
    trajectory_put = RPC.pipline_position_encoder_roll(current_position, [current_position[0],current_position[1],LOWER_HEIGHT_PUT,current_position[3]], s)
    open_gripper = [-1,-1,-1,-1,-1,4000]
    return trajectory_put + [open_gripper]

# Classical Combination
# Home - Move above - Grab - Home - Move above - Put - Home
def classical_combi(grab_position, put_position, s):
    grab_position = [grab_position[0], grab_position[1], grab_position[2]+HEIGHT_ADJUST, grab_position[3]]
    put_position = [put_position[0], put_position[1], put_position[2]+HEIGHT_ADJUST, put_position[3]]

    home_1 = start_home([grab_position[0],grab_position[1],grab_position[2]], s)
    grab_trajectory = grab(grab_position, s)
    grab_back = RPC.pipline_position_encoder([grab_position[0],grab_position[1],LOWER_HEIGHT_GRAB], grab_position, s)
    grab_put = RPC.pipline_position_encoder(grab_position, put_position, s)
    put_trajectory = put(put_position, s)
    home_2 = back_home([put_position[0], put_position[1], LOWER_HEIGHT_PUT], s)

    result = (home_1, grab_trajectory, grab_back, grab_put, put_trajectory, home_2)
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

