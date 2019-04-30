from imutils.video import VideoStream
import cv2
import imutils
import time
import socket
import math

import recordXY
import detect_block
import findXY
import Pipeline
import RPC
import ball_reasoning

#block_put_position = [[6.8, -13.2, 1, -0.95], [34.4, -14.7, 1, 0], [44.5, -14.44, 1, 0], [44.5, -14.5, 2.5, 0], [28.6, -14.8, 2.5, -1.57], [28.7, -11.3, 2, -1.57], [40.0, -11.2, 4, -1.57], [40.1, -21.0, 4, -1.57]]
block_put_position = [[6.8, -13.2, 1, -0.95], [34.4, -14.7, 1, 0]]

def Caliberate_camera(vs):
    recordXY.caliberate(vs)
    
def detect_block_grab(vs):
    while(1):
        frame = vs.read()
        frame = imutils.resize(frame, width=1600)
        frame = frame[0:frame.shape[0]-300,330:1200]
        cv2.imshow("frame", frame)
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if k%256 == 27:
            # ESC pressed
            print("picture chosen")
            break
    
    [img, threshold, blur, edges] = detect_block.find_edge(frame)
    cv2.imwrite("original.jpg", img)
    #detect_block.show_edge(img, threshold, blur, edges)
    block = detect_block.detect(edges, img)
    detect_block.draw_result(img, block)
    return detect_block.output_coordinates(block)

def transfer_to_real(block_pixel_position):
    result = dict()
    for key in range(len(block_pixel_position)):
        result[key] = findXY.find_dict(block_pixel_position[key])
    return result

def inverse_kinematics(block_real_position, s):
    put_index = 0
    for key in block_real_position:
        grab_position = block_real_position[key]
        if(grab_position[2] > 0):
            grab_roll = grab_position[2]  - 180
        else:
            grab_roll = grab_position[2] 
        grab_roll = grab_roll/180*math.pi
        grab_position = [grab_position[0][0]-4.5,grab_position[0][1]-0.5,1]
        put_position = block_put_position[put_index]
        put_roll = put_position[3]
        put_position = put_position[0:3]
        commands = Pipeline.classical_combi(grab_position, put_position, grab_roll, put_roll, s)
        Pipeline.C_execute(commands)
        put_index += 1
    HOME_POSITION = [20,-15,20]
    END_POSITION = [40,20,20]
    traj = RPC.pipline_position_encoder(HOME_POSITION,END_POSITION,s)
    Pipeline.C_execute([traj])
        

def detect_ball(vs):
    ball_image_poses = ball_reasoning.ballReason(vs)
    result = []
    for i in range(len(ball_image_poses)):
        if(ball_image_poses[i] == None):
            result.append(None)
        else:
            image_pos = list(ball_image_poses[i])
            result.append(findXY.find_list(image_pos))
    return result
        

def main():
    # start socket
    TCP_IP = '128.237.133.148'
    TCP_PORT = 2000
    print('Socket Information: %s:%d' % (TCP_IP, TCP_PORT))
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))
    time.sleep(1e-3)
    '''
    # start camera
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # calibration and find block
    #Caliberate_camera(vs)
    block_pixel_position = detect_block_grab(vs)
    block_real_position = transfer_to_real(block_pixel_position)
    print(block_pixel_position)
    print(block_real_position)
    '''
    # Inverse Kinematics
    # inverse_kinematics(block_real_position,s)
    HOME_POSITION = [20,-15,20]
    roll = -math.pi/2
    traj = RPC.pipline_position_encoder_roll(HOME_POSITION, [20/1.15,-20/0.975,20], roll, s)
    traj = RPC.pipline_position_encoder_roll([20/1.15,-20/0.975,20], [20/1.15,-20/0.975,12], roll, s)
    Pipeline.C_execute([traj])
    #ball_position = detect_ball(vs)
    #print(ball_position)


    s.close()
    #vs.stop()
    print("Socket close.")
    print("Camera close.")

main()
    
    