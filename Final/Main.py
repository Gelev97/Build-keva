from imutils.video import VideoStream
import cv2
import imutils
import time
import socket

import recordXY
import detect_block
import findXY
import Pipeline
import RPC
import ball_reasoning

block_put_position = []

def Caliberate_camera(vs):
    recordXY.caliberate(vs)
    
def detect_block_grab(vs):
    while(1):
        frame = vs.read()
        frame = imutils.resize(frame, width=1600)
        frame = frame[0:frame.shape[0]-300,300:1200]
        cv2.imshow("frame", frame)
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if k%256 == 27:
            # ESC pressed
            print("picture chosen")
            break
    
    [img, threshold, blur, edges] = detect_block.find_edge(frame)
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
    for key in block_put_position:
        grab_position = block_real_position[key]
        put_position = block_put_position[put_index]
        commands = Pipeline.classical_combi(grab_position, put_position, s)
        Pipeline.C_execute(commands)
        put_index += 1

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
    TCP_IP = '128.237.215.125'
    TCP_PORT = 2002
    print('Socket Information: %s:%d' % (TCP_IP, TCP_PORT))
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))
    time.sleep(1e-3)

    # start camera
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    '''
    # calibration and find block
    Caliberate_camera(vs)
    block_pixel_position = detect_block_grab(vs)
    block_real_position = transfer_to_real(block_pixel_position)
    print(block_pixel_position)
    print(block_real_position)
    
    # Inverse Kinematics
    inverse_kinematics(block_real_position,s)
    '''

    traj = RPC.pipline_position_encoder([25.5, 17.3, 10, 0], [25.5, 0, 10, 0], s)
    print(traj)
    #commands = Pipeline.classical_combi([25.5, 17.3, 1], [25.5, 0, 1], s)
    #print(commands)
    #Pipeline.C_execute(commands)

    
    #ball_position = detect_ball(vs)
    #print(ball_position)


    s.close()
    vs.stop()
    print("Socket close.")
    print("Camera close.")

main()
    
    
    