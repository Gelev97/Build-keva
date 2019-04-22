import rospy
import math
import time
import geometry_msgs
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler

PLANK_LENGTH=1.143
PLANK_WIDTH=0.1905
PLANK_HEIGHT=0.0635

plank_sdff = open('model_editor_models/keva_plank/model.sdf', 'r').read()

spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
plank_count = 0

def spawn_model(model, center, yaw):
	global plank_count
	initial_pose = Pose()
	initial_pose.position.x = center[0]
	initial_pose.position.y = center[1]
	initial_pose.position.z = center[2]
	initial_pose.orientation = geometry_msgs.msg.Quaternion(*quaternion_from_euler(0, 0, yaw))

	if model == 'ball':
		sdff = ball_sdff
	else:
		sdff = plank_sdff

	spawn_model_prox("keva_plank_" + str(plank_count), sdff, "", initial_pose, "world")
	plank_count += 1


def main():
	rospy.init_node('insert_object',log_level=rospy.INFO)
	rospy.wait_for_service('gazebo/spawn_sdf_model')
	time.sleep(10)
	spawn_model('plank', [0, 0, PLANK_HEIGHT/2+0.01], 82*math.pi/180)
	time.sleep(10)
	spawn_model('plank', [0, 0, PLANK_HEIGHT+PLANK_HEIGHT/2+0.01], 0*math.pi/180)
	time.sleep(10)
	spawn_model('plank', [0, 0, 2*PLANK_HEIGHT+PLANK_HEIGHT/2+0.01], (-44+360)*math.pi/180)
	time.sleep(10)
	spawn_model('plank', [0, 0, 3*PLANK_HEIGHT+PLANK_HEIGHT/2+0.01], 33*math.pi/180)
	time.sleep(10)


if __name__ == "__main__":
	main()
