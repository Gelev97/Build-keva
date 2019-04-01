import rospy
import time
from geometry_msgs.msg import Pose, Twist
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState, ModelStates
from std_msgs.msg import Empty

model_states = {}


def model_states_callback(msg):
	global model_states
	models = msg.name
	poses = msg.pose
	twists = msg.twist
	for i in range(len(models)):
		model_states[models[i]] = (poses[i], twists[i])


def update_ball():
	pose = Pose()
	pose.position.x = 0
	pose.position.y = 1
	pose.position.z = 2
	pose.orientation.x = 0
	pose.orientation.y = 0
	pose.orientation.z = 0
	pose.orientation.w = 0

	state = ModelState()
	state.model_name = "ball"
	state.pose = pose

	rospy.wait_for_service("/gazebo/set_model_state")
	
	try:
		ball = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
		ret = ball(state)
	except rospy.ServiceException, e:
		print "Service call failed: %s"%e



def main():
	sub = rospy.Subscriber('/gazebo/model_states', ModelStates, model_states_callback)
	while True:
		time.sleep(5)
		update_ball()


if __name__ == "__main__":
	main()
