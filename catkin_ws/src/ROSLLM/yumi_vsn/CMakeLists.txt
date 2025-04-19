cmake_minimum_required(VERSION 2.8.3)
project(yumi_ctrl)

find_package(catkin REQUIRED COMPONENTS
	std_msgs
	sensor_msgs
	geometry_msgs
	message_filters
	cv_bridge
	ros_numpy
	rospy
	tf
)

catkin_python_setup()
catkin_package()