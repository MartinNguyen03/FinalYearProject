cmake_minimum_required(VERSION 3.0.2)
project(rosllm_srvs)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  sensor_msgs
  message_generation
  message_runtime
)

add_service_files(
  FILES
  ExecuteBehavior.srv
  DetectRope.srv
  ObserveScene.srv
  VLM.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})
