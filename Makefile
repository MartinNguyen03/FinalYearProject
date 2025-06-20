ROS_IP:=10.0.1.111
DISPLAY := $(shell echo $${DISPLAY})
.PHONY: .compile

install-from-hub:
	docker pull martinnguyen03/fyp:latest
	@$(MAKE) -s .compile


.compile:
	
	docker container stop fypContainer || true && docker container rm fypContainer || true
	docker run \
		-it \
		-e ROS_IP="${ROS_IP}" \
		-e ROS_MASTER_URI="http://10.0.1.111:11311" \
		-e DISPLAY=${DISPLAY} \
    	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		-v /dev:/dev \
		-v ${PWD}/catkin_ws:/catkin_ws:rw \
		-v ${PWD}:/FinalYearProject:rw \
		--detach \
		--privileged \
		--runtime nvidia \
		--network host \
  		--gpus all \
		--name fypContainer \
		martinnguyen03/fyp:latest
	docker exec fypContainer bash -c "source /opt/ros/noetic/setup.bash && catkin build"
	docker container stop fypContainer
	
install_baseline:
	git -c ${PWD}/catkin_ws/src/ROSLLM/yumi_vsn/scripts/utils clone https://github.com/lar-unibo/dlo_perceiver.git
	git -c ${PWD}/catkin_ws/src/ clone https://github.com/BehaviorTree/Groot.git
	docker start fypContainer
	sleep 1
	docker exec -it fypContainer bash -c "source /opt/ros/noetic/setup.bash &&  rosdep install -i -r -y --from-paths . --ignore-src"
	sleep 1
	docker exec -it fypContainer bash -c "source /opt/ros/noetic/setup.bash && catkin build"
	docker container stop fypContainer
	
roscore:
	docker start fypContainer
	docker exec -it fypContainer bash -c "source /opt/ros/noetic/setup.bash && roscore"

description:
	docker start fypContainer
	docker exec -it fypContainer bash -c "source devel/setup.bash && roslaunch yumi_description rviz.launch"

tuner:
	xhost +si:localuser:root >> /dev/null
	docker start fypContainer
	docker exec -e DISPLAY=${DISPLAY} -e ROS_IP="10.0.1.111" -e ROS_MASTER_URI="http://10.0.1.111:11311" -it fypContainer bash -c "source devel/setup.bash && rosrun yumi_vsn colour_tuner.py"

yumi_terminal:
	xhost +si:localuser:root >> /dev/null
	docker start fypContainer
	docker exec -e DISPLAY=${DISPLAY} -e ROS_IP="10.0.1.111" -e ROS_MASTER_URI="http://10.0.1.111:11311" -it fypContainer bash 


main-camera:
	ssh -t yumi-NUC "source catkin_ws/devel/setup.bash && roslaunch yumi_realsense yumi_l515.launch"

yumi:
	ssh -t prl-orin "source catkin_ws/devel/setup.bash && roslaunch yumi_driver yumi_driver.launch"

get-ee-left:
	rosrun tf tf_echo yumi_base_link yumi_link_7_l

get-ee-right:
	rosrun tf tf_echo yumi_base_link yumi_link_7_r

open-ee-right:
	rostopic pub /yumi/gripper_r_position_cmd std_msgs/Float64 "data: 10.0"
open-ee-left:
	rostopic pub /yumi/gripper_l_position_cmd std_msgs/Float64 "data: 10.0"
close-ee-right:
	rostopic pub /yumi/gripper_r_position_cmd std_msgs/Float64 "data: 0.0"
close-ee-left:
	rostopic pub /yumi/gripper_l_position_cmd std_msgs/Float64 "data: 0.0"

moveit:
	docker start fypContainer
	docker exec -it fypContainer bash -c "source devel/setup.bash && roslaunch yumi_moveit_config moveit_planning_execution.launch"

dlo:
	xhost +si:localuser:root >> /dev/null
	docker start fypContainer
	sleep 1
	docker exec -it fypContainer bash -c "source devel/setup.bash && roslaunch yumi_ctrl dlo.launch"
	docker container stop fypContainer
test1:
	docker start fypContainer
	docker exec -it fypContainer bash -c "source devel/setup.bash && rosservice call /execute_behaviour "{action: 'right_place', rope: 'rope_o', marker: 'marker_b', site: 'site_uu'}""



ctrl:
	docker start fypContainer
	sleep 1
	docker exec -it fypContainer bash -c "source devel/setup.bash && roslaunch yumi_ctrl ctrl_node.launch"
	docker container stop fypContainer
vlm:
	docker start fypContainer
	docker exec -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e PYTHONPATH=$PYTHONPATH:/home/yumi/FinalYearProject/catkin_ws/src/ROSLLM/agent_comm/scripts -it fypContainer bash -c "source devel/setup.bash && roslaunch agent_comm ros_agent.launch"
demo:
	xhost +si:localuser:root >> /dev/null
	docker start fypContainer
	sleep 1
	docker exec -it fypContainer bash -c "source devel/setup.bash && cd /UniLace/user_gym_policy && python uni_lace_demo.py"
	docker container stop fypContainer 

vsn:
	docker start fypContainer
	docker exec -it fypContainer bash -c "source devel/setup.bash && roslaunch yumi_vsn yumi_vsn.launch"
	

recompile:
	docker start fypContainer
	docker exec -it fypContainer bash -c "source /opt/ros/noetic/setup.bash && catkin build"
	docker container stop fypContainer

install_dependencies:
	docker start fypContainer
	docker exec -it fypContainer bash -c "source /opt/ros/noetic/setup.bash &&  rosdep install -i -r -y --from-paths . --ignore-src"
	docker container stop fypContainer

	
groot:
	xhost +si:localuser:root >> /dev/null
	docker start fypContainer
	docker exec -e DISPLAY=${DISPLAY} -it fypContainer bash -c "source devel/setup.bash && rosrun groot Groot"
	docker container stop fypContainer

terminal:
	docker start fypContainer
	docker exec -it fypContainer bash

debug_dependencies:
	docker start fypContainer
	docker exec -it fypContainer bash -c "source /opt/ros/noetic/setup.bash && rosdep check --from-paths . --ignore-src"


stop:
	docker container stop fypContainer 

push:
	docker commit fypContainer martinnguyen03/fyp:latest
	docker tag martinnguyen03/fyp martinnguyen03/fyp
	docker push martinnguyen03/fyp

.display:
	docker fypContainer bash -c "export DISPLAY=$(DISPLAY)"