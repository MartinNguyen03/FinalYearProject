ROS_IP := 127.0.0.1
DISPLAY:=$(shell echo $$DISPLAY)
.PHONY: .compile

install-from-hub:
	docker pull martinnguyen03/fyp:latest
	@$(MAKE) -s .compile

install-from-source:
	docker pull martinnguyen03/fyp:latest

	docker create --name temp_container martinnguyen03/fyp:latest

	docker cp temp_container:/catkin_ws ${PWD}/catkin_ws
	docker cp temp_container:/UniLace ${PWD}/UniLace


.compile:
	
	docker container stop fypContainer || true && docker container rm fypContainer || true
	docker run \
		-it \
		-e ROS_IP="${ROS_IP}" \
		-e ROS_MASTER_URI="http://${ROS_IP}:11311" \
		-e DISPLAY=${DISPLAY} \
    	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		-v /dev:/dev \
		-v ${PWD}/catkin_ws:/catkin_ws:rw \
		-v ${PWD}:/FINALYEARPROJECT:rw \
		--detach \
		--privileged \
		--runtime nvidia \
		--network host \
  		--gpus all \
		--name fypContainer \
		martinnguyen03/fyp:latest
	docker exec fypContainer bash -c "source /opt/ros/noetic/setup.bash && catkin build"
	docker container stop fypContainer
	
yumi_terminal:
	xhost +si:localuser:root >> /dev/null
	docker start fypContainer
	docker exec -e ROS_MASTER_URI="http://${ROS_IP}:10.0.1.111" -it fypContainer bash 

ssh:

yumi:
	xhost +si:localuser:root >> /dev/null
	docker start fypContainer
	docker exec -it fypContainer bash -c "source devel/setup.bash && roslaunch yumi_driver yumi_drivers.launch"

	

dlo:
	xhost +si:localuser:root >> /dev/null
	docker start fypContainer
	sleep 1
	docker exec -it fypContainer bash -c "source devel/setup.bash && roslaunch yumi_ctrl dlo.launch"
	docker container stop fypContainer

ctrl:
	xhost +si:localuser:root >> /dev/null
	docker start fypContainer
	sleep 1
	docker exec -it fypContainer bash -c "source devel/setup.bash && roslaunch yumi_ctrl ctrl_node.launch"
	docker container stop fypContainer
vlm:
	xhost +si:localuser:root >> /dev/null
	docker start fypContainer
	docker exec -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -it fypContainer bash -c "source devel/setup.bash && roslaunch agent_comm ros_agent.launch"
demo:
	xhost +si:localuser:root >> /dev/null
	docker start fypContainer
	sleep 1
	docker exec -it fypContainer bash -c "source devel/setup.bash && cd /UniLace/user_gym_policy && python uni_lace_demo.py"
	docker container stop fypContainer 

start-live:
	xhost +si:localuser:root >> /dev/null
	docker start fypContainer
	docker exec -it fypContainer bash -c "source devel/setup.bash && roslaunch uni_lace uni_lace_live.launch" 
	docker container stop fypContainer

start-gym:
	xhost +si:localuser:root >> /dev/null
	docker start fypContainer
	sleep 1
	docker exec -it fypContainer bash -c "source devel/setup.bash && cd /UniLace/user_gym_policy && python uni_lace_gym_simple_policy.py"
	docker container stop fypContainer

install-baseline:
	git -C ${PWD}/catkin_ws/src clone git@github.com:ImperialCollegeLondon/robot_sl.git
	git -C ${PWD}/catkin_ws/src clone git@github.com:ImperialCollegeLondon/yumi-moveit.git
	docker start fypContainer
	sleep 1
	docker exec fypContainer bash -c "source /opt/ros/noetic/setup.bash && catkin build"
	docker container stop fypContainer 

run-baseline:
	xhost +si:localuser:root >> /dev/null
	docker start fypContainer
	docker exec -it fypContainer bash -c "source devel/setup.bash && roslaunch sl_ctrl robot_sl.launch sim:=true"

observation_manager:
	xhost +si:localuser:root >> /dev/null
	docker start fypContainer
	docker exec -it fypContainer bash -c "source devel/setup.bash && roslaunch observation_manager test.launch"
	docker container stop fypContainer
vsn:
	xhost +si:localuser:root >> /dev/null
	docker start fypContainer
	docker exec -e DISPLAY=${DISPLAY} -it fypContainer bash -c "source devel/setup.bash && roslaunch yumi_vsn yumi_vsn.launch"
	
get_observation:
	xhost +si:localuser:root >> /dev/null
	docker start fypContainer
	docker exec -it fypContainer bash -c "source devel/setup.bash && rosservice call /get_observation"
	docker container stop fypContainer

debug:
	xhost +si:localuser:root >> /dev/null
	docker start fypContainer
	docker exec -it fypContainer bash -c "source devel/setup.bash && bash"

recompile:
	docker start fypContainer
	docker exec -it fypContainer bash -c "source /opt/ros/noetic/setup.bash && catkin build"
	docker container stop fypContainer

install_dependencies:
	docker start fypContainer
	docker exec -it fypContainer bash -c "source /opt/ros/noetic/setup.bash &&  rosdep install -i -r -y --from-paths . --ignore-src"
	docker container stop fypContainer

llmAgent:
	xhost +si:localuser:root >> /dev/null
	docker start fypContainer
	docker exec -it fypContainer bash -c "source devel/setup.bash && roslaunch agent_comm ros_agent.launch"
	docker container stop fypContainer
	
l_camera:
	xhost +si:localuser:root >> /dev/null
	docker start fypContainer
	docker exec -it fypContainer bash -c "source devel/setup.bash && roslaunch realsense2_camera rs_d435.launch camera:=yumi_d435_l serial_no:=936322072387 filters:=spatial,temporal,pointcloud"
	docker container stop fypContainer

l515:
	xhost +si:localuser:root >> /dev/null
	docker start fypContainer
	docker exec -e DISPLAY=${DISPLAY} -it fypContainer bash -c "source devel/setup.bash && source ~/.bashrc &&  roslaunch realsense2_camera rs_l515.launch camera:=yumi_l515 serial_no:=f0232155 filters:=spatial,temporal,pointcloud"
	docker container stop fypContainer



realsense:
	xhost +si:localuser:root >> /dev/null
	docker start fypContainer
	docker exec -it fypContainer bash -c "source devel/setup.bash && roslaunch realsense2_camera rs_aligned_depth.launch"
	docker container stop fypContainer
llmClient:
	xhost +si:localuser:root >> /dev/null
	docker start fypContainer
	docker exec -it fypContainer bash -c "source devel/setup.bash && rosrun agent_comm llm_client"
	docker container stop fypContainer
	
groot:
	xhost +si:localuser:root >> /dev/null
	docker start fypContainer
	docker exec -e DISPLAY=${DISPLAY} -it fypContainer bash -c "source devel/setup.bash && rosrun groot Groot"
	
checkdisp:
	xhost +si:localuser:root >> /dev/null
	docker start fypContainer
	docker exec -it fypContainer bash -c "echo \$DISPLAY"
	docker container stop fypContainer
# => returns empty

# => returns empty

	docker container stop fypContainer
terminal:
	xhost +si:localuser:root >> /dev/null
	docker start fypContainer
	docker exec -it fypContainer bash

debug_dependencies:
	docker start fypContainer
	docker exec -it fypContainer bash -c "source /opt/ros/noetic/setup.bash && rosdep check --from-paths . --ignore-src"

roscore:
	docker start fypContainer
	docker exec -it fypContainer bash -c "source /opt/ros/noetic/setup.bash && roscore"

stop:
	docker container stop fypContainer 

push:
	docker commit fypContainer martinnguyen03/fyp:latest
	docker tag martinnguyen03/fyp martinnguyen03/fyp
	docker push martinnguyen03/fyp

.display:
	docker fypContainer bash -c "export DISPLAY=$(DISPLAY)"