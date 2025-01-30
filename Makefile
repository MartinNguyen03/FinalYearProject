ROS_IP := 127.0.0.1

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
	git -C ${PWD}/catkin_ws/src clone https://github.com/Unity-Technologies/ROS-TCP-Endpoint.git
	docker container stop fypContainer || true && docker container rm fypContainer || true
	docker run \
		-it \
		-e ROS_IP="${ROS_IP}" \
		-e ROS_MASTER_URI="http://${ROS_IP}:11311" \
		-e DISPLAY \
    	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		-v /dev:/dev \
		-v ${PWD}/catkin_ws:/catkin_ws:rw \
		-v ${PWD}:/UniLace:rw \
		--detach \
		--privileged \
		--runtime nvidia \
		--network host \
  		--gpus all \
		--name fypContainer \
		martinnguyen03/fyp:latest
	docker exec fypContainer bash -c "source /opt/ros/noetic/setup.bash && catkin build"
	docker container stop fypContainer

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

debug:
	xhost +si:localuser:root >> /dev/null
	docker start fypContainer
	docker exec -it fypContainer bash -c "source devel/setup.bash && bash"

recompile:
	docker start fypContainer
	docker exec -it fypContainer bash -c "source /opt/ros/noetic/setup.bash && catkin build"
	docker container stop fypContainer

stop:
	docker container stop fypContainer 

push:
	docker push martinnguyen03/fyp