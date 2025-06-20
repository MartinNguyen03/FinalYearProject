# Final Year Project - VLM integrated Task Planning for DLO Manipulation in Robotics
Source Code 

## Project Structure

```
├── catkin_ws
│   └── src
│   │   ├── Groot
│   │   └── ROSLLM
│   │       ├── agent_comm
│   │       │   ├── launch
│   │       │   ├── prompt
│   │       │   │   └── intro_and_conditions.txt
│   │       │   ├── scripts
│   │       │   │   ├── extern
│   │       │   │   │   └── DeepSeekVL
│   │       │   │   ├── vlm_node.py
│   │       │   │   └── vlm.py
│   │       │   ├── setup.py
│   │       │   └── src
│   │       ├── behavior_executor
│   │       │   ├── config
│   │       │   │   ├── bt_node_description.xml
│   │       │   │   └── gen_tree.xml
│   │       │   ├── extern
│   │       │   │   └── BehaviorTree.ROS
│   │       │   └── src
│   │       │       └── yumi_tree.cpp
│   │       ├── realsense-ros
│   │       ├── rosllm_srvs
│   │       ├── yumi_ctrl
│   │       │   ├── launch
│   │       │   │   ├── ctrl_node.launch
│   │       │   ├── params
│   │       │   ├── scripts
│   │       │   │   ├── ctrl_node.py
│   │       │   │   ├── scene_ctrl.py
│   │       │   │   ├── scene_logger.py
│   │       │   │   ├── scene_params.py
│   │       │   │   ├── utils.py
│   │       │   │   ├── yumi_ctrl.py
│   │       │   │   └── yumi_wrapper.py
│   │       │   └── setup.py
│   │       ├── yumi-prl
│   │       └── yumi_vsn
│   │           ├── launch
│   │           │   └── yumi_vsn.launch
│   │           ├── package.xml
│   │           ├── scripts
│   │           │   ├── colour_tuner.py
│   │           │   ├── dlo_vsn_coloured.py
│   │           │   ├── dlo_vsn_node.py
│   │           │   ├── platform_registration.py
│   │           │   └── utils
│   │           │       ├── color_tuner.py
│   │           │       ├── colour_segmentation.py
│   │           │       ├── images.py
│   │           │       ├── point_clouds.py
│   │           │       └── yumi_camera.py
│   │           └── setup.py
│   └── yumi.urdf
├── Dockerfile
├── LICENSE
├── Makefile
└── README.md
```

## Prerequisites
### System requirements
* This project has been tested on Ubuntu 20.04 and 22.04.
* Docker and Nvidia-docker installed.

## Installation
1. Clone this repo
   ```
   git clone https://github.com/MartinNguyen03/FinalYearProject.git
   ```
2. Download the Docker image from Docker Hub
   ```
   cd FINALYEARPROJECT
   make install-from-hub
   ```

### Installing baseline system
```
make install_baseline
```

## Running the project

### DLO Manipulation Live scene
The live scene is where the baseline system is tested.
1. Initialise roscore:
    ```
    make roscore
    ```
2. Start YuMi Driver:
    ```
    make yumi
    ```
3. Connect YuMi L515 Camera:
    ```
    make main-camera
    ```
4. Start VLM Agent:
    ```
    make vlm
    ```
5. Start Vision Module:
    ```
    make vsn
    ```
6. Start Control Module:
    ```
    make ctrl
    ```
The simulation can be stopped by pressing `Ctrl+C` in the terminal.

The live information is published in ROS topics.
The baseline system is tested with this scene. Following is what happens when the simulation starts.

## Individual Components

* `VLMAgent` Initialises the VLM as a ROS service for `Control` to communicate with via `VLM` Ros Service
* `YuMiVsn` Initialises the Vision Module Connecting to the L515 Camera initialisisng `DetectRope` and `ObserveScene` Ros Service
    - `YuMiCamera` Publish depth-images to ROS.
    - `ColourSegmentation` Handles all colour segmentation tasks
* `Main/Control` contains the main process of the system. It constructs the scene objects according to the parameters and executes primitves.
    - `YuMiTree` Initialised from `Control` subscribing to the `ExecuteBehavior` ROS Service to execute primitives.
    - `YuMi` and `YuMiWrapper` interface with ROS to control the robot.


### Randomisation
* Light intensity
* Camera depth noise
We model the camera depth noise as a Gaussian distribution.
The error rate is modelled after the [Intel RealSense L515](https://dev.intelrealsense.com/docs/lidar-camera-l515-datasheet)  (0.25% of depth).


## Acknowledgement
* Special thanks to [Haining Luo](https://github.com/HainingLuo) for his help on this project.
