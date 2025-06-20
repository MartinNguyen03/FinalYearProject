#!/usr/bin/python3
import os
import sys
import json
from re import findall
import sys
import signal
import numpy as np
from time import time
import subprocess
import rospy
np.float = np.float64
from ros_numpy import msgify
from rosllm_srvs.srv import VLM, VLMRequest
from sensor_msgs.msg import Image
from std_msgs.msg import String, Int16, Int8, Int8MultiArray, Float32MultiArray
import cv2
from cv_bridge import CvBridge 
from scene_ctrl import ScenePrimitives

BT_XML_PATH =  os.path.expanduser('/catkin_ws/src/ROSLLM/behaviour_executor/config/gen_tree.xml')
BT_EXEC_PATH = os.path.expanduser( '/catkin_ws/src/ROSLLM/behaviour_executor/src/yumi_tree.cpp')
PROMPT_PATH = os.path.expanduser('/catkin_ws/src//ROSLLM/agent_comm/prompt/intro_and_conditions.txt')
DEMO1 = os.path.expanduser('/catkin_ws/src/ROSLLM/agent_comm/prompt/demo1.txt')
DEMO2 = os.path.expanduser('/catkin_ws/src/ROSLLM/agent_comm/prompt/demo2.txt')
class CtrlNode:
    bt_srv = 'get_behaviour'
    vlm_srv = 'get_vlm'
    
    image_topic = "/yumi_l515/camera/color/image_raw"  # Update with your RealSense topic
    def __init__(self):
        self.demo = "1"
        auto_execution = True
        reset = True
        self.debug = True
        self.bridge = CvBridge()
        self.latest_image = None
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback)
        self.action_pub = rospy.Publisher('/scene_ctrl/action', String, queue_size=10)
        self.progress_pub = rospy.Publisher('/scene_ctrl/progress', Int16, queue_size=10)
        self.get_vlm_srv = rospy.ServiceProxy(self.vlm_srv, VLM)
        
        self.action_pub.publish(String('Initialising YuMi ...'))
        rospy.loginfo("Setting Scene Primitives...")
        self.scene_ctrl = ScenePrimitives(auto_execution, reset)
        rospy.loginfo("Scene Primitives set.")
        self.action_pub.publish(String('YuMi initialised.'))
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def run(self):
        start_time = time()
        self.scene_ctrl.add_to_log('[Start time] '+ str(start_time))
        bt_construct_time = time()
        self.action_pub.publish(String("Waiting for images and BT service..."))
        response = self.handleVLMTree()
        vlm_time = time() - start_time
        rospy.loginfo("VLM response time: {}".format(str(vlm_time)))
        self.action_pub.publish(String("VLM response received."))
        self.action_pub.publish(String("Generating BT XML..."))
        
        self.vlm_to_bt(response)
        bt_construct_time = time() - bt_construct_time
        rospy.loginfo("BT XML generated in {}".format(str(bt_construct_time)))
        # if self.debug == False:
        self.launch_bt_exec()
        rospy.loginfo("Mission accomplished in {}".format(str(time()-start_time)))
        
        
        
        
    def handleVLMTree(self):
        rospy.loginfo("Waiting for VLM service...")
        self.action_pub.publish(String("Waiting for images and VLM service..."))

        rospy.wait_for_service(self.vlm_srv)
        rospy.loginfo("VLM service is available.")
        try:
            while not rospy.is_shutdown():
                if self.scene_ctrl.pm.img_frame is not None:
                    
                    self.latest_image = self.scene_ctrl.pm.img_frame
                    rospy.loginfo("Image received, proceeding with VLM request.")
                    rospy.loginfo(f"ENter prompt")                     
                    prompt = input("Enter your prompt: ")
                    
                    prompt = self.init_prompt(prompt) 
                    # img = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')
                    # comp_img = self.bridge.cv2_to_compressed_imgmsg(img)
                    req = VLMRequest(prompt=prompt, img=self.latest_image)
                    resp = self.get_vlm_srv(req)
                    if self.demo is not None:
                        if self.demo == "1":
                            resp.response = self.read_prompt(DEMO1)
                        elif self.demo == "2":
                            resp.response = self.read_prompt(DEMO2)
                    rospy.loginfo(f"VLM Response:\n{resp.response}")
                    return resp.response
                
                else:
                    rospy.logwarn("No image received yet, waiting...")
                    rospy.sleep(1)
                    continue

        except rospy.ServiceException as e:
            rospy.logwarn(f"VLM request failed: {e}")
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down.")

    def signal_handler(self, signal, frame):
        '''Handles KeyboardInterrupts to ensure smooth exit'''
        rospy.logwarn('User Keyboard interrupt')
        self.stop()
        sys.exit(0)

        
    def image_callback(msg):
        global latest_image
        latest_image = msg
        
    def init_prompt(self, task: str) -> str:
        """
        Initializes the prompt for the VLM.
        """
        scene_str = ""
        scene_desc = []
        intro_prompt = self.read_prompt(PROMPT_PATH)
        
        rope_dict = self.scene_ctrl.pm.rope_dict
        for detected_rope in rope_dict.values():
            if detected_rope.marker_dict['marker_a']['marker_at'] is not None and detected_rope.marker_dict['marker_b']['marker_at'] is not None:
                scene_desc.append([detected_rope.name, 
                                   detected_rope.marker_dict['marker_a']['marker_at'], 
                                   detected_rope.marker_dict['marker_a']['colour'],
                                   detected_rope.marker_dict['marker_b']['marker_at'],
                                   detected_rope.marker_dict['marker_b']['colour']])
            else:
                rospy.logwarn(f"Rope {detected_rope.name} not fully detected.")
                self.scene_ctrl.init_target_poses()
                
        scene_str += f"Here are all the ropes in the image and their current location:\n"
        for rope_name, marker_a_at, marker_a_col, marker_b_at, marker_b_col in scene_desc:
            scene_str += f"Rope {rope_name}: marker_a ({marker_a_col}) at {marker_a_at}, marker_b ({marker_b_col}) at {marker_b_at}\n"
            
        scene_str += f"This is the current heirarchy of the ropes from top to bottom:\n"
        for rope in self.scene_ctrl.pm.heirarchy:
            scene_str += f"{rope}\n"
        final_prompt = f"{intro_prompt}\n{scene_str}\n, Here is the task you must complete as follows:\n {task}"
        rospy.loginfo(f"Final prompt: {final_prompt}")
        return final_prompt
    
    def read_prompt(self, filename):
        with open(filename, 'r') as file:
            return file.read()
        
    # VLM TO BT FUNCTIONS
    
    def vlm_to_bt(self, vlm_response: str):
        bt_xml = self.parse_vlm_response_to_bt(vlm_response)
        self.save_bt_xml(bt_xml)
        
        
    def parse_vlm_response_to_bt(self, vlm_response: str) -> str:
        """
        Parses the VLM output (assumed to be BT in simplified format) into BT-XML string.
        """
        commands = vlm_response.strip().splitlines()
        parsed_nodes = []

        for command in commands:
            command = command.strip()
            if command.startswith("YumiAction"):
                args = dict(findall(r'(\w+)\s*=\s*"([^"]+)"', command))
                required_keys = ['action', 'rope', 'marker', 'site']
                
                if all(key in args for key in required_keys):
                    cmd_xml = f'''  <YumiAction service_name="execute_behaviour"
                                    action="{args['action']}"
                                    rope="{args['rope']}"
                                    marker="{args['marker']}"
                                    site="{args['site']}"
                                    message="{{task}}" />'''
                    parsed_nodes.append(cmd_xml)
                else:
                    rospy.logwarn(f"YumiAction command missing required keys: {command}")
                    continue
            elif command.startswith("VisualCheck"):
                    retry_block = """\
                <RetryUntilSuccessful num_attempts="4">
                    <Timeout msec="300">
                        <VisualCheck service_name="get_vlm" response="{vlm_response}" img="{img}" />
                    </Timeout>
                </RetryUntilSuccessful>"""
                    parsed_nodes.append(retry_block)
            else:
                rospy.logwarn(f"Unknown command in VLM response: {command}")
                continue
            
        body = "\n".join(parsed_nodes)
        xml = f'''R"(
                <root>
                    <BehaviorTree>
                        <Sequence>
                {body}
                        </Sequence>
                    </BehaviorTree>
                </root>
                )"'''
        return xml
    
    def save_bt_xml(self, xml_string: str, path: str = BT_XML_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(xml_string)
        rospy.loginfo(f"BT XML saved to {path}")
        
    def launch_bt_exec(self):
        """
        Launches the BT executor with the generated XML file.
        """
        # Ensure the BT XML file is saved before launching
        cmd = ['rosrun', 'behavior_executor', 'yumi_tree']
        self.bt = subprocess.Popen(cmd)
        self.bt.wait()  # Wait for the process to complete
        rospy.sleep(1)
        
    
        
    def stop(self):
        self.bt.terminate()
        self.scene_ctrl.stop()
            
            
if __name__ == "__main__":
    rospy.init_node('ctrl_node', anonymous=True)

    # Run the sequence of actions
    scene_ctrl_node = CtrlNode()
    scene_ctrl_node.run()
    rospy.spin()


# EXAMPLE BT XML 
# R"(
#  <root >
#      <BehaviorTree>
#         <Sequence>
#             <YumiAction service_name = "execute_behaviour"
#                         action="{action}"
#                         rope="{rope}"
#                         marker="{marker}"
#                         site="{target}"
#                         message="{task}"
#             <PrintValue message="{task}"/>

#             <RetryUntilSuccessful num_attempts="4">
#                 <Timeout msec="300">
#                     <VisualCheck service_name="get_vlm"
#                           response="{vlm_response} />
#                 </Timeout>
#             </RetryUntilSuccessful>
#             <PrintValue message="{fibonacci_result}"/>
#         </Sequence>
#      </BehaviorTree>
#  </root>
#  )";