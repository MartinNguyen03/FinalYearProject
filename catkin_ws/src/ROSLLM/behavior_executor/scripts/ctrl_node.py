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
from rosllm_srvs.srv import VLM, VLMRequest, ExecuteBehaviour, ExecuteBehaviourRequest
from sensor_msgs.msg import Image
from std_msgs.msg import String, Int16, Int8, Int8MultiArray, Float32MultiArray

from yumi_ctrl.scripts.scene_ctrl import ScenePrimitives

BT_XML_PATH = '/catkin_ws/src/ROSLLM/behaviour_executor/config/gen_tree.xml'
BT_EXEC_PATH = '/catkin_ws/src/ROSLLM/behaviour_executor/src/yumi_tree.cpp'

class CtrlNode:
    bt_srv = 'get_behaviour'
    vlm_srv = 'get_vlm'
    image_topic = "/camera/color/image_raw"  # Update with your RealSense topic
    def __init__(self):
        auto_exec = True
        reset = True
        self.latest_image = None
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback)
        self.action_pub = rospy.Publisher('/scene_ctrl/action', String, queue_size=10)
        self.progress_pub = rospy.Publisher('/scene_ctrl/progress', Int16, queue_size=10)
        self.get_behaviour_srv = rospy.ServiceProxy(self.bt_srv, ExecuteBehaviour)
        self.get_vlm_srv = rospy.ServiceProxy(self.vlm_srv, VLM)
        
        
        self.action_pub.publish(String('Initialising YuMi ...'))
        self.scene_ctrl = ScenePrimitives(auto_exec, reset)
        self.action_pub.publish(String('YuMi initialised.'))
        self.handleVLMTree()
        signal.signal(signal.SIGINT, self.signal_handler)
        

    def handleVLMTree(self):

        self.action_pub.publish(String("Waiting for images and VLM service..."))

        rospy.wait_for_service(self.vlm_srv)
        try:
            while not rospy.is_shutdown():
                if self.scene_ctrl.pm.img_frame is not None:
                    
                    self.latest_image = self.scene_ctrl.pm.img_frame                     
                    prompt = input("Enter your prompt: ")
                    prompt = self.init_prompt(prompt) 
                    req = VLMRequest(prompt=prompt, img=msgify(self.latest_image, encoding='rgb8'))
                    resp = self.get_vlm_srv(req)

                    rospy.loginfo(f"VLM Response:\n{resp.response.response}")
                
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

    def stop(self):
        self.scene_ctrl.stop()
        
    def image_callback(msg):
        global latest_image
        latest_image = msg

# EXAMPLE BT XML 
# R"(
#  <root >
#      <BehaviorTree>
#         <Sequence>
#             <AddTwoInts service_name = "add_two_ints"
#                         first_int = "3" second_int = "4"
#                         sum = "{add_two_result}" />
#             <PrintValue message="{add_two_result}"/>

#             <RetryUntilSuccessful num_attempts="4">
#                 <Timeout msec="300">
#                     <Fibonacci server_name="fibonacci" order="5"
#                                result="{fibonacci_result}" />
#                 </Timeout>
#             </RetryUntilSuccessful>
#             <PrintValue message="{fibonacci_result}"/>
#         </Sequence>
#      </BehaviorTree>
#  </root>
#  )";

    def parse_vlm_response_to_bt(self, vlm_response: str) -> str:
        """
        Parses the VLM output (assumed to be BT in simplified format) into BT-XML string.
        """
        # Basic mock-up parsing. Adjust based on your VLM format.
        xml = f"""R"(
            <root>
                <BehaviorTree>
                    <Sequence>
                        {vlm_response}
                    </Sequence>
                </BehaviorTree>
        )
            """
        return xml
    
    def launch_bt_exec(self):
        """
        Launches the BT executor with the generated XML file.
        """
        # Ensure the BT XML file is saved before launching
        subprocess.Popen([BT_EXEC_PATH, BT_XML_PATH])
        
    def save_bt_xml(self, xml_string: str, path: str = BT_XML_PATH):
        with open(path, 'w') as f:
            f.write(xml_string)
        rospy.loginfo(f"BT XML saved to {path}")
    
    def vlm_to_bt(self, vlm_response: str):
        bt_xml = self.parse_vlm_response_to_bt(vlm_response)
        self.save_bt_xml(bt_xml)
        
    def init_prompt(self, task: str) -> str:
        """
        Initializes the prompt for the VLM.
        """
        scene_str = ""
        scene_desc = []
        intro_prompt = self.read_prompt('intro_and_conditions.txt')
        gen_bt_prompt = self.read_prompt('gen_bt.txt')
        
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
            scene_desc_str += f"Rope {rope_name}: marker_a ({marker_a_col}) at {marker_a_at}, marker_b ({marker_b_col}) at {marker_b_at}\n"
            
        scene_str += f"This is the current heirarchy of the ropes from top to bottom:\n"
        for rope in self.scene_ctrl.pm.heirarchy:
            scene_str += f"{rope}\n"
        final_prompt = f"{intro_prompt}\n{scene_str}\n{gen_bt_prompt}\n, Here is the task you must complete as follows:\n {task}"
        return final_prompt
    
    def read_prompt(self, filename):
        prompt_dir = "/ROSLLM/agent_comm/prompt"
        with open(os.path.join(prompt_dir, filename), 'r') as file:
            return file.read()
        
    def run(self):
        start_time = time()
        self.scene_ctrl.add_to_log('[Start time] '+ str(start_time))
       
        
  
        rospy.loginfo("Mission accomplished in {}".format(str(time()-start_time)))
        
            
            
            
if __name__ == "__main__":
    rospy.init_node('ctrl_node', anonymous=True)

    # Run the sequence of actions
    scene_ctrl_node = CtrlNode()
    scene_ctrl_node.run()
