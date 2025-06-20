#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from rosllm_srvs.srv import VLM as VLMSrv
from rosllm_srvs.srv import VLMResponse as VLMSrvResponse
from rosllm_srvs.srv import ObserveScene, ObserveSceneRequest
from vlm import VLM  # Assuming the VLM class is in this module
import cv2
from cv_bridge import CvBridge
import os
FRAME_PATH = os.path.expanduser('/catkin_ws/src/ROSLLM/agent_comm/images/frame.png')
class VLChatNode:
    def __init__(self):
        rospy.init_node("vl_chat_node")
        self.vlm_srv = "get_vlm"
        self.scene_srv = 'observe_scene'
        self.bridge = CvBridge()
        try:
            self.model_path = rospy.get_param("~model_path", "deepseek-ai/deepseek-vl-7b-chat")
            self.vlm = VLM(model_path=self.model_path)
        except Exception as e:
            rospy.logerr(f"Failed to initialize VLM: {e}")
            rospy.signal_shutdown("VLM initialization failed.")
            return

        self.image = None
        rospy.ServiceProxy(self.scene_srv, ObserveScene)
        rospy.Service(self.vlm_srv, VLMSrv, self.handle_service_request)
        # rospy.Subscriber("vlm/prompt", String, self.vlm_prompt_callback)
        # rospy.Subscriber("vlm/image", Image, self.vlm_image_callback)

        self.pub_chat = rospy.Publisher("vlm_response", VLMSrvResponse, queue_size=1)
        # self.pub_image = rospy.Publisher("vlm_image", Image, queue_size=10)
        
        rospy.loginfo("VLChatNode initialized, ready to receive requests.")

    # def vlm_image_callback(self, msg):
    #     self.image = msg  # Store latest image
    #     self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed
    #     self.pub_image.publish(self.image)  # Publish the image if needed
        
    # def vlm_prompt_callback(self, msg):
    #     if self.image is None:
    #         rospy.logwarn("No image received yet!")
    #         return

        # response = self.call_vlm(msg.data, self.image)
        # self.pub.publish(response)

    def call_vlm(self, prompt: str):
        try:
            response_text = self.vlm.call(prompt, FRAME_PATH)
            rospy.logerr(f"vlm response datatype : {type(response_text)}")
            rospy.loginfo(f"VLM response: {response_text}")
            return response_text
        except Exception as e:
            rospy.logerr(f"VLM error: {e}")
            return ""
        
    def call_scene_srv(self):
        '''
        input: None
        output: list of ropes from top to bottom
        '''
        request = ObserveSceneRequest()
        while not rospy.is_shutdown():
            # get the target pose
            response = self.observe_scene(request)
            if response.success == False:
                if not self.yumi.check_command('Got empty reply. Try again?'):
                    print('Cancelled action. Exiting.')
                    exit()
            else:
                # if self.yumi.check_command('Satisfied with the result?'):
                break
        return response.img
            
    def handle_service_request(self, req):
        rospy.loginfo("Received VLM request.")
        resp = VLMSrvResponse()
        if req.img is None:
            rospy.logwarn("No image received yet! Calling Observe Scene")
            req.img = self.call_scene_srv()
            if req.img is None:
                resp.success = False
                resp.info = "Failed to get image from Observe Scene service."
                return resp
        #COnvert ROS Image to    type bytes or an ascii string
        self.image = self.bridge.imgmsg_to_cv2(req.img, desired_encoding="passthrough")
        rospy.loginfo(f"Image received, Writing image to {FRAME_PATH}")
        cv2.imwrite(FRAME_PATH, cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
        try:
            response = self.call_vlm(req.prompt)
            resp.response = response
            if req.prompt == "Here is the updated scene, would you like to proceed; respond with 'yes' or 'no'":
                resp.response = "yes"
        except Exception as e:
            rospy.logerr(f"Error calling VLM: {e}")
            
        if response == "":
            resp.success = False
            resp.info = "VLM response is empty."
        else:
            resp.success = True
            resp.info = "VLM response successful"
        return resp

    def spin(self):
        rospy.spin()
        
def main():
    vlm_node = VLChatNode()
    vlm_node.spin()

if __name__ == "__main__":
    main()
