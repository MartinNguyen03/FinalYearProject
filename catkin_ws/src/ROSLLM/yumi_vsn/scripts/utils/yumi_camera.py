import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import rospy
from ros_numpy import numpify
from sensor_msgs.msg import Image, CameraInfo, CompressedImage

class RealCamera:
    def __init__(self, camera_name, roi=None, depth_thresh=None, jetson=True):
        '''
        Initialise a RealCamera object to interface with a RealSense camera over ROS.

        Args:
            camera_name (str): Name of the camera topic prefix.
            roi (tuple[int, int, int, int], optional): Region of interest for cropping (x1, y1, x2, y2).
            depth_thresh (float, optional): Depth threshold for filtering (not yet applied in code).
            jetson (bool): Whether the camera is on a Jetson device (applies HSV correction).
        '''
        self.roi = roi
        self.depth_thresh = depth_thresh
        self.jetson = jetson
        self.image_topic = f'/{camera_name}/color/image_raw'
        self.depth_topic = f'/{camera_name}/aligned_depth_to_color/image_raw'
        self.image_topic_compressed = f'/{camera_name}/color/image_raw/compressed'
        self.info_topic = f'/{camera_name}/aligned_depth_to_color/camera_info'
        self.frame = f'/{camera_name}_color_optical_frame'
        self.camera_info = rospy.wait_for_message(self.info_topic, CameraInfo).K
        self.img_msg = None
        self.depth_msg = None
        rospy.Subscriber(self.image_topic_compressed, CompressedImage, self.img_sub, queue_size=1)
        rospy.loginfo(f"Subscribed to compressed image topic: {self.image_topic_compressed}")
        rospy.Subscriber(self.depth_topic, Image, self.depth_sub, queue_size=1)

    def img_sub(self, msg):
        '''
        Callback function to store compressed image messages.

        Args:
            msg (sensor_msgs.msg.CompressedImage): Compressed image message.
        '''
        self.img_msg = msg

    def depth_sub(self, msg):
        '''
        Callback function to store depth image messages.

        Args:
            msg (sensor_msgs.msg.Image): Depth image message.
        '''
        self.depth_msg = msg

    def read_image(self) -> np.ndarray:
        '''
        Retrieves the latest colour image (compressed) from the camera.

        Returns:
            np.ndarray: BGR image of shape (H, W, 3), dtype=uint8.

        Notes:
            - Blocks until image is received.
            - Applies optional Jetson-specific HSV correction.
            - Applies ROI cropping if set.
        '''
        while self.img_msg is None:
            rospy.sleep(0.01)
            rospy.logwarn("Waiting for image message...")
        img_msg = self.img_msg
        np_arr = np.frombuffer(img_msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if self.jetson:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(img)
            img = cv2.merge([cv2.add(h, 5), cv2.add(s, -10), cv2.add(v, 10)])
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        if self.roi:
            img = img[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
        self.img_msg = None
        return img

    def read_image_ori(self):
        '''
        Retrieves the original (uncompressed) image via `sensor_msgs/Image`.

        Returns:
            np.ndarray: BGR image of shape (H, W, 3), dtype=uint8.

        Notes:
            - Converts from RGB to BGR (OpenCV standard).
            - Applies optional Jetson-specific HSV correction.
            - Applies ROI cropping if set.
        '''
        img_msg = rospy.wait_for_message(self.image_topic, Image)
        img = numpify(img_msg)
        if self.jetson:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(img)
            img = cv2.merge([cv2.add(h, 5), cv2.add(s, -10), cv2.add(v, 10)])
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if self.roi:
            img = img[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
        return img

    def read_depth(self):
        '''
        Retrieves the latest depth image.

        Returns:
            np.ndarray: Depth image of shape (H, W), dtype=uint16 or float32 depending on encoding.

        Notes:
            - Blocks until a depth image is received.
            - No filtering or scaling applied.
        '''
        while self.depth_msg is None:
            rospy.sleep(0.01)
        depth_msg = self.depth_msg
        depth = numpify(depth_msg)
        self.depth_msg = None
        return depth

    def read_camera_info(self):
        '''
        Retrieves the camera intrinsic matrix.

        Returns:
            list[float]: 3x3 intrinsic matrix (K) as a flat list of 9 elements.
        '''
        return self.camera_info

if __name__ == "__main__":
    realsense_camera = RealCamera()
