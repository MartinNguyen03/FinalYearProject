#!/usr/bin/env python3.10.0
import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('publisher')
        self.cameraDeviceNumber = 0 # 0 is the default camera
        self.camera = cv2.VideoCapture(self.cameraDeviceNumber) # Open the camera
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # Set the width of the camera frame
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Set the height of the camera frame
        
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) # Set the codec of the camera frame
        # CVBridge is used to convert OpenCV images to ROS images that can be sent through ROS topics
        self.bridge = CvBridge()

        # The name of the topic is used to transfer camera images
        # this topic name should match the name of the topic in the subscriber node 
        self.topicNameFrame = 'camera_frame' 

        self.queueSize = 20 # The queue size for the topics/messages

        # Create a publisher object that will publish the camera images to the topic: self.topicNameFrame
        # The message type is Image and the queue size is self.queueSize
        self.publisher = self.create_publisher(Image, self.topicNameFrame, self.queueSize)

        # Communication Period
        self.timerPeriod = 0.1
        
        # Create a timer that will call the timer_callback function every 0.1 seconds
        self.timer = self.create_timer(self.timerPeriod, self.timer_callback)

        self.i = 0 # Counter for the number of messages sent


    def timer_callback(self):
        # Read the camera frame
        ret, frame = self.camera.read()

        if not ret or frame is None:
            self.get_logger().info('No camera frame')
            return
        
        # Resize the frame to 640x480
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)

        if ret :
            imageMessage = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8') # Convert the frame to an Image message

            # Publish the image message to the topic
            self.publisher.publish(imageMessage)

        self.get_logger().info('(YOLO)Publishing image number: %d' % self.i)

        self.i += 1

def main(args=None):
    rclpy.init(args=args)

    cameraPublisher = CameraPublisher()

    rclpy.spin(cameraPublisher)

    cameraPublisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()