#!/usr/bin/env python3.10.0
import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np

class ImSub(Node):
    def __init__(self):
        super().__init__('subscriber')
        self.bridge = CvBridge()

       
        # The name of the topic is used to transfer camera images
        # this topic name should match the name of the topic in the publisher node
        self.topicNameFrame = 'camera_frame'

        self.queueSize = 20

        # Create a subscriber object that will subscribe to the topic: self.topicNameFrame
        # The message type is Image and the queue size is self.queueSize
        self.subscription = self.create_subscription(
            Image,
            self.topicNameFrame,
            self.listener_callback,
            self.queueSize)
        
        self.subscription  # prevent unused variable warning

    def listener_callback(self, imageMessage):
        self.get_logger().info('Receiving an image')
        frame = self.bridge.imgmsg_to_cv2(imageMessage)
        if frame is None:
            self.get_logger().info('No object detected')
        else:
            self.get_logger().info('Object detected')

        self.get_logger().info('Displaying the image')

        # img = np.zeros((512, 512, 3), dtype=np.uint8)

        # Display the image
        # cv2.imshow('Test Image', img)
        # cv2.waitKey()
       
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        
def main(args=None):
    rclpy.init(args=args)   

    imageSubscriber = ImSub()

    rclpy.spin(imageSubscriber)

    imageSubscriber.destroy_node()
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()