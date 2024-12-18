#!/usr/bin/env python3.10.0
import cv2
from ultralytics import YOLO
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('subscriber')
        self.bridge = CvBridge()

        self.model = YOLO('yolov8n.pt')
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
        frame = self.bridge.imgmsg_to_cv2(imageMessage, desired_encoding='bgr8')

        res = self.model(frame)
        if res[0].boxes is None:
            self.get_logger().info('No object detected')
        else:
            self.get_logger().info('Object detected')

        for box in res[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = box.conf[0]  # Confidence score
            cls = box.cls[0]  # Class index
            label = f"{self.model.names[int(cls)]} {conf:.2f}"

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self.get_logger().info('Displaying the image')
        cv2.imshow('Camera Video', frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)   

    imageSubscriber = ImageSubscriber()

    rclpy.spin(imageSubscriber)

    imageSubscriber.destroy_node()
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()