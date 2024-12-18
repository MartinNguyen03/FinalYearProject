#!/usr/bin/env python3.10.0
import cv2
import numpy as np
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from segment_anything import sam_model_registry, SamPredictor

class SamSubscriber(Node):
    def __init__(self):
        super().__init__('subscriber')
        self.bridge = CvBridge()
        self.sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
        self.predictor = SamPredictor(self.sam)
        self.topicNameFrame = 'camera_frame'
        self.queueSize = 20
        self.subscription = self.create_subscription(
            Image,
            self.topicNameFrame,
            self.listener_callback,
            self.queueSize)
    
    def listener_callback(self, imageMessage):
        self.get_logger().info('Receiving an image')
        frame = self.bridge.imgmsg_to_cv2(imageMessage, desired_encoding='bgr8')

        # SAM expects RGB input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb_frame)

        # Dummy points for segmentation (use real points/boxes for dynamic scenarios)
        input_points = np.array([[100, 100]])
        input_labels = np.array([1])

        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        for mask in masks:
            frame[mask > 0] = [0, 255, 0]  # Overlay segmentation

        cv2.imshow('SAM Segmentation', frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)   

    imageSubscriber = SamSubscriber()

    rclpy.spin(imageSubscriber)

    imageSubscriber.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
