#!/usr/bin/python3
import rospy
import numpy as np
import cv2
import os
# from realsense_camera import RealsenseCamera
# from colour_segmentation import ColourSegmentation
from utils.colour_segmentation import ColourSegmentation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(BASE_DIR, "../results")

if __name__ == '__main__':
    rospy.init_node('color_tuner', anonymous=True)
   
    img = cv2.imread(os.path.join(RESULT_PATH, 'img_raw_bgr.png'))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[0:550, 380:960]  # Crop the image to the region of interest
    seg = ColourSegmentation([90, 20, 0], [193, 145, 85], img, kernel_size=3, morph_op=[2,3], live_adjust=True)
    # cam = RealsenseCamera('yumi_d435_r', jetson=False)
    # seg = ColourSegmentation([0, 0, 110], [125, 115, 255], cam.read_image, kernel_size=3, morph_op=[2,3], live_adjust=True)
    print('{}, {}'.format(seg.thresh_l, seg.thresh_h))

    # print(cv2.cvtColor(np.uint8([[seg.thresh_l]]),cv2.COLOR_BGR2HSV))
    # print(cv2.cvtColor(np.uint8([[seg.thresh_h]]),cv2.COLOR_BGR2HSV))

    # [0, 130, 155], [20, 190, 205] d435_r hsv [0, 30, 155], [130, 125, 0] rgb
    # [51, 90, 155], [12, 190, 255] d435 hsv  [90, 20, 0], [193, 145, 75] rgb [0, 0, 100], [110, 80, 0] [0, 0, 125], [110, 110, 0]