#!/usr/bin/python3
import rospy
from dlo_vsn_coloured import dloVision


class dloVisionNode:
    auto_execution = False
    
    def __init__(self):
        # read parameters
        self.sl_visn = dloVision(self.auto_execution)    
        
if __name__ == "__main__":
    rospy.init_node('dlo_vsn_node', anonymous=True)
    
    dlo_vsn_node = dloVisionNode()
    rospy.spin()
