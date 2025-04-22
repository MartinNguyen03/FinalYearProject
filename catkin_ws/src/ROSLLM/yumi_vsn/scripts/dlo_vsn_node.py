#!/usr/bin/python3
import os, sys
import rospy
from dlo_vsn_coloured import shoelacingVision
import torch
MODEL_DIR = os.path.expanduser('~/Documents/FinalYearProject/FinalYearProject/dlo_perceiver')  # <-- update this
if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)
MODEL_PATH = os.path.join(MODEL_DIR, "dlo_perceiver.pt")
from model_contrastive import DLOPerceiver
from text_encoder import TextEncoder
from transformers import DistilBertTokenizer


class dloVisionNode:
    auto_execution = True
    
    
    def __init__(self):
        # read parameters
        self.sim = rospy.get_param("~sim", False)
        self.sl_visn = shoelacingVision(self.auto_execution, self.sim)    
        
if __name__ == "__main__":
    rospy.init_node('dlo_visn_node', anonymous=True)
    
    
    dlo_visn_node = dloVisionNode()
    rospy.spin()
