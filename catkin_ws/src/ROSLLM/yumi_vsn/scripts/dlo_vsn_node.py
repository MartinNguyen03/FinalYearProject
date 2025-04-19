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
    
    # Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dict = torch.load(MODEL_PATH)
    model_config = model_dict["config"]
    model_weights = model_dict["model"]
    model = DLOPerceiver(
        iterations=model_config["iterations"],
        n_latents=model_config["n_latents"],
        latent_dim=model_config["latent_dim"],
        depth=model_config["depth"],
        dropout=model_config["dropout"],
        img_encoder_type=model_config["img_encoder_type"],
    )
    model.load_state_dict(model_weights)
    model.to(device=device)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    text_encoder = TextEncoder(model_name="distilbert-base-uncased")
    dlo_visn_node = dloVisionNode(dloPerciever=model)
    rospy.spin()
