import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))  # Adjust as needed
sys.path.insert(0, project_root)
import torch
import rospy
from transformers import AutoModelForCausalLM
from extern.DeepSeekVL.deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from extern.DeepSeekVL.deepseek_vl.utils.io import load_pil_images
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
# from PIL import Image as PILImage

class VLM:
    def __init__(self, model_path: str):
        self.bridge = CvBridge()
        device = torch.device("cpu")
        rospy.loginfo(f"Using device: {device}")
        self.processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer

        self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, 
                                                          trust_remote_code=True, 
                                                          torch_dtype=torch.bfloat16).to(device).eval()

    def call(self, prompt: str, img_path: str) -> str:
        # Convert ROS image message to OpenCV

        
        rospy.loginfo(f"Processing image from path: {img_path}")
        # Process input
        conversation = [
            {
                "role": "User", 
                "content": "<image_placeholder>" + prompt, 
                "images": [img_path]
            },
            {
                "role": "Assistant", 
                "content": ""
            }
        ]

        rospy.loginfo(f"Conversation: {conversation}")
        pil_image = load_pil_images(conversation)
        
        inputs = self.processor(conversations=conversation, 
                                images=pil_image, 
                                force_batchify=True).to(self.model.device)
        rospy.loginfo(f"Inputs prepared: {inputs}")
        # Get image embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**inputs)
        rospy.loginfo(f"Inputs embeddings prepared: {inputs_embeds.shape}")
        # Generate response
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        response = self.tokenizer.decode(
                    outputs[0].cpu().tolist(), 
                    skip_special_tokens=True)
        rospy.loginfo(f"Generated response: {response}")
        return response
