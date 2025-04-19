import sys
import os
deepseek_vl_path = os.path.join(os.path.dirname(__file__), "../extern/DeepSeek-VL")
sys.path.append(deepseek_vl_path)
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class VLM:
    def __init__(self, model_path: str):
        self.bridge = CvBridge()
        self.processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer

        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.model = self.model.to(torch.bfloat16).cuda().eval()

    def __call__(self, prompt: str, img_msg: Image):
        # Convert ROS image message to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

        # Convert to PIL image format (DeepSeek requires PIL images)
        pil_image = [cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)]

        # Process input
        conversation = [
            {"role": "User", "content": "<image_placeholder>" + prompt, "images": pil_image},
            {"role": "Assistant", "content": ""}
        ]

        inputs = self.processor(conversations=conversation, images=pil_image, force_batchify=True).to(self.model.device)

        # Get image embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**inputs)

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

        response = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return response
