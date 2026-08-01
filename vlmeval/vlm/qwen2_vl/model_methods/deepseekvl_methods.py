import torch
from transformers import AutoModelForCausalLM

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

import io
import base64
from PIL import Image

def pil_image_to_data_url(image: Image.Image, format="PNG"):
    buffered = io.BytesIO()
    image.save(buffered, format=format)  # PNG, JPEG 等
    img_bytes = buffered.getvalue()
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:image/{format.lower()};base64,{base64_str}"
    return data_url


# 暂不支持attention的提取