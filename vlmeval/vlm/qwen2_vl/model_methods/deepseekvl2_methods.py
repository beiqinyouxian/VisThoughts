import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
import io
import base64
from PIL import Image

# IMAGE_TOKEN_INDEX=128815 # tiny
IMAGE_TOKEN_INDEX=100003 # small
NUM_IMG_TOKENS = 1024
NUM_PATCHES = 32
# ATT_LAYER = 12
# ATT_HEAD=10
IMAGE_RESOLUTION=None


def pil_image_to_data_url(image: Image.Image, format="PNG"):
    buffered = io.BytesIO()
    image.save(buffered, format=format)  # PNG, JPEG 等
    img_bytes = buffered.getvalue()
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:image/{format.lower()};base64,{base64_str}"
    return data_url

def split_model(model_name):
    global IMAGE_TOKEN_INDEX
    device_map = {}
    model_splits = {        
        '/data/VLM/deepseek-vl2-tiny': [3, 4, 4, 4], # 2 GPU
        '/data/VLM/deepseek-vl2-small': [6, 8, 8, 8], # 4 GPU 
        # '/data/VLM/deepseek-vl2': [10,10,10], # 3 GPU
        '/data/VLM/deepseek-vl2': [6, 8, 8, 8], # 4 GPU
    }
    if "tiny" in model_name:
        IMAGE_TOKEN_INDEX=128815 # tiny
    elif "small" in model_name:
        IMAGE_TOKEN_INDEX=100003 # small
    num_layers_per_gpu = model_splits[model_name]
    num_layers = sum(num_layers_per_gpu)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language.model.layers.{layer_cnt}'] = i 
            layer_cnt += 1
    device_map['vision'] = 0
    device_map['projector'] = 0
    device_map['image_newline'] = 0
    device_map['view_seperator'] = 0
    device_map['language.model.embed_tokens'] = 0
    device_map['language.model.norm'] = 0
    device_map['language.lm_head'] = 0
    device_map[f'language.model.layers.{num_layers - 1}'] = 0
    return device_map

def auto_param_rel_attention_deepseekvl2(image, prompt, general_prompt, model, processor, LAYERS, HEADS):
    # Prepare inputs for the prompt
    data_url = pil_image_to_data_url(image)
    conversation = [
    {
        "role": "<|User|>",
        "content": "<image>\n"
                   f"{prompt}",
        "images": [
            data_url,
        ],
    },
    {"role": "<|Assistant|>", "content": ""}
    ]

    general_conversation = [
    {
        "role": "<|User|>",
        "content": "<image>\n"
                   f"{general_prompt}",
        "images": [
            data_url,
        ],
    },
    {"role": "<|Assistant|>", "content": ""}
    ]

    pil_images=[image]
    prepare_inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(model.device)
    general_prepare_inputs = processor(
        conversations=general_conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(model.device)
    # run image encoder to get the image embeddings
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
    general_inputs_embeds = model.prepare_inputs_embeds(**general_prepare_inputs)

    pos = prepare_inputs['input_ids'][0].tolist().index(IMAGE_TOKEN_INDEX)
    general_pos=general_prepare_inputs['input_ids'][0].tolist().index(IMAGE_TOKEN_INDEX)

    # Compute attention map 
    att_output = model.language(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    use_cache=True,
    output_attentions=True
    )['attentions']

    general_output = model.language(
    inputs_embeds=general_inputs_embeds,
    attention_mask=general_prepare_inputs.attention_mask,
    use_cache=True,
    output_attentions=True
    )['attentions']
  
    temp_result={}
    for LAYER in range(LAYERS):
        for HEAD in range(HEADS):
            att_map = att_output[LAYER][0, HEAD, -1, pos:pos+NUM_IMG_TOKENS].to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
            general_att_map = general_output[LAYER][0, HEAD, -1, general_pos:general_pos+NUM_IMG_TOKENS].to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
            att_map = att_map / general_att_map
            temp_result[f"{LAYER}_{HEAD}"] = att_map
    return temp_result

def auto_param_orin_attention_deepseekvl2(image, prompt, general_prompt, model, processor, LAYERS, HEADS):
    # Prepare inputs for the prompt
    data_url = pil_image_to_data_url(image)
    conversation = [
    {
        "role": "<|User|>",
        "content": "<image>\n"
                   f"{prompt}",
        "images": [
            data_url,
        ],
    },
    {"role": "<|Assistant|>", "content": ""}
    ]

    pil_images=[image]
    prepare_inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(model.device)
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)


    pos = prepare_inputs['input_ids'][0].tolist().index(IMAGE_TOKEN_INDEX)


    # Compute attention map 
    att_output = model.language(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    use_cache=True,
    output_attentions=True
    )['attentions']

    temp_result={}
    for LAYER in range(LAYERS):
        for HEAD in range(HEADS):
            att_map = att_output[LAYER][0, HEAD, -1, pos:pos+NUM_IMG_TOKENS].to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
            temp_result[f"{LAYER}_{HEAD}"] = att_map
    return temp_result

def manual_param_rel_attention_deepseekvl2(image, prompt, general_prompt, model, processor, LAYER, HEAD):
    
     # Prepare inputs for the prompt
    data_url = pil_image_to_data_url(image)
    conversation = [
    {
        "role": "<|User|>",
        "content": "<image>\n"
                   f"{prompt}",
        "images": [
            data_url,
        ],
    },
    {"role": "<|Assistant|>", "content": ""}
    ]

    general_conversation = [
    {
        "role": "<|User|>",
        "content": "<image>\n"
                   f"{general_prompt}",
        "images": [
            data_url,
        ],
    },
    {"role": "<|Assistant|>", "content": ""}
    ]

    pil_images=[image]
    prepare_inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(model.device)
    general_prepare_inputs = processor(
        conversations=general_conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(model.device)

    # run image encoder to get the image embeddings
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
    general_inputs_embeds = model.prepare_inputs_embeds(**general_prepare_inputs)

    pos = prepare_inputs['input_ids'][0].tolist().index(IMAGE_TOKEN_INDEX)
    general_pos=general_prepare_inputs['input_ids'][0].tolist().index(IMAGE_TOKEN_INDEX)

    # Compute attention map 
    att_output = model.language(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    use_cache=True,
    output_attentions=True
    )['attentions']

    general_output = model.language(
    inputs_embeds=general_inputs_embeds,
    attention_mask=general_prepare_inputs.attention_mask,
    use_cache=True,
    output_attentions=True
    )['attentions']


    att_map = att_output[LAYER][0, HEAD, -1, pos:pos+NUM_IMG_TOKENS].to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
    general_att_map = general_output[LAYER][0, HEAD, -1, general_pos:general_pos+NUM_IMG_TOKENS].to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
    att_map = att_map / general_att_map

    return att_map


def manual_param_orin_attention_deepseekvl2(image, prompt, general_prompt, model, processor, LAYER, HEAD):
    
    # Prepare inputs for the prompt
    data_url = pil_image_to_data_url(image)
    conversation = [
    {
        "role": "<|User|>",
        "content": " <image>\n"
                   f"{prompt}",
        "images": [
            data_url,
        ],
    },
    {"role": "<|Assistant|>", "content": ""}
    ]

    
    pil_images=[image]
    prepare_inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(model.device)
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)


    pos = prepare_inputs['input_ids'][0].tolist().index(IMAGE_TOKEN_INDEX)


    # Compute attention map 
    att_output = model.language(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    use_cache=True,
    output_attentions=True
    )['attentions']


    # Compute attention map for the 14th layer
    att_map = att_output[LAYER][0, HEAD, -1, pos:pos+NUM_IMG_TOKENS].to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
    return att_map