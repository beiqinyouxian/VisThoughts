import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from skimage.measure import block_reduce
from ..utils import *
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
# currently select 22 but feel free to try other layers
ATT_LAYER = 24


def _get_layer_att(outputs, layer_idx):
    """兼容 sdpa（嵌套 tuple）和 eager（扁平 tensor）两种 attention 输出格式。"""
    att = outputs['attentions'][layer_idx]
    if isinstance(att, (tuple, list)):
        att = att[0]
    return att


def prepare_qwen2_5_input(messages, processor):

    """
    Prepare the input for Qwen2.5VL.
    """

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    return inputs
def rel_attention_qwen2_5(image, prompt, general_prompt, model, processor):

    """
    Compute relative attention scores for Qwen2.5VL.
    
    This function computes the relative attention scores between the input and general inputs
    for Qwen2.5VL. It first computes the attention scores for the input and general inputs, 
    and finally computes the relative attention scores.
    """

    image_str = encode_base64(image)

    messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{image_str}'}, {"type": "text", "text": prompt}]}]
    general_messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{image_str}'}, {"type": "text", "text": general_prompt}]}]

    inputs = prepare_qwen2_5_input(messages, processor).to(model.device, torch.bfloat16)
    general_inputs = prepare_qwen2_5_input(general_messages, processor).to(model.device, torch.bfloat16)

    att_shape = (inputs['image_grid_thw'][0, 1:] / 2).cpu().numpy().astype(int).tolist()

    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')

    pos = inputs['input_ids'].tolist()[0].index(vision_start_token_id) + 1
    pos_end = inputs['input_ids'].tolist()[0].index(vision_end_token_id)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        general_outputs = model(**general_inputs, output_attentions=True)

        att = _get_layer_att(outputs, ATT_LAYER)[0, :, -1, pos:pos_end].mean(dim=0).to(torch.float32).detach().cpu().numpy()
        general_att = _get_layer_att(general_outputs, ATT_LAYER)[0, :, -1, pos:pos_end].mean(dim=0).to(torch.float32).detach().cpu().numpy()

        att_map = att / general_att

        att_map = att_map.reshape(att_shape)

        return att_map

def orin_attention_qwen2_5(image, prompt, general_prompt, model, processor):

    """
    Compute relative attention scores for Qwen2.5VL.
    
    This function computes the relative attention scores between the input and general inputs
    for Qwen2.5VL. It first computes the attention scores for the input and general inputs, 
    and finally computes the relative attention scores.
    """

    image_str = encode_base64(image)

    messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{image_str}'}, {"type": "text", "text": prompt}]}]

    inputs = prepare_qwen2_5_input(messages, processor).to(model.device, torch.bfloat16)

    att_shape = (inputs['image_grid_thw'][0, 1:] / 2).cpu().numpy().astype(int).tolist()

    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')

    pos = inputs['input_ids'].tolist()[0].index(vision_start_token_id) + 1
    pos_end = inputs['input_ids'].tolist()[0].index(vision_end_token_id)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
       

        att = _get_layer_att(outputs, ATT_LAYER)[0, :, -1, pos:pos_end].mean(dim=0).to(torch.float32).detach().cpu().numpy()



        att_map = att.reshape(att_shape)

        return att_map

def auto_param_rel_attention_qwen2_5(image, prompt, general_prompt, model, processor, LAYERS, HEADS):

    """
    Compute relative attention scores for Qwen2.5VL.
    
    This function computes the relative attention scores between the input and general inputs
    for Qwen2.5VL. It first computes the attention scores for the input and general inputs, 
    and finally computes the relative attention scores.
    """

    image_str = encode_base64(image)

    messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{image_str}'}, {"type": "text", "text": prompt}]}]
    general_messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{image_str}'}, {"type": "text", "text": general_prompt}]}]

    inputs = prepare_qwen2_5_input(messages, processor).to(model.device, torch.bfloat16)
    general_inputs = prepare_qwen2_5_input(general_messages, processor).to(model.device, torch.bfloat16)

    att_shape = (inputs['image_grid_thw'][0, 1:] / 2).cpu().numpy().astype(int).tolist()
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')

    pos = inputs['input_ids'].tolist()[0].index(vision_start_token_id) + 1
    pos_end = inputs['input_ids'].tolist()[0].index(vision_end_token_id)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        general_outputs = model(**general_inputs, output_attentions=True)
        temp_result={}
        for LAYER in range(LAYERS):
            for HEAD in range(HEADS):
                att = _get_layer_att(outputs, LAYER)[0, HEAD, -1, pos:pos_end].to(torch.float32).detach().cpu().numpy()
                general_att = _get_layer_att(general_outputs, LAYER)[0, HEAD, -1, pos:pos_end].to(torch.float32).detach().cpu().numpy()
                epsilon = 1e-7
                general_att_safe = np.where(general_att == 0, epsilon, general_att)
                att_map = att / general_att_safe
                # att_map = att / general_att
                att_map = att_map.reshape(att_shape)
                temp_result[f"{LAYER}_{HEAD}"] = att_map

    # 显式释放 GPU 上的注意力张量
    del outputs, general_outputs, inputs, general_inputs
    torch.cuda.empty_cache()

    return temp_result
 


def auto_param_orin_attention_qwen2_5(image, prompt, general_prompt, model, processor, LAYERS, HEADS):

    """
    Compute relative attention scores for Qwen2.5VL.
    
    This function computes the relative attention scores between the input and general inputs
    for Qwen2.5VL. It first computes the attention scores for the input and general inputs, 
    and finally computes the relative attention scores.
    """

    image_str = encode_base64(image)

    messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{image_str}'}, {"type": "text", "text": prompt}]}]

    inputs = prepare_qwen2_5_input(messages, processor).to(model.device, torch.bfloat16)

    att_shape = (inputs['image_grid_thw'][0, 1:] / 2).cpu().numpy().astype(int).tolist()

    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')

    pos = inputs['input_ids'].tolist()[0].index(vision_start_token_id) + 1
    pos_end = inputs['input_ids'].tolist()[0].index(vision_end_token_id)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

       
        temp_result={}
        for LAYER in range(LAYERS):
            for HEAD in range(HEADS):
                att = _get_layer_att(outputs, LAYER)[0, HEAD, -1, pos:pos_end].to(torch.float32).detach().cpu().numpy()

  
                att_map = att.reshape(att_shape)
                temp_result[f"{LAYER}_{HEAD}"] = att_map

    # 显式释放 GPU 上的注意力张量
    del outputs, inputs
    torch.cuda.empty_cache()

    return temp_result