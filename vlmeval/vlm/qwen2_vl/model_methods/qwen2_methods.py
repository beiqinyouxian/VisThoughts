import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from skimage.measure import block_reduce
from ..utils import *
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
# currently select 22 but feel free to try other layers
ATT_LAYER = 24


def _get_layer_att(outputs, layer_idx):
    """提取指定层的注意力张量，兼容 sdpa（嵌套 tuple）和 eager（扁平 tensor）两种格式。"""
    att = outputs['attentions'][layer_idx]
    if isinstance(att, (tuple, list)):
        att = att[0]
    return att



def rel_attention_qwen2(image, prompt, general_prompt, model, processor):

    """
    Compute relative attention scores for Qwen2VL.
    
    This function computes the relative attention scores between the input and general inputs
    for Qwen2.5VL. It first computes the attention scores for the input and general inputs, 
    and finally computes the relative attention scores.
    """

    conversation = [{"role": "user","content": [{"type": "image",},{"type": "text", "text": prompt},],}]
    general_conversation = [{"role": "user","content": [{"type": "image",},{"type": "text", "text": general_prompt},],}]
    # Preprocess the inputs
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    general_text_prompt = processor.apply_chat_template(general_conversation, add_generation_prompt=True)

    inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to(model.device, torch.bfloat16)
    general_inputs = processor(text=[general_text_prompt], images=[image], padding=True, return_tensors="pt").to(model.device, torch.bfloat16)

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

def orin_attention_qwen2(image, prompt, general_prompt, model, processor):

    """
    Compute relative attention scores for Qwen2.5VL.
    
    This function computes the relative attention scores between the input and general inputs
    for Qwen2.5VL. It first computes the attention scores for the input and general inputs, 
    and finally computes the relative attention scores.
    """

    conversation = [{"role": "user","content": [{"type": "image",},{"type": "text", "text": prompt},],}]
    # Preprocess the inputs
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

    inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to(model.device, torch.bfloat16)
    att_shape = (inputs['image_grid_thw'][0, 1:] / 2).cpu().numpy().astype(int).tolist()
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    pos = inputs['input_ids'].tolist()[0].index(vision_start_token_id) + 1
    pos_end = inputs['input_ids'].tolist()[0].index(vision_end_token_id)
    outputs = model(**inputs, output_attentions=True)
    att = _get_layer_att(outputs, ATT_LAYER)[0, :, -1, pos:pos_end].mean(dim=0).to(torch.float32).detach().cpu().numpy()
    att_map = att.reshape(att_shape)
    return att_map

def auto_param_rel_attention_qwen2(image, prompt, general_prompt, model, processor, LAYERS, HEADS, start_layer=0):

    """
    Compute relative attention scores for Qwen2VL.
    
    This function computes the relative attention scores between the input and general inputs
    for Qwen2VL. It first computes the attention scores for the input and general inputs, 
    and finally computes the relative attention scores.
    
    LAYERS: 总层数；start_layer: 实际扫描起始层（默认 0 扫描全部，设为 >0 仅扫描后 N 层）。
    """

    conversation = [{"role": "user","content": [{"type": "image",},{"type": "text", "text": prompt},],}]
    general_conversation = [{"role": "user","content": [{"type": "image",},{"type": "text", "text": general_prompt},],}]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    general_text_prompt = processor.apply_chat_template(general_conversation, add_generation_prompt=True)

    inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to(model.device, torch.bfloat16)
    general_inputs = processor(text=[general_text_prompt], images=[image], padding=True, return_tensors="pt").to(model.device, torch.bfloat16)

    att_shape = (inputs['image_grid_thw'][0, 1:] / 2).cpu().numpy().astype(int).tolist()

    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')

    pos = inputs['input_ids'].tolist()[0].index(vision_start_token_id) + 1
    pos_end = inputs['input_ids'].tolist()[0].index(vision_end_token_id)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        general_outputs = model(**general_inputs, output_attentions=True)
       
        temp_result={}
        for LAYER in range(start_layer, LAYERS):
            for HEAD in range(HEADS):
                att = _get_layer_att(outputs, LAYER)[0, HEAD, -1, pos:pos_end].to(torch.float32).detach().cpu().numpy()
                general_att = _get_layer_att(general_outputs, LAYER)[0, HEAD, -1, pos:pos_end].to(torch.float32).detach().cpu().numpy()
                epsilon = 1e-7
                general_att_safe = np.where(general_att == 0, epsilon, general_att)
                att_map = att / general_att_safe
                att_map = att_map.reshape(att_shape)
                temp_result[f"{LAYER}_{HEAD}"] = att_map

    del outputs, general_outputs, inputs, general_inputs
    torch.cuda.empty_cache()
    
    return temp_result
 


def auto_param_orin_attention_qwen2(image, prompt, general_prompt, model, processor, LAYERS, HEADS, start_layer=0):

    """
    Compute original attention scores for Qwen2VL.
    
    This function computes the original attention scores (without relative normalization) for Qwen2VL.
    It extracts attention maps for all layers and heads.
    
    LAYERS: 总层数；start_layer: 实际扫描起始层（默认 0 扫描全部）。
    """

    conversation = [{"role": "user","content": [{"type": "image",},{"type": "text", "text": prompt},],}]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to(model.device, torch.bfloat16)
    att_shape = (inputs['image_grid_thw'][0, 1:] / 2).cpu().numpy().astype(int).tolist()
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    pos = inputs['input_ids'].tolist()[0].index(vision_start_token_id) + 1
    pos_end = inputs['input_ids'].tolist()[0].index(vision_end_token_id)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

        temp_result={}
        for LAYER in range(start_layer, LAYERS):
            for HEAD in range(HEADS):
                att = _get_layer_att(outputs, LAYER)[0, HEAD, -1, pos:pos_end].to(torch.float32).detach().cpu().numpy()

                att_map = att.reshape(att_shape)
                temp_result[f"{LAYER}_{HEAD}"] = att_map

    del outputs, inputs
    torch.cuda.empty_cache()
    
    return temp_result


def batch_rel_attention_qwen2(image, prompts, general_prompt, model, processor, LAYERS, HEADS, start_layer=0):
    """批量相对注意力提取：将所有关键词 prompt 合并为一次前向传播。

    返回:
        relative_maps: dict[keyword_index -> dict[head_key -> numpy_array]]
    """
    from qwen_vl_utils import process_vision_info

    num_keywords = len(prompts)
    if num_keywords == 0:
        return {}

    # 1) 构建批量消息：每个关键词一个对话，共享同一张图
    batch_messages = []
    for kw in prompts:
        batch_messages.append([
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": str(kw)},
                ],
            }
        ])

    # 2) 应用聊天模板 + 处理视觉信息
    texts = processor.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)
    images_batch, videos_batch = process_vision_info(batch_messages)
    batch_inputs = processor(
        text=texts, images=images_batch, videos=videos_batch,
        padding=True, return_tensors="pt",
    ).to(model.device, torch.bfloat16)

    # 3) 视觉 token 区间（所有样本共享同一张图，因此 vision token 位置应一致）
    vision_start_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    # 取第一个样本的 input_ids 确定 vision token 区间
    first_ids = batch_inputs['input_ids'][0].tolist()
    pos = first_ids.index(vision_start_id) + 1
    pos_end = first_ids.index(vision_end_id)

    # 4) 批量前向：所有关键词一次完成
    with torch.no_grad():
        batch_outputs = model(**batch_inputs, output_attentions=True)

    # 5) 通用提示仅需计算一次
    general_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": general_prompt},
            ],
        }
    ]
    gen_text = processor.apply_chat_template(general_messages, tokenize=False, add_generation_prompt=True)
    gen_images, gen_videos = process_vision_info(general_messages)
    gen_inputs = processor(
        text=gen_text, images=gen_images, videos=gen_videos,
        padding=True, return_tensors="pt",
    ).to(model.device, torch.bfloat16)

    with torch.no_grad():
        general_outputs = model(**gen_inputs, output_attentions=True)

    # 6) 逐关键词提取相对注意力图
    relative_maps: dict[int, dict[str, np.ndarray]] = {}

    for kw_idx in range(num_keywords):
        head_maps: dict[str, np.ndarray] = {}
        for LAYER in range(start_layer, LAYERS):
            for HEAD in range(HEADS):
                att = (
                    _get_layer_att(batch_outputs, LAYER)[kw_idx, HEAD, -1, pos:pos_end]
                    .to(torch.float32).detach().cpu().numpy()
                )
                gen_att = (
                    _get_layer_att(general_outputs, LAYER)[0, HEAD, -1, pos:pos_end]
                    .to(torch.float32).detach().cpu().numpy()
                )
                gen_att_safe = np.where(gen_att == 0, np.float32(1e-7), gen_att)
                att_shape = (
                    batch_inputs['image_grid_thw'][0, 1:] / 2
                ).cpu().numpy().astype(int).tolist()
                att_map = (att / gen_att_safe).reshape(att_shape)
                head_maps[f"{LAYER}_{HEAD}"] = att_map
        relative_maps[kw_idx] = head_maps

    del batch_outputs, general_outputs, batch_inputs, gen_inputs
    torch.cuda.empty_cache()

    return relative_maps