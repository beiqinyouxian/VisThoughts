import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from skimage.measure import block_reduce
from ..utils import *
from transformers import AutoProcessor, LlavaForConditionalGeneration,LlavaProcessor,AutoModelForCausalLM
# hyperparameters
NUM_IMG_TOKENS = 576
NUM_PATCHES = 24
PATCH_SIZE = 14
IMAGE_RESOLUTION = 336
IMAGE_TOKEN_INDEX = 32000
ATT_LAYER = 14


def _get_layer_att(att_list, layer_idx):
    """兼容 sdpa（嵌套 tuple）和 eager（扁平 tensor）两种 attention 输出格式。"""
    att = att_list[layer_idx]
    if isinstance(att, (tuple, list)):
        att = att[0]
    return att


LLAVA_V1_5_SYSTEM_PROMPT = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions."
)


def _ensure_processor_llava_fields(processor, model):
    if getattr(processor, 'patch_size', None) is None:
        try:
            processor.patch_size = model.config.vision_config.patch_size
        except Exception:
            pass
    if getattr(processor, 'vision_feature_select_strategy', None) is None:
        processor.vision_feature_select_strategy = getattr(
            model.config, 'vision_feature_select_strategy', 'default'
        )
    if getattr(processor, 'num_additional_image_tokens', 0) in (None, 0):
        processor.num_additional_image_tokens = 1


def _collect_image_token_ids(model):
    special_ids = set()
    for attr in ['image_token_index', 'image_token_id']:
        value = getattr(model.config, attr, None)
        if isinstance(value, int) and value >= 0:
            special_ids.add(value)
    if not special_ids:
        special_ids.add(IMAGE_TOKEN_INDEX)
    return special_ids


def _build_llava_v15_prompt(text: str) -> str:
    return f'{LLAVA_V1_5_SYSTEM_PROMPT} USER: <image>\n{text} ASSISTANT:'


def _find_image_token_start(input_ids, model) -> int:
    ids = input_ids[0].tolist() if hasattr(input_ids, '__getitem__') else list(input_ids)
    special_ids = _collect_image_token_ids(model)
    for idx, token_id in enumerate(ids):
        if token_id in special_ids:
            return idx
    raise ValueError(
        f'Image token(s) {sorted(special_ids)} not found in input_ids (len={len(ids)}). '
        'Ensure prompt contains <image> placeholder.'
    )


def _prepare_llava_attention_inputs(image, prompt, model, processor):
    _ensure_processor_llava_fields(processor, model)
    text = _build_llava_v15_prompt(str(prompt))
    target_dtype = next(model.parameters()).dtype
    inputs = processor(
        images=[image],
        text=text,
        return_tensors='pt',
    ).to(model.device, target_dtype)
    pos = _find_image_token_start(inputs['input_ids'], model)
    return inputs, pos


def _safe_rel_attention_ratio(att_map, general_att_map, eps=1e-8):
    """相对注意力 = specific / general；分母为 0 或近 0 的位置置 0，避免除零警告。"""
    att_map = np.asarray(att_map, dtype=np.float32)
    general_att_map = np.asarray(general_att_map, dtype=np.float32)
    ratio = np.zeros_like(att_map, dtype=np.float32)
    valid = general_att_map > eps
    np.divide(att_map, general_att_map, out=ratio, where=valid)
    return ratio


def gradient_attention_llava(image, prompt, general_prompt, model, processor):
    """
    Generates an attention map using gradient-weighted attention from LLaVA model.
    
    This function computes attention maps from the LLaVA model and weights them by their
    gradients with respect to the loss. It focuses on the attention paid to image tokens
    in the final token prediction, highlighting regions relevant to the prompt.
    
    Args:
        image: Input image to analyze
        prompt: Text prompt for which to generate attention
        general_prompt: General text prompt (not directly used in this function)
        model: LLaVA model instance
        processor: LLaVA processor for preparing inputs
        
    Returns:
        att_map: A 2D numpy array of shape (NUM_PATCHES, NUM_PATCHES) representing 
                the gradient-weighted attention map
    """
    inputs, pos = _prepare_llava_attention_inputs(image, prompt, model, processor)
    
    # Compute loss
    outputs = model(**inputs, output_attentions=True)
    CE = nn.CrossEntropyLoss()
    zero_logit = outputs.logits[:, -1, :]
    true_class = torch.argmax(zero_logit, dim=1)
    loss = -CE(zero_logit, true_class)
    
    # Compute attention and gradients
    attention = outputs.attentions[ATT_LAYER]
    grads = torch.autograd.grad(loss, attention, retain_graph=True)
    grad_att = attention * F.relu(grads[0])
    
    # Compute the attention maps
    att_map = grad_att[0, :, -1, pos:pos+NUM_IMG_TOKENS].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
    
    return att_map

def rel_attention_llava(image, prompt, general_prompt, model, processor):
    """
    Generates a relative attention map by comparing specific prompt attention to general prompt attention.
    
    This function computes attention maps for both a specific prompt and a general prompt in the LLaVA model,
    then calculates their ratio to highlight regions that are uniquely relevant to the specific prompt.
    It focuses on the attention paid to image tokens in the final token prediction.
    
    Args:
        image: Input image to analyze
        prompt: Specific text prompt for which to generate attention
        general_prompt: General text prompt for baseline comparison
        model: LLaVA model instance
        processor: LLaVA processor for preparing inputs
        
    Returns:
        att_map: A 2D numpy array of shape (NUM_PATCHES, NUM_PATCHES) representing 
                the relative attention map (specific/general)
    """
    inputs, pos = _prepare_llava_attention_inputs(image, prompt, model, processor)
    general_inputs, general_pos = _prepare_llava_attention_inputs(
        image, general_prompt, model, processor
    )

    # Compute attention map for the 14th layer
    att_map = _get_layer_att(model(**inputs, output_attentions=True)['attentions'], ATT_LAYER)[0, :, -1, pos:pos+NUM_IMG_TOKENS].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
    # att_map = _get_layer_att(model(**inputs, output_attentions=True)['attentions'], 14)[0, 13, -1, pos:pos+NUM_IMG_TOKENS].to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
 

    # Compute general attention map for the 14th layer
    general_att_map = _get_layer_att(model(**general_inputs, output_attentions=True)['attentions'], ATT_LAYER)[0, :, -1, general_pos:general_pos+NUM_IMG_TOKENS].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
    # general_att_map = _get_layer_att(model(**general_inputs, output_attentions=True)['attentions'], 14)[0, 13, -1, general_pos:general_pos+NUM_IMG_TOKENS].to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
 
    # Normalize attention map
    att_map = _safe_rel_attention_ratio(att_map, general_att_map)

    return att_map


def enhanced_rel_attention_llava(image, prompt, general_prompt, model, processor):
    inputs, pos = _prepare_llava_attention_inputs(image, prompt, model, processor)
    general_inputs, general_pos = _prepare_llava_attention_inputs(
        image, general_prompt, model, processor
    )

    selected_heads = [13, 24, 26]

    # Compute attention map for the 14th layer
    # att_map = _get_layer_att(model(**inputs, output_attentions=True)['attentions'], ATT_LAYER)[0, :, -1, pos:pos+NUM_IMG_TOKENS].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
    att_map = _get_layer_att(model(**inputs, output_attentions=True)['attentions'], 14)[0,selected_heads, -1, pos:pos+NUM_IMG_TOKENS].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
 

    # Compute general attention map for the 14th layer
    # general_att_map = _get_layer_att(model(**general_inputs, output_attentions=True)['attentions'], ATT_LAYER)[0, :, -1, general_pos:general_pos+NUM_IMG_TOKENS].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
    general_att_map = _get_layer_att(model(**general_inputs, output_attentions=True)['attentions'], 14)[0, :, -1, general_pos:general_pos+NUM_IMG_TOKENS].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
 
    # Normalize attention map
    att_map = _safe_rel_attention_ratio(att_map, general_att_map)

    return att_map

def orin_attention_llava(image, prompt, general_prompt, model, processor):
    inputs, pos = _prepare_llava_attention_inputs(image, prompt, model, processor)

    # Compute attention map for the 14th layer
    att_map = _get_layer_att(model(**inputs, output_attentions=True)['attentions'], ATT_LAYER)[0, :, -1, pos:pos+NUM_IMG_TOKENS].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
    # att_map = _get_layer_att(model(**inputs, output_attentions=True)['attentions'], 14)[0, 24, -1, pos:pos+NUM_IMG_TOKENS].to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
 
    return att_map

def special_orin_attention_llava(image, prompt, general_prompt, model, processor):
    
    selected_heads = [13, 24, 26]
    inputs, pos = _prepare_llava_attention_inputs(image, prompt, model, processor)

    # Compute attention map for the 14th layer
    # att_map = _get_layer_att(model(**inputs, output_attentions=True)['attentions'], ATT_LAYER)[0, :, -1, pos:pos+NUM_IMG_TOKENS].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
    att_map = _get_layer_att(model(**inputs, output_attentions=True)['attentions'], 14)[0, selected_heads, -1, pos:pos+NUM_IMG_TOKENS].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
 
    return att_map

def pure_gradient_llava(image, prompt, general_prompt, model, processor):
    """
    Generates a gradient-based attention map using direct image gradients in LLaVA.
    
    This function computes gradients of the loss with respect to the input image pixels
    for both specific and general prompts. It then calculates their ratio and applies
    a high-pass filter to highlight fine-grained details that are uniquely relevant to the specific prompt.
    
    Args:
        image: Input image to analyze
        prompt: Specific text prompt for which to generate gradients
        general_prompt: General text prompt for baseline comparison
        model: LLaVA model instance
        processor: LLaVA processor for preparing inputs
        
    Returns:
        grad: A 2D numpy array representing the processed gradient map highlighting
              regions relevant to the specific prompt
    """
    inputs, _ = _prepare_llava_attention_inputs(image, prompt, model, processor)
    general_inputs, _ = _prepare_llava_attention_inputs(image, general_prompt, model, processor)
    
    # Apply high pass filter
    high_pass = high_pass_filter(image, IMAGE_RESOLUTION, reduce=False)
    
    # Enable gradients
    inputs['pixel_values'].requires_grad = True
    general_inputs['pixel_values'].requires_grad = True
    
    # Initialize loss criterion
    criterion = nn.CrossEntropyLoss()
    
    # Forward pass for inputs
    zero_logit = model(**inputs, output_hidden_states=False).logits[:, -1, :]
    true_class = torch.argmax(zero_logit, dim=1)
    loss = -criterion(zero_logit, true_class)
    
    # Compute gradients
    grads = torch.autograd.grad(loss, inputs['pixel_values'], retain_graph=True)[0]
    
    # Forward pass for general_inputs
    general_zero_logit = model(**general_inputs, output_hidden_states=False).logits[:, -1, :]
    general_true_class = torch.argmax(general_zero_logit, dim=1)
    general_loss = -criterion(general_zero_logit, general_true_class)
    
    # Compute general gradients
    general_grads = torch.autograd.grad(general_loss, general_inputs['pixel_values'], retain_graph=True)[0]
    
    # Process gradients
    grads = grads.to(torch.float32).detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    general_grads = general_grads.to(torch.float32).detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    
    # Compute gradient norms
    grad = np.linalg.norm(grads, axis=2)
    general_grad = np.linalg.norm(general_grads, axis=2)
    
    # Normalize and apply high pass filter
    grad = _safe_rel_attention_ratio(grad, general_grad)
    high_pass = high_pass > np.median(high_pass)
    grad = grad * high_pass
    
    # Reduce gradient block size
    grad = block_reduce(grad, block_size=(PATCH_SIZE, PATCH_SIZE), func=np.mean)
    
    return grad


def auto_param_rel_attention_llava(image, prompt, general_prompt, model, processor, LAYERS, HEADS):
    inputs, pos = _prepare_llava_attention_inputs(image, prompt, model, processor)
    general_inputs, general_pos = _prepare_llava_attention_inputs(
        image, general_prompt, model, processor
    )

    # Compute attention map for the 14th layer
    att_output = model(**inputs, output_attentions=True)['attentions']
    general_output = model(**general_inputs, output_attentions=True)['attentions']
 
    temp_result={}
    for LAYER in range(LAYERS):
        for HEAD in range(HEADS):
            
            att_map = _get_layer_att(att_output, LAYER)[0, HEAD, -1, pos:pos+NUM_IMG_TOKENS].to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
            general_att_map = _get_layer_att(general_output, LAYER)[0, HEAD, -1, general_pos:general_pos+NUM_IMG_TOKENS].to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
            att_map = _safe_rel_attention_ratio(att_map, general_att_map)
            temp_result[f"{LAYER}_{HEAD}"] = att_map

    # 显式释放 GPU 上的注意力张量
    del att_output, general_output, inputs, general_inputs
    torch.cuda.empty_cache()

    return temp_result


def auto_param_orin_attention_llava(image, prompt, general_prompt, model, processor, LAYERS, HEADS):
    inputs, pos = _prepare_llava_attention_inputs(image, prompt, model, processor)


    # Compute attention map for the 14th layer
    att_output = model(**inputs, output_attentions=True)['attentions']

 
    temp_result={}
    for LAYER in range(LAYERS):
        for HEAD in range(HEADS):
            
            att_map = _get_layer_att(att_output, LAYER)[0, HEAD, -1, pos:pos+NUM_IMG_TOKENS].to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)

            temp_result[f"{LAYER}_{HEAD}"] = att_map

    # 显式释放 GPU 上的注意力张量
    del att_output, inputs
    torch.cuda.empty_cache()

    return temp_result

def manual_param_rel_attention_llava(image, prompt, general_prompt, model, processor, LAYER, HEAD):
    inputs, pos = _prepare_llava_attention_inputs(image, prompt, model, processor)
    general_inputs, general_pos = _prepare_llava_attention_inputs(
        image, general_prompt, model, processor
    )

    # Compute attention map for the 14th layer
    att_map = _get_layer_att(model(**inputs, output_attentions=True)['attentions'], LAYER)[0, HEAD, -1, pos:pos+NUM_IMG_TOKENS].to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
    general_att_map = _get_layer_att(model(**general_inputs, output_attentions=True)['attentions'], LAYER)[0, HEAD, -1, general_pos:general_pos+NUM_IMG_TOKENS].to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
 
    # Normalize attention map
    att_map = _safe_rel_attention_ratio(att_map, general_att_map)

    return att_map


def manual_param_orin_attention_llava(image, prompt, general_prompt, model, processor, LAYER, HEAD):
    inputs, pos = _prepare_llava_attention_inputs(image, prompt, model, processor)

    # Compute attention map for the 14th layer
    att_map = _get_layer_att(model(**inputs, output_attentions=True)['attentions'], LAYER)[0, HEAD, -1, pos:pos+NUM_IMG_TOKENS].to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)

    return att_map