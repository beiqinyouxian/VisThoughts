from __future__ import annotations
import csv
import hashlib
import json
import importlib
import logging
import math
import os
import re
import tempfile
import warnings
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import StoppingCriteria

from PIL import Image
from vlmeval.dataset import DATASET_MODALITY
from vlmeval.smp import get_cache_path, get_gpu_memory, listinstr
from ..base import BaseModel
from .prompt_enhance import LLaVAPromptMixinEnhance

# 与注意力模块相关的包
from .model_methods import llava_methods
from .utils import *
from .auto_threshold import auto_otsu

VLLM_MAX_IMAGE_INPUT_NUM = 24

# LLaVA-1.5(Vicuna) 默认系统提示：部分 HF 权重的 processor 不带 chat_template，
# 此时回退到该 Vicuna 对话格式手工拼接 prompt。
LLAVA_V1_5_SYSTEM_PROMPT = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions."
)

# 首先检查媒体文件是否是支持的URL前缀，是则原样返回，若不是且本地文件存在就转换成 file:// 开头的 URL 返回（'file://' + image）
def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def create_image_content(image_path, min_pixels, max_pixels):
    base64_image, mime_type = encode_image(image_path)
    return {
        "type": "image",
        "image": f"data:{mime_type};base64,{base64_image}",
        'min_pixels': min_pixels,
        'max_pixels': max_pixels
    }


def encode_image(image_path, max_side=None):
    from mimetypes import guess_type
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    image_format = mime_type.split("/")[-1].upper() if mime_type else "JPEG"

    from PIL import Image
    image = Image.open(image_path)
    # Handle the alpha channel
    if image.mode == "RGBA":
        image = _rgba_to_rgb(image)
    if max_side:
        image = _resize_image(image, max_side)
    encoded_image = _encode_image(image, image_format)

    return encoded_image, mime_type


def _encode_image(image, image_format):
    from io import BytesIO
    with BytesIO() as output:
        image.convert("RGB").save(output, format=image_format)
        import base64
        base64_encoded_data = base64.b64encode(output.getvalue()).decode("utf-8")
    return base64_encoded_data


def _rgba_to_rgb(image):
    from PIL import Image
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, image).convert("RGB")


def _resize_image(image, max_side):
    resize_scale = max_side / max(image.size)
    new_size = (
        int(image.size[0] * resize_scale),
        int(image.size[1] * resize_scale),
    )
    return image.resize(new_size)


def process_video(video_path, num_frames, min_pixels, max_pixels):
    import cv2

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second

    # the sampling rate using max number of frames
    sampling_gap_maxframe = (
        1 if not num_frames else math.ceil(frame_count / num_frames)
    )
    sampling_gap = max(math.ceil(fps / 5), sampling_gap_maxframe)

    frame_number = 0
    images = []

    while True:
        import tempfile
        success, frame = cap.read()
        if not success:
            break
        # Sample frames based on the dynamic sampling rate
        if frame_number % sampling_gap == 0:
            # Create a temporary file for the frame
            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False
            ) as temp_frame:
                cv2.imwrite(temp_frame.name, frame)
                images.append(create_image_content(temp_frame.name, min_pixels, max_pixels))
                os.remove(temp_frame.name)
        frame_number += 1
    if frame_number == 0:
        raise ValueError(f"Failed to read video from {video_path}, check data...")
    logging.info(
        f"Sampled {len(images)}/{frame_number} frames from video {video_path}"
    )
    cap.release()
    return images


class KeywordsStoppingCriteria(StoppingCriteria):
    # 自定义停止准则：
    # 在生成过程中一旦检测到“任意目标关键词（或其 token 序列）已出现”，
    # 就提前终止继续解码，避免无意义的冗长生成。
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            # 将每个关键词编码成 token id 序列，供后续做“尾部匹配”。
            cur_keyword_ids = tokenizer(keyword).input_ids
            if (
                len(cur_keyword_ids) > 1
                and cur_keyword_ids[0] == tokenizer.bos_token_id
            ):
                # 去掉可能存在的 BOS，保证匹配时与真实生成片段对齐。
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        # 记录初始输入长度，用于仅检查“新生成”的部分。
        self.start_len = input_ids.shape[1]

    def __call__(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        # 目前仅支持 batch size = 1 的停止判断。
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        # 只解码最近窗口（最长关键词长度），降低每步检查开销。
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [
            keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids
        ]
        for keyword_id in self.keyword_ids:
            # 先做 token 级精确匹配：若序列尾部命中关键词 token，立即停止。
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        # 再做文本级匹配：处理分词差异或边界情况，作为兜底判断。
        outputs = self.tokenizer.batch_decode(
            output_ids[:, -offset:], skip_special_tokens=True
        )[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        # 未命中任何关键词，继续生成。
        return False


CHAT_TEMPLATE = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"  # noqa: E501

UNTIL = ["<|diff_marker|>"]


class LLaVAEnhance(LLaVAPromptMixinEnhance, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = False

    # 功能：初始化模型、推理后端与两阶段注意力相关配置。
    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.001, # top_p：柔性按概率总和砍候选
        top_k=1, # top_k：硬性按固定个数砍候选
        temperature=0.01,
        repetition_penalty=1.0,
        do_sample: bool = True,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,  # if True, will try to only extract stuff in the last \boxed{}.
        verbose: bool = False,
        use_audio_in_video: bool = False,
        **kwargs,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        self.generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.use_audio_in_video = use_audio_in_video

        # 两阶段注意力参数相关配置
        self.model_type = kwargs.pop('model_type', 'llava')
        two_stage_config = kwargs.pop('two_stage_config', None)
        if two_stage_config is None:
            two_stage_config = {}
        elif not isinstance(two_stage_config, dict):
            logging.warning('two_stage_config should be a dict, fallback to default two-stage settings.')
            two_stage_config = {}
        # ----在此处设置默认的推理模式----
        self.reasoning_mode = kwargs.pop('reasoning_mode', two_stage_config.get('reasoning_mode', 'two_stage_attention'))
        # 参数选择 single_stage 或者 two_stage_attention


        # 阶段一（细粒度描述）最多生成 token 数，过小可能信息不足，过大可能冗长且更慢。
        self.stage1_max_new_tokens = kwargs.pop('stage1_max_new_tokens', two_stage_config.get('stage1_max_new_tokens', 512))
        # 阶段一提示词模板：引导模型先做与问题相关的视觉描述，再进入最终答案推理。
        self.stage1_prompt_template = kwargs.pop(
            'stage1_prompt_template',
            two_stage_config.get(
                'stage1_prompt_template',
                (
                    'The question is: "{question}" '
                    'Please first provide a fine-grained description of the regions in the image that are relevant to the question, and do not give the final answer directly. '
                    # 'Please first provide a fine-grained description of the key regions in the image. '
                ),
            ),
        )
        # 阶段一关键词来源：local(本地模型) / api(API模型)。
        self.stage1_keyword_source = str(kwargs.pop(
            'stage1_keyword_source', two_stage_config.get('stage1_keyword_source', 'api')
        )).lower()
        # API 阶段一模型名称与连接参数（仅在 stage1_keyword_source=api 时生效）。
        self.stage1_api_model = kwargs.pop(
            'stage1_api_model', two_stage_config.get('stage1_api_model', 'doubao-seed-2-0-lite-260428')
        )
        self.stage1_api_base_url = kwargs.pop(
            'stage1_api_base_url', two_stage_config.get('stage1_api_base_url', 'https://ark.cn-beijing.volces.com/api/v3')
        )
        self.stage1_api_key_env = kwargs.pop(
            'stage1_api_key_env', two_stage_config.get('stage1_api_key_env', 'ARK_API_KEY')
        )
        self.stage1_api_prompt_template = kwargs.pop(
            'stage1_api_prompt_template',
            two_stage_config.get(
                'stage1_api_prompt_template',
                (
                    'The question is: "{question}". '
                    'Please extract exactly 4 concise visual grounding nouns or noun phrases that are most relevant '
                    'to answering the question from the image. Use nouns only (no verbs or adjectives). '
                    'Do not include any person names or surnames. '
                    'Return only comma-separated nouns.'
                ),
            ),
        )
        # 从阶段一文本中最多提取的关键词数量，用于后续关键词-注意力对齐与聚合。
        self.max_keywords = kwargs.pop('max_keywords', two_stage_config.get('max_keywords', 4))
        #  ----------注意力提取相关----------- 
        self.head_num = kwargs.pop('head_num', two_stage_config.get('head_num', 4))
        self.attention_type = kwargs.pop('attention_type', two_stage_config.get('attention_type', 'rel'))

        # 参与注意力统计的末尾层数，通常越靠后的层语义相关性越强。
        self.attention_last_n_layers = kwargs.pop(
            'attention_last_n_layers', two_stage_config.get('attention_last_n_layers', 8)
        )
        # 注意力阈值分位数（如 0.8 表示保留前 20% 较高注意力区域）用于生成显著区域掩码。
        self.attention_threshold_quantile = kwargs.pop(
            'attention_threshold_quantile', two_stage_config.get('attention_threshold_quantile', 0.8)
        )
        # 背景区域衰减系数：越小表示背景被压暗/抑制得越明显。（旧 mask 模式保留；现默认用 crop）
        self.mask_background_alpha = kwargs.pop(
            'mask_background_alpha', two_stage_config.get('mask_background_alpha', 0.2)
        )
        # 视觉证据裁剪：bbox 外扩比例（相对区域宽高），避免裁得过紧。
        self.crop_padding_ratio = float(kwargs.pop(
            'crop_padding_ratio', two_stage_config.get('crop_padding_ratio', 0.1)
        ))
        # 裁剪边长下限（像素），过小区域会向外扩展到该尺寸。
        self.crop_min_size = int(kwargs.pop(
            'crop_min_size', two_stage_config.get('crop_min_size', 32)
        ))
        # 是否保存注意力调试数据（中间张量/统计结果等），用于排查两阶段行为。
        self.save_attention_debug = kwargs.pop(
            'save_attention_debug', two_stage_config.get('save_attention_debug', False)
        )
        # 注意力调试输出目录；为 None 时通常不额外落盘目录文件。此处还可以保存attention可视化结果
        self.attention_debug_dir = kwargs.pop(
            'attention_debug_dir', two_stage_config.get('attention_debug_dir', './temp_debug')
        )
        # self.attention_debug_dir = kwargs.pop(
        #     'attention_debug_dir', two_stage_config.get('attention_debug_dir', None)
        # )
        # 是否保存全局热力图叠加图，便于直观看到模型关注区域。
        self.save_heatmap_overlay = kwargs.pop(
            'save_heatmap_overlay', two_stage_config.get('save_heatmap_overlay', True)
        )
        # 热力图与原图融合透明度，值越大热力图颜色越明显。
        self.heatmap_overlay_alpha = kwargs.pop(
            'heatmap_overlay_alpha', two_stage_config.get('heatmap_overlay_alpha', 0.45)
        )
        # 多关键词注意力融合方式（如 max），决定最终区域由哪种策略聚合。
        self.keyword_attention_fusion = kwargs.pop(
            'keyword_attention_fusion', two_stage_config.get('keyword_attention_fusion', 'max')
        )
        # 关键词权重幂次：>1 会放大高权重关键词影响，<1 会更平滑。
        self.keyword_attention_weight_power = kwargs.pop(
            'keyword_attention_weight_power', two_stage_config.get('keyword_attention_weight_power', 1.0)
        )
        # 每个关键词取 top-k 注意力位置（0 常表示不做该截断）。
        self.keyword_top_k = kwargs.pop('keyword_top_k', two_stage_config.get('keyword_top_k', 0))
        # 关键词有效注意力下限，低于该值的响应可被过滤掉。
        self.keyword_min_attention = kwargs.pop(
            'keyword_min_attention', two_stage_config.get('keyword_min_attention', 0.0)
        )
        # 是否按关键词分别保存叠加图，便于分析单个关键词的视觉对齐效果。
        self.save_per_keyword_overlay = kwargs.pop(
            'save_per_keyword_overlay', two_stage_config.get('save_per_keyword_overlay', True)
        )
        # 单关键词区域提取的二值化方式（otsu/quantile）：otsu 更适合突出单一显著区域。
        self.keyword_region_binarize = str(kwargs.pop(
            'keyword_region_binarize', two_stage_config.get('keyword_region_binarize', 'otsu')
        )).lower()
        # 单关键词连通域选择准则：
        # - integrated: 区域内热度总和最大（兼顾强且大，默认）
        # - peak: 包含全局峰值点的连通域
        # - area: 面积最大的连通域
        self.keyword_region_criterion = str(kwargs.pop(
            'keyword_region_criterion', two_stage_config.get('keyword_region_criterion', 'integrated')
        )).lower()
        # 是否对选出的单一区域做形态学后处理（孔洞填充 + 轻度闭运算）。
        self.keyword_region_morph = bool(kwargs.pop(
            'keyword_region_morph', two_stage_config.get('keyword_region_morph', True)
        ))
        # 自动选头配置：开启后将执行全头遍历打分与去冗余筛选。
        self.auto_head_mining = kwargs.pop(
            'auto_head_mining', two_stage_config.get('auto_head_mining', True)
        )
        # 头选择模式：
        # - mine: 遍历并自动挖掘相关头
        # - inference: 使用已选头直接推理（不执行挖掘评分）
        self.head_selection_mode = str(kwargs.pop(
            'head_selection_mode', two_stage_config.get('head_selection_mode', 'mine')
        )).lower()
        self.auto_head_top_k = kwargs.pop(
            'auto_head_top_k', two_stage_config.get('auto_head_top_k', self.head_num)
        )
        self.inference_selected_heads = kwargs.pop(
            'inference_selected_heads', two_stage_config.get('inference_selected_heads', [])
        )
        self.inference_selected_heads_by_dataset = kwargs.pop(
            'inference_selected_heads_by_dataset', two_stage_config.get('inference_selected_heads_by_dataset', {})
        )
        self.inference_selected_heads_path = kwargs.pop(
            'inference_selected_heads_path', two_stage_config.get('inference_selected_heads_path', None)
        )
        self.inference_selected_heads_path_by_dataset = kwargs.pop(
            'inference_selected_heads_path_by_dataset',
            two_stage_config.get('inference_selected_heads_path_by_dataset', {}),
        )
        self.head_similarity_threshold = kwargs.pop(
            'head_similarity_threshold', two_stage_config.get('head_similarity_threshold', 0.95)
        )
        self.auto_head_export = kwargs.pop(
            'auto_head_export', two_stage_config.get('auto_head_export', True)
        )
        self.auto_head_stats_dir = kwargs.pop(
            'auto_head_stats_dir',
            two_stage_config.get('auto_head_stats_dir', './temp_debug/head_stats'),
        )
        self.auto_head_metric_weights = kwargs.pop(
            'auto_head_metric_weights',
            two_stage_config.get(
                'auto_head_metric_weights',
                {'iou': 0.45, 'pointing': 0.25, 'auc': 0.20, 'entropy': 0.10},
            ),
        )
        # 开放词汇监督输入：可配置为预计算 JSON，或由外部 detector/segmenter 回调提供。
        self.open_vocab_guidance_path = kwargs.pop(
            'open_vocab_guidance_path', two_stage_config.get('open_vocab_guidance_path', None)
        )
        self.open_vocab_runtime_enable = kwargs.pop(
            'open_vocab_runtime_enable', two_stage_config.get('open_vocab_runtime_enable', True)
        )
        self.open_vocab_detector_backend = kwargs.pop(
            'open_vocab_detector_backend', two_stage_config.get('open_vocab_detector_backend', 'owlv2')
        )
        self.open_vocab_detector_model = kwargs.pop(
            'open_vocab_detector_model',
            two_stage_config.get('open_vocab_detector_model', 'google/owlv2-base-patch16-ensemble'),
        )
        self.open_vocab_detector_threshold = kwargs.pop(
            'open_vocab_detector_threshold', two_stage_config.get('open_vocab_detector_threshold', 0.15)
        )
        self.open_vocab_segmenter_backend = kwargs.pop(
            'open_vocab_segmenter_backend', two_stage_config.get('open_vocab_segmenter_backend', 'sam')
        )
        self.open_vocab_segmenter_model = kwargs.pop(
            'open_vocab_segmenter_model',
            two_stage_config.get('open_vocab_segmenter_model', 'facebook/sam-vit-base'),
        )
        self.use_hybrid_pseudo_gt = kwargs.pop(
            'use_hybrid_pseudo_gt', two_stage_config.get('use_hybrid_pseudo_gt', True)
        )
        self._open_vocab_guidance = {}
        if self.open_vocab_guidance_path and os.path.exists(self.open_vocab_guidance_path):
            try:
                with open(self.open_vocab_guidance_path, 'r', encoding='utf-8') as fin:
                    payload = json.load(fin)
                if isinstance(payload, dict):
                    self._open_vocab_guidance = payload
            except Exception as err:
                logging.warning('failed to load open vocab guidance file %s: %s', self.open_vocab_guidance_path, err)
        self._open_vocab_runtime_cache: dict[str, dict[str, Any]] = {}
        self._open_vocab_detector = None
        self._open_vocab_segmenter = None
        self._open_vocab_runtime_disabled = False
        self.head_mining_profile: dict[str, dict[str, Any]] = {}
        self._loaded_head_profile_scopes: set[str] = set()
        # 缓存最近一次两阶段流程的调试信息，供后续日志/外部调用读取。
        self.last_two_stage_debug: dict[str, Any] = {}

        assert model_path is not None
        
       

        if not os.path.exists(model_path):
            cache_path = get_cache_path(model_path, repo_type='models')
            if cache_path is None:
                snapshot_download(repo_id=model_path)
                cache_path = get_cache_path(model_path, repo_type='models')
            model_path = cache_path

        self.model_path = model_path

        from transformers import AutoProcessor, LlavaForConditionalGeneration

        # LLaVA(HF) 使用统一的 processor + LlavaForConditionalGeneration。
        self.processor = AutoProcessor.from_pretrained(self.model_path)

        # 两阶段注意力挖掘依赖前向注意力，故不启用 vLLM/LMDeploy 加速后端。
        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0
        self.use_vllm = False
        self.use_lmdeploy = False
        self.limit_mm_per_prompt = VLLM_MAX_IMAGE_INPUT_NUM

        # 使用 eager attention 才能通过 output_attentions 取到注意力权重（sdpa 无法导出）。
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype='auto', device_map="auto", attn_implementation="eager"
        )
        self.model.eval()

        # 部分 LLaVA-1.5 权重的 processor 缺少 patch_size / vision_feature_select_strategy /
        # num_additional_image_tokens，会导致扩展 <image> 占位 token 时出现
        # `int // None` 或 “图像 token 数与特征数不匹配（575 vs 576）” 的错误，这里从模型 config 补齐。
        if getattr(self.processor, 'patch_size', None) is None:
            try:
                self.processor.patch_size = self.model.config.vision_config.patch_size
            except Exception as err:
                logging.warning('failed to set processor.patch_size from model config: %s', err)
        if getattr(self.processor, 'vision_feature_select_strategy', None) is None:
            self.processor.vision_feature_select_strategy = getattr(
                self.model.config, 'vision_feature_select_strategy', 'default'
            )
        # CLIP 视觉塔含 1 个 CLS token；配合 default 选择策略需 num_additional_image_tokens=1，
        # 否则图像 token 数会比实际特征数少 1（如 575 vs 576）。
        if getattr(self.processor, 'num_additional_image_tokens', 0) in (None, 0):
            self.processor.num_additional_image_tokens = 1

        torch.cuda.empty_cache()

    # 功能：将统一消息格式转换为 LLaVA(HF) chat 模板可接受的 content 列表，并收集 PIL 图像。
    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None):
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        返回 (content, images)：content 为 chat 模板占位内容，images 为对应的 PIL.Image 列表。
        """
        content = []
        images = []
        for s in inputs:
            if s['type'] == 'image':
                image_path = s['value']
                if image_path.startswith('file://'):
                    image_path = image_path[7:]
                images.append(Image.open(image_path).convert('RGB'))
                content.append({'type': 'image'})
            elif s['type'] == 'text':
                content.append({'type': 'text', 'text': s['value']})
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
        return content, images

    # 功能：将 conversation 转成 prompt 文本。processor 无 chat_template 时回退到 Vicuna 手工拼接。
    def _apply_chat_template(self, conversation) -> str:
        try:
            template = getattr(self.processor, 'chat_template', None)
            tok_template = getattr(getattr(self.processor, 'tokenizer', None), 'chat_template', None)
            if template or tok_template:
                return self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        except Exception as err:
            logging.warning('apply_chat_template failed, fallback to LLaVA-1.5 vicuna prompt: %s', err)
        return self._build_fallback_prompt(conversation)

    # 功能：无 chat_template 时，按 LLaVA-1.5(Vicuna) 对话格式手工拼接 prompt。
    def _build_fallback_prompt(self, conversation) -> str:
        system = self.system_prompt if self.system_prompt else LLAVA_V1_5_SYSTEM_PROMPT
        segments = []
        for turn in conversation:
            role_tag = 'USER' if turn['role'] == 'user' else 'ASSISTANT'
            seg = ''
            for item in turn.get('content', []):
                if item.get('type') == 'image':
                    seg += '<image>\n'
                elif item.get('type') == 'text':
                    seg += item.get('text', '')
            segments.append(f'{role_tag}: {seg}')
        prompt = system + ' ' + ' '.join(segments)
        if conversation and conversation[-1].get('role') == 'user':
            prompt += ' ASSISTANT:'
        return prompt

    # 功能：构造 LLaVA(HF) 推理所需的 conversation、模板文本与张量化输入。
    def _prepare_transformer_inputs(self, message, dataset=None):
        content, images = self._prepare_content(message, dataset=dataset)

        conversation = [{'role': 'user', 'content': content}]
        if self.verbose:
            print(f'\033[31m{conversation}\033[0m')

        text = self._apply_chat_template(conversation)

        target_dtype = next(self.model.parameters()).dtype
        inputs = self.processor(
            images=images if images else None,
            text=text,
            return_tensors='pt',
        ).to(self.model.device, target_dtype)
        return conversation, text, inputs

    # 功能：按配置对模型输出做后处理（如提取最后一个 boxed 内容）。
    def _post_process_response(self, response: str) -> str:
        if not self.post_process:
            return response
        resp = response.split('\\boxed{')[-1]
        lt = len(resp)
        counter, end = 1, None
        for i in range(lt):
            if resp[i] == '{':
                counter += 1
            elif resp[i] == '}':
                counter -= 1
            if counter == 0:
                end = i
                break
            elif i == lt - 1:
                end = lt
                break
        if end is not None:
            return resp[:end]
        return response

    # 功能：将生成 token 解码为文本，并执行统一后处理与可选日志打印。
    def _decode_generated_ids(self, inputs, generated_ids) -> str:
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]
        response = self._post_process_response(response)
        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response

    # 功能：基于输入张量执行一次生成，并返回解码后的文本结果。
    def _generate_with_inputs(self, inputs, generation_kwargs=None) -> str:
        kwargs = dict(self.generate_kwargs)
        if generation_kwargs:
            kwargs.update(generation_kwargs)
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **kwargs)
        return self._decode_generated_ids(inputs, generated_ids)

    # 功能：从消息中提取纯文本问题内容，用于关键词与阶段提示构建。
    def _extract_question_text(self, message) -> str:
        text_chunks = []
        for item in message:
            if item.get('type') == 'text':
                text_chunks.append(item.get('value', ''))
        return ' '.join(chunk for chunk in text_chunks if chunk).strip()

    # 功能：在原始消息末尾追加第一阶段分析指令，构建阶段一输入。
    def _stage1_message(self, message):
        question = self._extract_question_text(message)
        stage1_instruction = self.stage1_prompt_template.format(question=question)
        # stage1_instruction = "Please first provide a fine-grained description of the regions in the image."
        # 仅保留包含图像的消息
        stage1_message = [dict(item) for item in message if item.get('type') == 'image']
        stage1_message.append({'type': 'text', 'value': stage1_instruction})
        return stage1_message

    # 功能：将消息中的图片转换为 API 可接受的 image_url 字段。
    def _to_stage1_api_image_url(self, image_path: str) -> str | None:
        if not image_path:
            return None
        if image_path.startswith(('http://', 'https://')):
            return image_path
        if image_path.startswith('file://'):
            image_path = image_path[7:]
        if not os.path.exists(image_path):
            return None
        try:
            base64_image, mime_type = encode_image(image_path)
            return f'data:{mime_type};base64,{base64_image}'
        except Exception as err:
            logging.warning('failed to encode stage1 api image %s: %s', image_path, err)
            return None

    # 功能：使用 API 模型直接完成阶段一关键词生成。
    def _generate_stage1_with_api(self, message) -> str:
        try:
            ark_module = importlib.import_module('volcenginesdkarkruntime')
            Ark = getattr(ark_module, 'Ark')
        except Exception as err:
            logging.warning('stage1 api mode unavailable, volcenginesdkarkruntime not installed: %s', err)
            return ''

        api_key = os.getenv(self.stage1_api_key_env)
        if not api_key:
            logging.warning('stage1 api mode missing env var %s, fallback to local stage1', self.stage1_api_key_env)
            return ''

        question = self._extract_question_text(message)
        stage1_instruction = self.stage1_api_prompt_template.format(question=question)
        content = []
        for item in message:
            if item.get('type') != 'image':
                continue
            image_url = self._to_stage1_api_image_url(item.get('value', ''))
            if image_url:
                content.append({'type': 'input_image', 'image_url': image_url})

        if not content:
            logging.warning('stage1 api mode found no valid images, fallback to local stage1')
            return ''
        content.append({'type': 'input_text', 'text': stage1_instruction})

        try:
            client = Ark(base_url=self.stage1_api_base_url, api_key=api_key)
            response = client.responses.create(
                model=self.stage1_api_model,
                input=[{'role': 'user', 'content': content}],
            )
        except Exception as err:
            logging.warning('stage1 api call failed, fallback to local stage1: %s', err)
            return ''

        # 兼容不同 SDK 返回结构，优先提取 output_text。
        output_text = getattr(response, 'output_text', None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        try:
            if hasattr(response, 'model_dump'):
                payload = response.model_dump()
            elif isinstance(response, dict):
                payload = response
            else:
                payload = {}
            if isinstance(payload, dict):
                if isinstance(payload.get('output_text'), str) and payload.get('output_text').strip():
                    return payload.get('output_text').strip()
                output = payload.get('output', [])
                chunks = []
                for block in output:
                    if not isinstance(block, dict):
                        continue
                    for part in block.get('content', []):
                        if isinstance(part, dict):
                            text = part.get('text')
                            if isinstance(text, str) and text.strip():
                                chunks.append(text.strip())
                if chunks:
                    return ' '.join(chunks)
        except Exception:
            pass
        return str(response).strip()

    # 功能：从文本中提取并排序关键词，供注意力分析与图像掩膜阶段使用。
    def _extract_keywords(self, text: str) -> list[str]:
        # 升级版：短语优先（位置/颜色），再做词项补充和打分排序。
        candidates = re.findall(r'[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_-]{2,}', text)
        location_terms = [
            'top-left', 'top right', 'bottom-left', 'bottom right', 'left side', 'right side',
            'center', 'middle', 'upper', 'lower', 'foreground', 'background',
            'distant', 'nearby', 'top', 'bottom',
        ]
        color_terms = [
            'red', 'blue', 'green', 'yellow', 'black', 'white', 'gray', 'grey', 'purple', 'orange', 'pink',
        ]
        stopwords = {
            'the', 'a', 'an', 'this', 'that', 'there', 'which', 'about',
            'image', 'picture', 'question', 'answer', 'describe', 'description',
            'relevant', 'region', 'fine-grained', 'final',
        }
        # 明确语义但缺少可定位视觉实体指代的词，避免被当作关键词。
        non_referential_terms = {
            'yes', 'no', 'true', 'false',
            'correct', 'incorrect', 'right', 'wrong',
            'ok', 'okay', 'maybe', 'unknown', 'none', 'null', 'n/a',
        }
        phrase_patterns = [
            r'\b(?:top[-\s]?left|top[-\s]?right|bottom[-\s]?left|bottom[-\s]?right|left side|right side|center|middle|foreground|background|upper|lower|top|bottom)\s+[a-z][a-z0-9_-]{1,}\b',
            r'\b(?:red|blue|green|yellow|black|white|gray|grey|purple|orange|pink)\s+[a-z][a-z0-9_-]{1,}\b',
        ]
        phrase_hits = []
        for pattern in phrase_patterns:
            phrase_hits.extend(re.findall(pattern, text, flags=re.IGNORECASE))
        # 将正则命中的短语做标准化，避免后续打分/去重时出现类型和格式不一致。
        # 兼容 re.findall 在含捕获分组模式下返回 tuple 的情况。
        normalized_phrases = []
        for item in phrase_hits:
            if isinstance(item, tuple):
                # tuple 场景下把各分组拼回完整短语。
                item = ''.join(item)
            # 统一转为字符串并去除首尾空白字符。
            item = str(item).strip()
            if len(item) >= 2:
                # 过滤过短噪声词，只保留有语义价值的候选短语。
                normalized_phrases.append(item)

        # 英文场景下的轻量词性启发：用于在不引入额外依赖的情况下实现“名词优先”。
        english_non_nouns = {
            'is', 'are', 'was', 'were', 'be', 'being', 'been',
            'do', 'does', 'did', 'doing',
            'have', 'has', 'had', 'having',
            'show', 'shows', 'showed', 'showing',
            'look', 'looks', 'looked', 'looking',
            'sit', 'sits', 'sat', 'sitting',
            'stand', 'stands', 'stood', 'standing',
            'walk', 'walks', 'walked', 'walking',
            'run', 'runs', 'ran', 'running',
            'wear', 'wears', 'wore', 'wearing',
            'hold', 'holds', 'held', 'holding',
            'describe', 'describes', 'described', 'describing',
            'final', 'relevant',
        }
        noun_suffixes = ('tion', 'sion', 'ment', 'ness', 'ity', 'ship', 'age', 'ism', 'ist', 'er', 'or')
        non_noun_suffixes = ('ing', 'ed', 'ly', 'able', 'ible', 'ous', 'ive', 'al')

        def _is_noun_like(term: str) -> bool:
            # 多词短语默认更可能是实体描述（如 "red car", "top-left person"），优先视为名词短语。
            normalized = term.strip().lower()
            parts = [p for p in re.split(r'[\s_-]+', normalized) if p]
            if not parts:
                return False
            if len(parts) >= 2:
                head = parts[-1]
                if head not in english_non_nouns and not head.endswith(non_noun_suffixes):
                    return True
                return head.endswith(noun_suffixes)

            token = parts[0]
            # 含中文字符时，缺少词性工具的情况下按“名词候选”处理，避免误伤中文目标词。
            if re.search(r'[\u4e00-\u9fff]', token):
                return True
            if token in english_non_nouns:
                return False
            if token.endswith(noun_suffixes):
                return True
            if token.endswith(non_noun_suffixes):
                return False
            # 默认将未知英文词视为名词候选，以提升目标实体召回。
            return True

        def _is_non_referential(term: str) -> bool:
            normalized = str(term).strip().lower()
            if not normalized:
                return True
            if normalized in non_referential_terms:
                return True
            parts = [p for p in re.split(r'[\s_-]+', normalized) if p]
            if len(parts) == 1 and parts[0] in non_referential_terms:
                return True
            return False

        def _score_term(term: str) -> int:
            # 对候选词/短语进行启发式打分，用于后续关键词排序。
            term_lower = term.lower()
            # 基础分：所有候选至少有 1 分，避免被完全过滤。
            score = 1
            # 名词优先：名词/名词短语显著加权，非名词轻度降权。
            if _is_noun_like(term):
                score += 4
            else:
                score -= 1
            # 命中位置相关词（如 top-left、background）则额外加分。
            if any(loc in term_lower for loc in location_terms):
                score += 2
            # 命中颜色词（如 red、blue）则额外加分。
            if any(color in term_lower for color in color_terms):
                score += 2
            # 词更长通常信息量更高，因此长度 >=4 再加一分。
            if len(term) >= 4:
                score += 1
            return score

        # key(统一小写词项) -> 累计分数，用于最终排序。
        score_map: dict[str, int] = {}
        # key -> 首次出现顺序，用于同分时保持稳定排序。
        order_map: dict[str, int] = {}
        # key -> 原始词面形式（保留大小写/连字符），用于最终输出。
        value_map: dict[str, str] = {}
        index = 0
        # 先处理短语命中：短语通常语义更完整，因此后续会给予额外权重。
        for phrase in normalized_phrases:
            key = phrase.lower()
            if key in stopwords:
                continue
            if _is_non_referential(phrase):
                continue
            # 短语分 = 基础词项分 + 额外短语加权(+3)。
            score_map[key] = score_map.get(key, 0) + _score_term(phrase) + 3
            if key not in order_map:
                order_map[key] = index
                value_map[key] = phrase
                index += 1
        # 再处理普通候选词：用于补充短语之外的关键信息。
        for word in candidates:
            key = word.lower()
            if key in stopwords:
                continue
            if _is_non_referential(word):
                continue
            # 单词分仅使用 _score_term，不含短语额外加权。
            score_map[key] = score_map.get(key, 0) + _score_term(word)
            if key not in order_map:
                order_map[key] = index
                value_map[key] = word
                index += 1

        # 排序优先级：分数降序 -> 词长降序 -> 首次出现顺序升序。
        sorted_keys = sorted(
            score_map.keys(),
            key=lambda k: (0 if _is_noun_like(value_map[k]) else 1, -score_map[k], -len(k), order_map[k]),
        )
        # 仅保留前 max_keywords 个关键词作为最终输出。
        keywords = [value_map[k] for k in sorted_keys[:self.max_keywords]]
        return keywords

    # 功能：在 token 序列中查找目标子序列出现位置并返回对应索引。
    def _find_subsequence_positions(self, source_ids: list[int], target_ids: list[int]) -> list[int]:
        if not target_ids or len(target_ids) > len(source_ids):
            return []
        positions = []
        target_len = len(target_ids)
        for start in range(len(source_ids) - target_len + 1):
            if source_ids[start:start + target_len] == target_ids:
                positions.extend(range(start, start + target_len))
        return positions

    # 功能：收集输入序列中视觉相关特殊 token 的位置索引。
    def _collect_visual_token_positions(self, input_ids: list[int]) -> list[int]:
        # LLaVA(HF) 的图像占位 token（默认 32000），优先读取模型配置中的 image_token_index/id。
        special_ids = set()
        for attr in ['image_token_index', 'image_token_id']:
            value = getattr(self.model.config, attr, None)
            if isinstance(value, int) and value >= 0:
                special_ids.add(value)
        if not special_ids:
            special_ids.add(32000)

        return [idx for idx, token_id in enumerate(input_ids) if token_id in special_ids]

    # 功能：聚合关键词对视觉 token 的注意力，得到一维热度向量。
    def _aggregate_attention_heatmap(self, attentions, keyword_positions, visual_positions):
        # 若注意力张量、关键词位置或视觉 token 位置任一为空，则无法构建热力图。
        if not attentions or not keyword_positions or not visual_positions:
            return None
        # 仅使用最后 N 层注意力，通常后层更偏向语义对齐信息。
        total_layers = len(attentions)
        start = max(0, total_layers - int(self.attention_last_n_layers))
        selected_layers = list(range(start, total_layers))
        # 与注意力同设备创建索引，避免跨设备索引报错。
        device = attentions[selected_layers[0]].device
        visual_index = torch.tensor(visual_positions, device=device, dtype=torch.long)
        # heat 对应每个视觉 token 的聚合响应值。
        heat = torch.zeros(len(visual_positions), device=device)

        for layer_idx in selected_layers:
            layer = attentions[layer_idx][0]  # [heads, seq, seq]
            # 关键词 token 索引。
            kw_idx = torch.tensor(keyword_positions, device=device, dtype=torch.long)
            # 先对 head 和关键词维度求均值，得到“关键词对全序列”的平均关注强度。
            kw_to_all = layer[:, kw_idx, :].mean(dim=(0, 1))  # [seq]
            # 将全序列注意力裁剪到视觉 token 子集，并与历史层结果做逐点最大融合。
            heat = torch.maximum(heat, kw_to_all.index_select(0, visual_index))

        # 对跨层聚合结果做平均，控制层数变化带来的尺度偏移。
        if len(selected_layers) > 0:
            heat = heat / len(selected_layers)
        # 返回 CPU numpy，便于后续可视化和非 torch 流程处理。
        return heat.detach().float().cpu().numpy()

    # 功能：将一维视觉热度重排并归一化为二维热力图。
    def _reshape_heatmap(self, token_heat: np.ndarray, inputs) -> np.ndarray:
        # LLaVA(HF) 视觉 token 数固定为 24x24（336/14），直接按方阵重排；
        # 若长度不符则退化为近似方阵，保证仍可 reshape 与可视化。
        side = max(1, int(round(math.sqrt(max(1, token_heat.shape[0])))))
        needed = side * side
        data = token_heat[:needed]
        if data.shape[0] < needed:
            data = np.pad(data, (0, needed - data.shape[0]), mode='constant')
        heatmap = data.reshape(side, side)
        # Min-Max 归一化到 [0, 1]，便于后续阈值处理和叠加显示。
        heatmap = heatmap - heatmap.min()
        denom = heatmap.max() + 1e-8
        return heatmap / denom

    # 功能：从消息中提取首张图片路径，兼容 file:// 前缀。
    def _extract_primary_image_path(self, message):
        for item in message:
            if item.get('type') == 'image':
                image_path = item.get('value')
                if image_path.startswith('file://'):
                    return image_path[7:]
                return image_path
        return None

    # 功能：根据融合热力图阈值区域做 fallback crop，并返回调试信息。
    def _build_masked_image(self, image_path: str, heatmap: np.ndarray):
        from PIL import Image

        # 输入图路径无效时直接返回，避免后续 I/O 异常。
        if image_path is None or not os.path.exists(image_path):
            return None, {}
        # 读取原图；热力图缩放到原图尺寸后按分位数阈值得到前景区域。
        image = Image.open(image_path).convert('RGB')
        img_arr = np.array(image).astype(np.float32)
        width, height = image.size
        resized_heat = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
        heat = np.array(resized_heat).astype(np.float32) / 255.0

        threshold = float(np.quantile(heat, self.attention_threshold_quantile))
        binary = heat >= threshold
        if not binary.any():
            return None, {}

        bbox = self._bbox_from_mask(binary, width, height)
        if bbox is None:
            return None, {}
        crop_path, debug = self._save_image_crop(
            image_path, bbox, stem_suffix='fallback_crop'
        )
        if crop_path is None:
            return None, {}

        # 可选保存热力图叠加图，用于人工检查注意力是否对齐目标区域。
        overlay_path = None
        output_dir = self.attention_debug_dir
        if self.save_heatmap_overlay:
            overlay = self._build_heatmap_overlay(img_arr, heat)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                image_stem = os.path.splitext(os.path.basename(image_path))[0] or 'image'
                overlay_path = os.path.join(output_dir, f'{image_stem}_heatmap_overlay.png')
                Image.fromarray(overlay).save(overlay_path)
            else:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as fout:
                    Image.fromarray(overlay).save(fout.name)
                    overlay_path = fout.name

        debug.update({
            'mask_threshold': threshold,
            'mask_foreground_ratio': float(binary.mean()),
            'crop_style': 'combined_fallback',
        })
        if overlay_path is not None:
            debug['heatmap_overlay_path'] = overlay_path
        if self.save_attention_debug:
            debug_dir = self.attention_debug_dir or tempfile.mkdtemp(prefix='llava_attn_debug_')
            os.makedirs(debug_dir, exist_ok=True)
            heatmap_path = os.path.join(debug_dir, 'heatmap.npy')
            np.save(heatmap_path, heat)
            debug['heatmap_path'] = heatmap_path
        return crop_path, debug

    # 功能：从单个关键词热力图中仅提取“最强且面积最大”的单一连通区域，去除其余噪声。
    def _extract_dominant_region_mask(self, heat_2d: np.ndarray) -> np.ndarray:
        import cv2
        from scipy.ndimage import binary_fill_holes

        # 输入非法时返回空掩码，避免后续索引异常。
        if heat_2d is None or heat_2d.size == 0:
            return np.zeros((1, 1), dtype=bool)
        heat = np.asarray(heat_2d, dtype=np.float32)
        if heat.ndim != 2:
            heat = heat.reshape(heat.shape[0], -1)
        # Min-Max 归一化到 [0, 1]，统一二值化尺度。
        heat = heat - float(heat.min())
        denom = float(heat.max()) + 1e-8
        heat = heat / denom

        # 二值化：默认 Otsu 自动阈值，更利于突出单一显著区域。
        if self.keyword_region_binarize == 'quantile':
            thr = float(np.quantile(heat, self.attention_threshold_quantile))
            binary = (heat >= thr).astype(np.uint8)
        else:
            binary = auto_otsu(heat).astype(np.uint8)

        # 无前景像素时直接返回全零掩码。
        if binary.sum() <= 0:
            return np.zeros_like(binary, dtype=bool)

        # 连通域分析：label 0 为背景，1..num-1 为前景连通域。
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return np.zeros_like(binary, dtype=bool)

        criterion = self.keyword_region_criterion
        best_label = -1
        if criterion == 'peak':
            # 选取包含全局峰值点的连通域。
            peak_idx = int(np.argmax(heat))
            py, px = np.unravel_index(peak_idx, heat.shape)
            peak_label = int(labels[py, px])
            best_label = peak_label if peak_label > 0 else -1

        if best_label <= 0:
            best_score = -1.0
            for label in range(1, num_labels):
                comp = labels == label
                if criterion == 'area':
                    score = float(stats[label, cv2.CC_STAT_AREA])
                else:
                    # integrated（默认）：区域内热度总和，兼顾强度与面积。
                    score = float(heat[comp].sum())
                if score > best_score:
                    best_score = score
                    best_label = label

        if best_label <= 0:
            return np.zeros_like(binary, dtype=bool)

        mask = labels == best_label

        # 形态学后处理：填补内部孔洞 + 轻度闭运算平滑边缘。
        if self.keyword_region_morph:
            mask = binary_fill_holes(mask)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            mask = closed.astype(bool)

        return mask.astype(bool)

    # 功能：逐关键词提取单一显著区域并按并集融合，得到最终针对性掩码（原图分辨率）。
    def _build_combined_keyword_mask(self, image_path: str, keyword_items: list[dict[str, Any]]):
        from PIL import Image

        if image_path is None or not os.path.exists(image_path):
            return None, {}
        if not keyword_items:
            return None, {}

        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        combined = np.zeros((height, width), dtype=bool)
        per_keyword_debug = []
        per_keyword_masks = []

        for item in keyword_items:
            heatmap = item.get('heatmap', None)
            if heatmap is None or np.asarray(heatmap).size == 0:
                continue
            # 与 _build_masked_image 保持一致：先缩放到原图尺寸再做区域提取。
            heat_uint8 = (np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0) * 255).astype(np.uint8)
            resized = Image.fromarray(heat_uint8).resize((width, height), Image.BILINEAR)
            heat = np.array(resized).astype(np.float32) / 255.0
            region = self._extract_dominant_region_mask(heat)
            keyword = item.get('keyword', '')
            if region.shape != (height, width) or not region.any():
                per_keyword_debug.append({
                    'keyword': keyword,
                    'region_area_ratio': 0.0,
                })
                continue
            combined |= region
            per_keyword_masks.append({
                'keyword': keyword,
                'mask': region,
            })
            per_keyword_debug.append({
                'keyword': keyword,
                'region_area_ratio': float(region.mean()),
            })

        debug = {
            'combined_foreground_ratio': float(combined.mean()) if combined.size else 0.0,
            'per_keyword_region': per_keyword_debug,
            # 仅供后续 crop 使用，写入 last_two_stage_debug 前应弹出，避免大数组落盘。
            'per_keyword_masks': per_keyword_masks,
        }
        return combined, debug

    # 功能：由布尔掩码计算带 padding / 最小边长约束的裁剪框 (x1, y1, x2, y2)。
    def _bbox_from_mask(self, mask_bool: np.ndarray, width: int, height: int):
        binary = np.asarray(mask_bool).astype(bool)
        if binary.ndim != 2 or not binary.any():
            return None
        ys, xs = np.where(binary)
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_ratio = max(0.0, float(self.crop_padding_ratio))
        pad_x = max(1, int(round(bw * pad_ratio))) if pad_ratio > 0 else 0
        pad_y = max(1, int(round(bh * pad_ratio))) if pad_ratio > 0 else 0
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)

        min_size = max(1, int(self.crop_min_size))
        cur_w, cur_h = x2 - x1, y2 - y1
        if cur_w < min_size:
            expand = min_size - cur_w
            x1 = max(0, x1 - expand // 2)
            x2 = min(width, x1 + min_size)
            x1 = max(0, x2 - min_size)
        if cur_h < min_size:
            expand = min_size - cur_h
            y1 = max(0, y1 - expand // 2)
            y2 = min(height, y1 + min_size)
            y1 = max(0, y2 - min_size)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    # 功能：按 bbox 裁剪原图并保存，返回路径与调试信息。
    def _save_image_crop(
        self,
        image_path: str,
        bbox: tuple[int, int, int, int],
        stem_suffix: str = 'crop',
    ):
        from PIL import Image

        if image_path is None or not os.path.exists(image_path):
            return None, {}
        if bbox is None:
            return None, {}

        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None, {}

        cropped = image.crop((x1, y1, x2, y2))
        output_dir = self.attention_debug_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            image_stem = os.path.splitext(os.path.basename(image_path))[0] or 'image'
            crop_path = os.path.join(output_dir, f'{image_stem}_{stem_suffix}.png')
            cropped.save(crop_path)
        else:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as fout:
                cropped.save(fout.name)
                crop_path = fout.name

        debug = {
            'crop_bbox': [x1, y1, x2, y2],
            'crop_size': [x2 - x1, y2 - y1],
            'crop_image_path': crop_path,
            'masked_image_path': crop_path,  # 兼容旧调试字段名
            'visual_prompt_mode': 'crop',
        }
        return crop_path, debug

    # 功能：基于布尔掩码做 Combined crop（并集区域外接框裁剪）。
    def _build_masked_image_from_mask(self, image_path: str, mask_bool: np.ndarray):
        from PIL import Image

        if image_path is None or not os.path.exists(image_path):
            return None, {}
        if mask_bool is None or not np.asarray(mask_bool).any():
            return None, {}

        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        binary = np.asarray(mask_bool).astype(bool)
        if binary.shape != (height, width):
            resized_mask = Image.fromarray((binary * 255).astype(np.uint8)).resize(
                (width, height), Image.NEAREST
            )
            binary = np.array(resized_mask) > 0

        bbox = self._bbox_from_mask(binary, width, height)
        if bbox is None:
            return None, {}

        crop_path, debug = self._save_image_crop(
            image_path, bbox, stem_suffix='combined_crop'
        )
        if crop_path is None:
            return None, {}

        output_dir = self.attention_debug_dir
        mask_png_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            image_stem = os.path.splitext(os.path.basename(image_path))[0] or 'image'
            mask_png_path = os.path.join(output_dir, f'{image_stem}_combined_mask.png')
            Image.fromarray((binary * 255).astype(np.uint8)).save(mask_png_path)

        debug.update({
            'mask_foreground_ratio': float(binary.mean()),
            'mask_source': 'per_keyword_dominant_region_crop',
            'crop_style': 'combined',
        })
        if mask_png_path is not None:
            debug['combined_mask_path'] = mask_png_path
        return crop_path, debug

    # 功能：将热力图伪彩色叠加到原图数组上，生成可视化图像。
    def _build_heatmap_overlay(self, img_arr: np.ndarray, heat: np.ndarray) -> np.ndarray:
        alpha = float(max(0.0, min(1.0, self.heatmap_overlay_alpha)))
        heat = np.clip(heat.astype(np.float32), 0.0, 1.0)
        # 轻量伪彩色：低响应偏蓝，高响应偏红。
        red = heat * 255.0
        green = np.sqrt(heat) * 180.0
        blue = (1.0 - heat) * 220.0
        color = np.stack([red, green, blue], axis=-1)
        overlay = img_arr * (1.0 - alpha) + color * alpha
        return np.clip(overlay, 0, 255).astype(np.uint8)

    # 功能：将关键词转换为适合文件名的稳定 tag。
    def _keyword_file_tag(self, keyword: str, index: int = 0) -> str:
        raw = str(keyword or '').strip().lower()
        safe = re.sub(r'[^0-9a-zA-Z_-]+', '_', raw).strip('_')
        if not safe:
            safe = f'kw_{int(index):02d}'
        digest = hashlib.md5(raw.encode('utf-8')).hexdigest()[:8]
        return f'{int(index):02d}_{safe}_{digest}'

    # 功能：基于热力图生成并保存叠加图，返回输出路径。
    def _save_overlay_from_heatmap(
        self,
        image_path: str,
        heatmap: np.ndarray,
        output_path: str | None = None,
    ) -> str | None:
        from PIL import Image

        # 原图路径无效时直接返回，避免图像读取异常。
        if image_path is None or not os.path.exists(image_path):
            return None
        # 读取原图并转换为数组，后续用于与热力图做可视化叠加。
        image = Image.open(image_path).convert('RGB')
        img_arr = np.array(image).astype(np.float32)
        # 将热力图缩放到原图尺寸，确保逐像素叠加时空间对齐。
        resized_heat = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
        heat = np.array(resized_heat).astype(np.float32) / 255.0
        # 构建伪彩色叠加图并保存到临时 PNG。
        overlay = self._build_heatmap_overlay(img_arr, heat)
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            Image.fromarray(overlay).save(output_path)
            return output_path
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as fout:
            Image.fromarray(overlay).save(fout.name)
            return fout.name

    # 功能：将热力图数组保存为灰度 PNG，便于逐关键词直接查看。
    def _save_heatmap_png(self, heatmap: np.ndarray, output_path: str | None = None) -> str | None:
        from PIL import Image

        heat = np.clip(heatmap.astype(np.float32), 0.0, 1.0)
        heat_png = (heat * 255).astype(np.uint8)
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            Image.fromarray(heat_png).save(output_path)
            return output_path
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as fout:
            Image.fromarray(heat_png).save(fout.name)
            return fout.name

    # 功能：保存单关键词伪监督信息（检测框/掩码）及其可视化叠加图。
    def _save_keyword_pseudo_gt_artifacts(
        self,
        image_path: str,
        pseudo_gt: dict[str, Any],
        info_npy_path: str | None = None,
        mask_npy_path: str | None = None,
        mask_png_path: str | None = None,
        overlay_png_path: str | None = None,
    ) -> dict[str, str]:
        from PIL import Image, ImageDraw

        if not isinstance(pseudo_gt, dict) or not pseudo_gt:
            return {}

        saved: dict[str, str] = {}
        payload = dict(pseudo_gt)

        # 统一检测框/分数字段格式，便于离线分析和可视化复用。
        detection_boxes = payload.get('detection_boxes', [])
        if detection_boxes is None:
            detection_boxes = []
        detection_boxes = [
            [int(round(v)) for v in box]
            for box in detection_boxes
            if isinstance(box, (list, tuple, np.ndarray)) and len(box) == 4
        ]
        payload['detection_boxes'] = detection_boxes
        payload['detection_scores'] = [
            float(s) for s in payload.get('detection_scores', []) or []
        ]
        if payload.get('bbox_gt', None) is not None:
            payload['bbox_gt'] = [int(round(v)) for v in payload.get('bbox_gt', [])[:4]]
        if payload.get('q_conf', None) is not None:
            payload['q_conf'] = float(payload.get('q_conf'))
        if payload.get('segmentation_score', None) is not None:
            payload['segmentation_score'] = float(payload.get('segmentation_score'))

        mask_gt = payload.get('mask_gt', None)
        mask_arr = None
        if mask_gt is not None:
            mask_arr = (np.asarray(mask_gt) > 0).astype(np.uint8)
            payload['mask_gt'] = mask_arr
        det_mask = payload.get('det_mask', None)
        if det_mask is not None:
            payload['det_mask'] = (np.asarray(det_mask) > 0).astype(np.uint8)
        seg_mask = payload.get('seg_mask', None)
        if seg_mask is not None:
            payload['seg_mask'] = (np.asarray(seg_mask) > 0).astype(np.uint8)

        if info_npy_path:
            os.makedirs(os.path.dirname(info_npy_path), exist_ok=True)
            np.save(info_npy_path, payload, allow_pickle=True)
            saved['pseudo_gt_npy_path'] = info_npy_path

        if mask_arr is not None and mask_arr.ndim == 2:
            if mask_npy_path:
                os.makedirs(os.path.dirname(mask_npy_path), exist_ok=True)
                np.save(mask_npy_path, mask_arr.astype(np.uint8))
                saved['mask_npy_path'] = mask_npy_path
            if mask_png_path:
                os.makedirs(os.path.dirname(mask_png_path), exist_ok=True)
                Image.fromarray((mask_arr * 255).astype(np.uint8)).save(mask_png_path)
                saved['mask_png_path'] = mask_png_path

        if overlay_png_path and image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            img_arr = np.array(image).astype(np.float32)
            h, w = img_arr.shape[:2]

            if mask_arr is not None and mask_arr.ndim == 2:
                if mask_arr.shape != (h, w):
                    resized_mask = Image.fromarray((mask_arr * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
                    mask_arr = (np.array(resized_mask) > 0).astype(np.uint8)
                mask_bool = mask_arr.astype(bool)
                if mask_bool.any():
                    # 用红色半透明覆盖目标掩码区域，突出伪监督前景。
                    tint = np.array([255.0, 64.0, 64.0], dtype=np.float32)
                    alpha = 0.30
                    img_arr[mask_bool] = img_arr[mask_bool] * (1.0 - alpha) + tint * alpha

            overlay_img = Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))
            draw = ImageDraw.Draw(overlay_img)

            # 先画候选检测框（橙色），再画最终框（绿色）。
            for box in detection_boxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                x1, x2 = sorted((max(0, min(w - 1, x1)), max(0, min(w - 1, x2))))
                y1, y2 = sorted((max(0, min(h - 1, y1)), max(0, min(h - 1, y2))))
                if x2 > x1 and y2 > y1:
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 170, 0), width=2)
            bbox_gt = payload.get('bbox_gt', None)
            if isinstance(bbox_gt, (list, tuple)) and len(bbox_gt) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox_gt]
                x1, x2 = sorted((max(0, min(w - 1, x1)), max(0, min(w - 1, x2))))
                y1, y2 = sorted((max(0, min(h - 1, y1)), max(0, min(h - 1, y2))))
                if x2 > x1 and y2 > y1:
                    draw.rectangle([x1, y1, x2, y2], outline=(60, 240, 60), width=3)

            os.makedirs(os.path.dirname(overlay_png_path), exist_ok=True)
            overlay_img.save(overlay_png_path)
            saved['pseudo_overlay_path'] = overlay_png_path

        return saved

    # 功能：按设定策略（max/mean/weighted）融合多个关键词热力图。
    def _fuse_keyword_heatmaps(self, heatmaps: list[np.ndarray], scores: list[float]) -> np.ndarray | None:
        # 没有可融合热图时返回 None。
        if not heatmaps:
            return None
        # 堆叠为 [K, H, W]，K 为关键词数量。
        stack = np.stack(heatmaps, axis=0).astype(np.float32)
        mode = str(self.keyword_attention_fusion).lower()
        if mode == 'mean':
            # 平均融合：每个关键词贡献相同。
            fused = stack.mean(axis=0)
        elif mode == 'weighted':
            # 加权融合：按关键词得分分配权重，并可通过幂次调节权重尖锐程度。
            weights = np.array(scores, dtype=np.float32)
            weights = np.maximum(weights, 1e-8) ** float(self.keyword_attention_weight_power)
            weight_sum = float(weights.sum())
            if weight_sum <= 0:
                # 权重异常退化为均值融合，避免除零。
                fused = stack.mean(axis=0)
            else:
                fused = (stack * weights[:, None, None]).sum(axis=0) / weight_sum
        else:
            # 默认 max 融合：保留任一关键词的强响应区域。
            fused = stack.max(axis=0)
        # 统一归一化到 [0, 1]，方便后续阈值/可视化处理。
        fused = fused - fused.min()
        denom = fused.max() + 1e-8
        return fused / denom

    # 功能：按注意力阈值与 top-k 规则筛选用于融合的关键词条目。
    def _select_keyword_items_for_fusion(self, items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        # 空输入时返回空保留集与空丢弃集。
        if not items:
            return [], []
        # 先按最小平均注意力阈值过滤。去掉为0的
        min_score = float(max(0.0, self.keyword_min_attention))
        # 注意：item 内包含 numpy.ndarray（如 heatmap）时，不能用 `item in/not in`
        # 触发字典相等比较，否则会走到 ndarray 的逐元素比较并抛出布尔歧义错误。
        kept = [item for item in items if float(item.get('mean_attention', 0.0)) >= min_score]
        dropped = [item for item in items if float(item.get('mean_attention', 0.0)) < min_score]
        if not kept:
            return [], dropped

        # 按平均注意力降序排序，必要时截断 top-k。
        kept = sorted(kept, key=lambda x: float(x.get('mean_attention', 0.0)), reverse=True)
        top_k = int(self.keyword_top_k)
        if top_k > 0 and len(kept) > top_k:
            dropped.extend(kept[top_k:])
            kept = kept[:top_k]
        return kept, dropped

    # 功能：替换消息中的首张图片为新路径，保持其余内容不变。
    def  _replace_primary_image(self, message, new_image_path: str):
        # 仅替换第一张图片，避免多图场景下误改全部图像输入。
        replaced = []
        image_replaced = False
        for item in message:
            copied = dict(item)
            if not image_replaced and copied.get('type') == 'image':
                copied['value'] = new_image_path
                image_replaced = True
            replaced.append(copied)
        return replaced

    # 功能：在原来图像的基础上添加 crop 视觉提示，保持其余内容不变。
    # 示例：原消息为 [原图, 问题文本]，调用后变为 [原图, crop图..., 问题文本]
    def  _add_masked_image(self, message, new_image_path):
        image_paths = new_image_path if isinstance(new_image_path, (list, tuple)) else [new_image_path]
        image_paths = [p for p in image_paths if p]
        replaced = []
        prompt_inserted = False
        for item in message:
            copied = dict(item)
            replaced.append(copied)
            if not prompt_inserted and copied.get('type') == 'image':
                for path in image_paths:
                    replaced.append({'type': 'image', 'value': path})
                prompt_inserted = True
        return replaced


    # 功能：在原来图像的基础上添加 crop 视觉提示和关键词，保持其余内容不变。
    # 示例：原消息为 [原图, 问题文本]，调用后变为 [原图, crop图..., 关键词, 问题文本]
    def  _add_masked_image_and_keywords(self, message, new_image_path, keywords: list[str]):
        image_paths = new_image_path if isinstance(new_image_path, (list, tuple)) else [new_image_path]
        image_paths = [p for p in image_paths if p]
        replaced = []
        prompt_inserted = False
        for item in message:
            copied = dict(item)
            replaced.append(copied)
            if not prompt_inserted and copied.get('type') == 'image':
                for path in image_paths:
                    replaced.append({'type': 'image', 'value': path})
                replaced.append({
                    'type': 'text',
                    'value': (
                        f"Keywords: {', '.join(keywords)}. "
                        "Please analyze based on the image and the question."
                    ),
                })
                prompt_inserted = True
        return replaced

    def _model_signature(self) -> str:
        model_name = os.path.basename(str(self.model_path)).lower()
        if '13b' in model_name:
            return 'llava_1_5_13b'
        return 'llava_1_5_7b'

    def _dataset_scope_tag(self, dataset: str | None) -> str:
        value = str(dataset).strip() if dataset is not None else ''
        if not value:
            return 'global'
        slug = ''.join(c if (c.isalnum() or c in {'-', '_'}) else '_' for c in value.lower())
        slug = '_'.join(part for part in slug.split('_') if part)
        return slug or 'global'

    def _dataset_key_candidates(self, dataset: str | None) -> list[str]:
        value = str(dataset).strip() if dataset is not None else ''
        if not value:
            return []
        lower = value.lower()
        candidates = [value]
        if lower != value:
            candidates.append(lower)
        tag = self._dataset_scope_tag(dataset)
        if tag not in candidates:
            candidates.append(tag)
        return candidates

    def _profile_scope_name(self, dataset: str | None) -> str:
        # 以「模型 + 数据集」作为画像作用域，避免不同数据集相互污染。
        return f"{self._model_signature()}__{self._dataset_scope_tag(dataset)}"

    def _load_head_profile_from_disk(self, scope_name: str):
        """首次导出前，从历史 metrics 文件恢复累计统计，避免跨次运行丢失。"""
        if scope_name in self._loaded_head_profile_scopes:
            return
        self._loaded_head_profile_scopes.add(scope_name)
        metrics_path = os.path.join(self.auto_head_stats_dir, f'{scope_name}_head_metrics.csv')
        if not os.path.exists(metrics_path):
            return
        try:
            model_profile = self.head_mining_profile.setdefault(scope_name, {})
            with open(metrics_path, 'r', encoding='utf-8', newline='') as fin:
                reader = csv.DictReader(fin)
                for row in reader:
                    head_key = str(row.get('head', '')).strip()
                    if not head_key:
                        continue
                    seen = max(0, int(float(row.get('seen', 0) or 0)))
                    if seen <= 0:
                        continue
                    selected = max(0, int(float(row.get('selected', 0) or 0)))
                    mean_score = float(row.get('mean_score', 0.0) or 0.0)
                    std_score = max(0.0, float(row.get('std_score', 0.0) or 0.0))
                    score_sum = mean_score * seen
                    score_sq_sum = (std_score * std_score + mean_score * mean_score) * seen
                    slot = model_profile.setdefault(
                        head_key,
                        {'seen': 0, 'selected': 0, 'score_sum': 0.0, 'score_sq_sum': 0.0},
                    )
                    slot['seen'] += seen
                    slot['selected'] += selected
                    slot['score_sum'] += score_sum
                    slot['score_sq_sum'] += score_sq_sum
        except Exception as err:
            logging.warning('failed to load historical head profile from %s: %s', metrics_path, err)
    
    # 根据模型型号的不同，设置注意力层和头的数量，并且设置默认的特征注意力头
    def _resolve_attention_scan_config(self) -> dict[str, Any]:
        signature = self._model_signature()
        # LLaVA-1.5 的语言塔基于 Vicuna：7B=32层/32头，13B=40层/40头。
        # 默认头取自 llava_methods.enhanced_rel_attention_llava 的经验头（layer14 的 13/24/26），
        # 用下划线格式以匹配 map_func 返回的 f"{LAYER}_{HEAD}" 键。
        if signature == 'llava_1_5_13b':
            return {'layers': 40, 'heads': 40, 'default_items': ["14_13", "14_24", "14_26"]}
        if signature == 'llava_1_5_7b':
            return {'layers': 32, 'heads': 32, 'default_items': ["14_13", "14_24", "14_26"]}
        raise ValueError(
            f'Unsupported model for attention scan config: signature={signature!r}, '
            f'model_path={self.model_path!r}, model_type={self.model_type!r}. '
            f'Expected one of: llava_1_5_7b, llava_1_5_13b.'
        )
    # 选择不同的注意力提取机制
    def _resolve_attention_map_func(self):
        if self.attention_type == "orin":
            return llava_methods.auto_param_orin_attention_llava
        return llava_methods.auto_param_rel_attention_llava

    # 标准化头键列表，确保每个头键都包含下划线
    def _normalize_head_keys(self, head_items: Any) -> list[str]:
        if head_items is None:
            return []
        if isinstance(head_items, str):
            value = head_items.strip()
            return [value] if "_" in value else []
        if isinstance(head_items, list):
            keys = []
            for item in head_items:
                if isinstance(item, str) and "_" in item:
                    keys.append(item.strip())
                elif isinstance(item, dict):
                    key = item.get('head')
                    if isinstance(key, str) and "_" in key:
                        keys.append(key.strip())
            return keys
        return []

    # ! 注意力头选择参数设置的详细说明
    def _load_inference_selected_heads(self, scan_cfg: dict[str, Any], dataset: str | None = None) -> list[str]:
        # 优先使用显式传入列表
        direct = self._normalize_head_keys(self.inference_selected_heads)
        if direct:
            return direct[:max(1, int(self.head_num))]

        # 支持按数据集给定的头列表（用于不同评测集加载不同 top-head）。
        if isinstance(self.inference_selected_heads, dict):
            for key in self._dataset_key_candidates(dataset):
                from_map = self._normalize_head_keys(self.inference_selected_heads.get(key))
                if from_map:
                    return from_map[:max(1, int(self.head_num))]
        if isinstance(self.inference_selected_heads_by_dataset, dict):
            for key in self._dataset_key_candidates(dataset):
                from_map = self._normalize_head_keys(self.inference_selected_heads_by_dataset.get(key))
                if from_map:
                    return from_map[:max(1, int(self.head_num))]

        # 次优先使用文件（通常是 *_top_heads.json），先尝试数据集专属，再尝试显式全局路径。
        candidate_paths = []
        if isinstance(self.inference_selected_heads_path_by_dataset, dict):
            for key in self._dataset_key_candidates(dataset):
                path_value = self.inference_selected_heads_path_by_dataset.get(key)
                if isinstance(path_value, str) and path_value.strip():
                    candidate_paths.append(path_value.strip())
        scope_name = self._profile_scope_name(dataset)
        candidate_paths.append(os.path.join(self.auto_head_stats_dir, f'{scope_name}_top_heads.json'))
        if isinstance(self.inference_selected_heads_path, str) and '{dataset}' in self.inference_selected_heads_path:
            dataset_tag = self._dataset_scope_tag(dataset)
            candidate_paths.append(self.inference_selected_heads_path.format(dataset=dataset_tag))
        path = self.inference_selected_heads_path
        if isinstance(path, str) and path.strip():
            candidate_paths.append(path.strip())
        # 最后兼容历史「仅按模型」命名导出。
        candidate_paths.append(os.path.join(self.auto_head_stats_dir, f'{self._model_signature()}_top_heads.json'))
        seen = set()
        for path in candidate_paths:
            if not path or path in seen:
                continue
            seen.add(path)
            if not os.path.exists(path):
                continue
            try:
                with open(path, 'r', encoding='utf-8') as fin:
                    payload = json.load(fin)
                from_file = self._normalize_head_keys(payload)
                if from_file:
                    return from_file[:max(1, int(self.head_num))]
            except Exception as err:
                logging.warning('failed to load inference_selected_heads_path %s: %s', path, err)

        # 最终回退到默认头模板
        return scan_cfg['default_items'][:max(1, int(self.head_num))]

    # 懒加载开放词汇检测/分割运行时；成功返回 True，失败则仅回退到预计算的 guidance 文件。
    def _ensure_open_vocab_runtime(self) -> bool:
        # 未开启运行时，或此前初始化失败已被永久禁用
        if not self.open_vocab_runtime_enable or self._open_vocab_runtime_disabled:
            return False
        # 检测器或分割器任一已加载，视为就绪
        if self._open_vocab_detector is not None or self._open_vocab_segmenter is not None:
            return True
        try:
            # 按配置加载 OWLv2 开放词汇检测器（用于关键词→框）
            if str(self.open_vocab_detector_backend).lower() == 'owlv2':
                from transformers import Owlv2ForObjectDetection, Owlv2Processor

                detector_processor = Owlv2Processor.from_pretrained(self.open_vocab_detector_model)
                detector_model = Owlv2ForObjectDetection.from_pretrained(self.open_vocab_detector_model)
                detector_model.to(self.model.device)
                detector_model.eval()
                self._open_vocab_detector = {
                    'processor': detector_processor,
                    'model': detector_model,
                }
                print("OWLv2 开放词汇检测器加载成功")
            # 按配置加载 SAM 分割器（由检测框生成掩码伪标签）
            if str(self.open_vocab_segmenter_backend).lower() == 'sam':
                from transformers import SamModel, SamProcessor

                seg_processor = SamProcessor.from_pretrained(self.open_vocab_segmenter_model)
                seg_model = SamModel.from_pretrained(self.open_vocab_segmenter_model)
                seg_model.to(self.model.device)
                seg_model.eval()
                self._open_vocab_segmenter = {
                    'processor': seg_processor,
                    'model': seg_model,
                }
                print("SAM 分割器加载成功")
            # 至少需有检测器，后续 _runtime_open_vocab_guidance 才可用
            return self._open_vocab_detector is not None
        except Exception as err:
            logging.warning('open-vocab runtime init failed, fallback to guidance file only: %s', err)
            # 标记禁用，避免每次请求重复尝试加载并重试失败
            self._open_vocab_runtime_disabled = True
            return False

    # ! 在线为「图像 + 关键词」生成开放词汇监督：检测框 →（可选）SAM 掩码，结果带缓存。
    def _runtime_open_vocab_guidance(self, image_path: str, keyword: str) -> dict[str, Any]:
        # 先检测是否有缓存，有缓存的话直接返回之前处理好的检测框和掩膜
        cache_key = f"{os.path.basename(str(image_path))}::{str(keyword).strip().lower()}"
        if cache_key in self._open_vocab_runtime_cache:
            return self._open_vocab_runtime_cache[cache_key]
        # 再检查开放词汇检测/分割器是否准备好
        if not self._ensure_open_vocab_runtime():
            return {}
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            return {}
        result: dict[str, Any] = {}
        try:
            # OWLv2：以关键词为文本提示，在图上检测相关目标框
            detector = self._open_vocab_detector
            if detector is not None:
                proc = detector['processor']
                model = detector['model']
                with torch.no_grad():
                    inputs = proc(text=[str(keyword)], images=image, return_tensors='pt')
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                    # print("outputs:", outputs)
                    # (H, W) → (W, H)，与 post_process 的 target_sizes 约定一致
                    target_sizes = torch.tensor([image.size[::-1]], device=self.model.device)
                    # transformers>=5.8 moves OWLv2 API to post_process_grounded_object_detection.
                    if hasattr(proc, "post_process_grounded_object_detection"):
                        post = proc.post_process_grounded_object_detection(
                            outputs=outputs,
                            threshold=float(self.open_vocab_detector_threshold),
                            target_sizes=target_sizes,
                            text_labels=[[str(keyword)]],
                        )[0]
                    else:
                        post = proc.post_process_object_detection(
                            outputs=outputs,
                            threshold=float(self.open_vocab_detector_threshold),
                            target_sizes=target_sizes,
                        )[0]
                boxes = post['boxes'].detach().cpu().numpy().tolist()
                scores = post['scores'].detach().cpu().numpy().tolist()
                
                result['boxes'] = [[int(round(v)) for v in box] for box in boxes]
                result['scores'] = [float(s) for s in scores]
        except Exception as err:
            logging.debug('runtime detector failed for keyword %s: %s', keyword, err)
            result = {}

        try:
            # SAM：用得分最高的首个检测框做提示，生成二值掩码伪标签
            segmenter = self._open_vocab_segmenter
            if segmenter is not None and result.get('boxes'):
                seg_proc = segmenter['processor']
                seg_model = segmenter['model']
                with torch.no_grad():
                    seg_inputs = seg_proc(images=image, input_boxes=[result['boxes'][:1]], return_tensors='pt')
                    seg_inputs = {k: v.to(self.model.device) for k, v in seg_inputs.items()}
                    seg_outputs = seg_model(**seg_inputs)
                    masks = seg_proc.image_processor.post_process_masks(
                        seg_outputs.pred_masks.cpu(),
                        seg_inputs['original_sizes'].cpu(),
                        seg_inputs['reshaped_input_sizes'].cpu(),
                    )[0]
                    iou_scores = seg_outputs.iou_scores.detach().cpu().numpy()[0]
                best_mask = None
                best_score = None
                # 在 SAM 输出的多候选掩码中取 IoU 分数最高者
                for i in range(masks.shape[0]):
                    for j in range(masks.shape[1]):
                        score = float(iou_scores[i, j])
                        mask_item = (masks[i, j].numpy() > 0).astype(np.uint8)
                        if best_score is None or score > best_score:
                            best_score = score
                            best_mask = mask_item
                if best_mask is not None:
                    result['mask'] = best_mask
                    result['seg_score'] = float(best_score if best_score is not None else 0.0)
        except Exception as err:
            logging.debug('runtime segmenter failed for keyword %s: %s', keyword, err)

        if result:
            self._open_vocab_runtime_cache[cache_key] = result
        return result

    # ! 按「图像 + 关键词」查找开放词汇监督：优先在线检测/分割，否则回退到预加载 JSON。
    def _lookup_open_vocab_guidance(self, image_path: str, keyword: str) -> dict[str, Any]:
        # 第一优先级：运行时 OWL 检测 +（可选）SAM 掩码，结果可能来自缓存
        runtime_value = self._runtime_open_vocab_guidance(image_path=image_path, keyword=keyword)
        if runtime_value:
            runtime_value['source'] = 'open_vocab_runtime'
            return runtime_value
        # 第二优先级：初始化时从 open_vocab_guidance_path 加载的离线标注
        print("在线词汇监督加载失败")
        if not self._open_vocab_guidance:
            return {}
        image_key = os.path.basename(str(image_path))
        keyword_key = str(keyword).strip().lower()
        image_item = self._open_vocab_guidance.get(image_key, {})
        if not isinstance(image_item, dict):
            return {}
        value = image_item.get(keyword_key, {})
        if isinstance(value, dict):
            value = dict(value)
            value['source'] = 'open_vocab_guidance_file'
            return value
        return {}

    # ! 为单个关键词构建混合伪 GT：检测框 + 分割掩码融合，无有效监督时回退到注意力热图。
    def _build_hybrid_pseudo_gt_for_keyword(self, image_path: str, keyword: str, image_size, fallback_heatmap: np.ndarray | None = None):
        guidance = self._lookup_open_vocab_guidance(image_path=image_path, keyword=keyword)
        # 从开放词汇监督中解析检测与分割字段（运行时或离线 JSON）
        detection_boxes = guidance.get('boxes', []) if isinstance(guidance, dict) else []
        detection_scores = guidance.get('scores', []) if isinstance(guidance, dict) else []
        segmentation_mask = guidance.get('mask', None) if isinstance(guidance, dict) else None
        segmentation_score = guidance.get('seg_score', None) if isinstance(guidance, dict) else None

        # 优先用分割掩码，否则用最高分检测框矩形；二者融合为 mask_gt / bbox_gt / q_conf
        # 这里是将检测结果映射为图片中，获得二值化的真值
        pseudo_gt = build_hybrid_pseudo_gt(
            image_size=image_size,
            detection_boxes=detection_boxes,
            detection_scores=detection_scores,
            segmentation_mask=segmentation_mask,
            segmentation_score=segmentation_score,
        )
        pseudo_gt['detection_boxes'] = [
            [int(round(v)) for v in box]
            for box in detection_boxes
            if isinstance(box, (list, tuple, np.ndarray)) and len(box) == 4
        ]
        pseudo_gt['detection_scores'] = [float(s) for s in detection_scores]
        if segmentation_score is not None:
            pseudo_gt['segmentation_score'] = float(segmentation_score)
        pseudo_gt['source'] = guidance.get('source', 'open_vocab_hybrid')

        # 如果标签掩码存在，则返回标签真值，否则继续后面的代码逻辑
        if pseudo_gt.get('mask_gt', np.zeros((1, 1), dtype=np.uint8)).sum() > 0:
            return pseudo_gt

        # 若暂未提供外部开放词汇标签，使用注意力自举掩膜作为低置信伪监督回退。
        if fallback_heatmap is None:
            # print("没有标签真值，也没有后备热图")
            return pseudo_gt
        # 目前后面的内容不会参与运行
        ''' 在“主方案失败时，用注意力热图自举一个低置信的伪 GT”，保证流程不断，同时控制噪声风险'''
        # fallback_mask = auto_otsu(min_max_scale(fallback_heatmap))
        # pseudo_gt = build_hybrid_pseudo_gt(
        #     image_size=image_size,
        #     detection_boxes=[],
        #     detection_scores=[],
        #     segmentation_mask=fallback_mask,
        #     segmentation_score=0.2,
        # )
        # pseudo_gt['source'] = 'attention_bootstrap'
        # # 自举路径固定低置信，避免弱监督信号被当作高可信伪标签
        # pseudo_gt['q_conf'] = min(float(pseudo_gt.get('q_conf', 0.2)), 0.2)
        # print("Attention, there is no open-vocab guidance, use attention heatmap to bootstrap a low-confidence pseudo GT")
        # return pseudo_gt

    def _has_valid_pseudo_supervision(self, pseudo_gt: dict[str, Any]) -> bool:
        """判断关键词是否具备可用于挖头统计的伪监督（检测或分割任一存在）。"""
        if not isinstance(pseudo_gt, dict) or not pseudo_gt:
            return False
        has_det = bool(pseudo_gt.get('has_detection', False))
        has_seg = bool(pseudo_gt.get('has_segmentation', False))
        if has_det or has_seg:
            return True
        detection_boxes = pseudo_gt.get('detection_boxes', [])
        if isinstance(detection_boxes, (list, tuple)) and len(detection_boxes) > 0:
            return True
        seg_mask = pseudo_gt.get('seg_mask', None)
        if seg_mask is not None:
            try:
                return bool(np.asarray(seg_mask).sum() > 0)
            except Exception:
                return False
        return False

    # ! 基于伪 GT 对各关键词的注意力头打分，跨词聚合后选 top-k 且去冗余的头。
    def _mine_heads_for_keywords(self, keyword_att_maps: dict[str, dict[str, np.ndarray]], pseudo_gts: dict[str, dict[str, Any]], scan_cfg):
        keyword_head_metrics = {}
        # 逐关键词、逐 head 打分，得到每个关键词的每个头的打分结果
        for keyword, head_maps in keyword_att_maps.items():
            # 获取当前关键词的伪 GT
            gt = pseudo_gts.get(keyword, {})
            # 无检测框且无语义掩码时，不参与关键注意力头统计。
            if not self._has_valid_pseudo_supervision(gt):
                continue
            # 初始化当前关键词的每个头的打分结果
            metrics_per_head = {}
            # 逐头计算 IoU / pointing / AUC 等，并按伪 GT 置信度加权
            for head_key, att_map in head_maps.items():
                metrics_per_head[head_key] = score_single_head(
                    att_map=att_map,
                    pseudo_gt=gt,
                    score_weights=self.auto_head_metric_weights,
                    att_quantile=self.attention_threshold_quantile,
                )
            keyword_head_metrics[keyword] = metrics_per_head

        # 跨关键词汇总均值、稳定性，得到每个头的 final_score
        aggregated = aggregate_head_scores(keyword_head_metrics)
        ref_maps = next(iter(keyword_att_maps.values())) if keyword_att_maps else {}
        # 按分数排序，用余弦相似度剔除与已选头过于冗余的候选
        ranked_heads, selected = select_top_heads_with_pruning(
            head_agg_scores=aggregated,
            reference_att_maps=ref_maps,
            top_k=self.auto_head_top_k,
            similarity_threshold=self.head_similarity_threshold,
        )
        used_default_fallback = False
        if not selected:
            print("当前没有找到合适的注意力头，使用默认头")
            selected = [{'head': key} for key in scan_cfg['default_items'][:max(1, int(self.head_num))]]
            used_default_fallback = True
        return {
            'keyword_head_metrics': keyword_head_metrics,
            'head_agg_scores': aggregated,
            'ranked_heads': ranked_heads,
            'selected_heads': selected,
            'selected_head_keys': [item['head'] for item in selected],
            'used_default_fallback': used_default_fallback,
        }

    # ? 把本次 head mining 结果写入模型级统计画像，并在配置允许时导出到磁盘。
    def _export_head_mining_statistics(self, selected_heads, head_agg_scores, dataset: str | None = None):
        scope_name = self._profile_scope_name(dataset)
        self._load_head_profile_from_disk(scope_name)
        # 更新内存画像：各头出现次数、被选次数、分数均值/方差
        self.head_mining_profile = update_model_head_profile(
            profile=self.head_mining_profile,
            model_name=scope_name,
            selected_heads=selected_heads,
            head_agg_scores=head_agg_scores,
        )
        if not self.auto_head_export:
            return {}
        export_payload = export_model_head_profile(
            profile=self.head_mining_profile,
            model_name=scope_name,
            output_dir=self.auto_head_stats_dir,
            top_k=max(8, int(self.auto_head_top_k)),
        )
        if export_payload:
            export_payload['profile_scope'] = scope_name
            export_payload['dataset_scope'] = self._dataset_scope_tag(dataset)
        return export_payload

    # 功能：前向提取关键词到视觉 token 的注意力，并生成关键词热力图集合。
    def _forward_attention_for_keywords(self, message, keywords, dataset=None):
        # 无关键词时无法做“关键词->注意力”对齐，直接返回空结果。
        if not keywords:
            return None
        # 复制一份输入消息，避免后续处理意外修改外部原始 message。
        attn_message = [dict(item) for item in message]
        # 读取首图作为注意力分析主图；当前流程默认首个元素为 image。
        current_image = Image.open(attn_message[0]['value']).convert("RGB")
        # 与原始 specific_norm_res 流程保持一致：先在方图空间提取注意力，再回映到原图空间。
        analysis_image = resize_to_square(current_image, current_image.size)
        # 解析注意力扫描配置（层、头、默认候选头等）与映射函数实现。
        # 加载每个模型对应的的注意力层和头数量
        scan_cfg = self._resolve_attention_scan_config()
        map_func = self._resolve_attention_map_func()
        # print(f"scan_cfg:{scan_cfg}")
        # 通用辅助提示：给注意力提取函数一个稳定的上下文问题。
        general_prompt = "Write a general description of the image. Answer the question using a single word or phrase."
        image_size = current_image.size
        # head 选择模式：mine=自动挖掘，inference=使用预选头；非法值回退 mine。
        mode = str(self.head_selection_mode).lower()
        if mode not in {'mine', 'inference'}:
            mode = 'mine'

        # keyword_att_maps: 每个关键词对应的“head_key -> 归一化注意力图”。
        keyword_att_maps: dict[str, dict[str, np.ndarray]] = {}
        # pseudo_gts: 每个关键词对应的伪监督（mask/source/confidence），仅 mine 模式使用。
        pseudo_gts: dict[str, dict[str, Any]] = {}
        # mine 模式下，仅记录具备伪监督（检测框或语义掩码）的关键词。
        valid_supervision_keywords: set[str] = set()
        # keyword_heatmap_items: 后续用于融合的关键词级热图与统计信息。
        keyword_heatmap_items = []

        # 第一轮：逐关键词提取所有候选头的注意力图。
        for keyword in keywords:
            # mine 模式先尝试构建伪 GT（用于后续 head 打分/筛选）。
            if mode == 'mine':
                pseudo_gts[keyword] = self._build_hybrid_pseudo_gt_for_keyword(
                    image_path=attn_message[0]['value'],
                    keyword=keyword,
                    image_size=image_size,
                    fallback_heatmap=None,
                )
                if self._has_valid_pseudo_supervision(pseudo_gts[keyword]):
                    valid_supervision_keywords.add(keyword)
                # print(f"get :pseudo_gts:{pseudo_gts}")
                # print("pseudo_gts:", pseudo_gts)
                # 若初次伪 GT 为空（无有效 mask），用后备热图再次尝试构建伪 GT。
                # if pseudo_gts[keyword].get('mask_gt', np.zeros((1, 1), dtype=np.uint8)).sum() <= 0:
                    # print("本keyword无有效mask")
                
        
            # 当前实现把关键词本身作为 prompt 触发关键词相关注意力。
            prompt = str(keyword)
            raw_att_maps = map_func(
                analysis_image,
                prompt,
                general_prompt,
                self.model,
                self.processor,
                scan_cfg['layers'],
                scan_cfg['heads'],
            )
            # print(f"get_raw_att_maps:{raw_att_maps}")
            # raw_att_maps代表了当前keyword所有的注意力头map，包括了所有的层和头
            
            # 将不同层/头返回的原始注意力统一到原图尺寸，便于后续融合与可视化。
            norm_head_maps = prepare_attention_maps_for_image(raw_att_maps, image_size=image_size)
            keyword_att_maps[keyword] = norm_head_maps

            # 逐关键词后立即释放 GPU 缓存，避免多个关键词的注意力张量累积
            del raw_att_maps
            torch.cuda.empty_cache()
     

        # 第二轮：根据模式确定最终用于融合的 head 列表。
        # used_default_fallback：若本次走了“默认头兜底”，则跳过统计导出，避免污染头挖掘画像。
        used_default_fallback = False
        if mode == 'mine' and self.auto_head_mining:
            
            # 自动挖掘：基于伪 GT 对候选头打分并选出最优头集合。
            # 如果对应的keyword没有真值mask的话，则只计算有真值mask的，最后统计出来当前这些关键词综合对应的注意力头索引
            mining_result = self._mine_heads_for_keywords(keyword_att_maps, pseudo_gts, scan_cfg)
            selected_heads = mining_result['selected_head_keys']
            head_agg_scores = mining_result['head_agg_scores']
            ranked_heads = mining_result['ranked_heads']
            used_default_fallback = bool(mining_result.get('used_default_fallback', False))
        elif mode == 'mine': # 这个模式一般不会用到
            # mine 但关闭自动挖掘：直接使用默认头。
            selected_heads = scan_cfg['default_items'][:max(1, int(self.head_num))]
            ranked_heads = []
            head_agg_scores = {}
            used_default_fallback = True
        else:
            # inference 模式：读取外部配置的已选头（可按数据集区分）。
            selected_heads = self._load_inference_selected_heads(scan_cfg=scan_cfg, dataset=dataset)
            ranked_heads = []
            head_agg_scores = {}

        # 第三轮：对每个关键词，仅用 selected_heads 生成最终关键词热图并记录统计。
        for keyword in keywords:
            # mine 模式下，无伪监督关键词不参与关键词级热图统计与后续关键头统计链路。
            if mode == 'mine' and keyword not in valid_supervision_keywords:
                continue
            maps_for_kw = keyword_att_maps.get(keyword, {})
            chosen_maps = [maps_for_kw[k] for k in selected_heads if k in maps_for_kw]
            # 某关键词若没有命中任何已选头，跳过该关键词。
            if not chosen_maps:
                continue
            # 将该关键词命中的selected_heads多头注意力图融合为单图，并计算均值强度作为权重参考。
            heatmap = composite_attn_map(chosen_maps)
            score = float(heatmap.mean())
            keyword_heatmap_items.append({
                'keyword': keyword,
                'token_count': 0,
                'mean_attention': score,
                'heatmap': heatmap,
                'pseudo_gt_source': pseudo_gts.get(keyword, {}).get('source', 'inference_selected_heads'),
                'pseudo_gt_confidence': float(pseudo_gts.get(keyword, {}).get('q_conf', 0.0)) if mode == 'mine' else 0.0,
                'pseudo_gt': pseudo_gts.get(keyword, {}) if mode == 'mine' else {},
            })

        # 所有关键词都失败时返回 None，触发上层回退策略。
        if not keyword_heatmap_items:
            print("所有关键词heatmap都提取失败，返回None")
            return None

        # 根据阈值和 top-k 规则筛选可参与融合的关键词。（去除掉评分为0 ，和多余关键词，但默认都保留）
        all_keyword_heatmap_items = list(keyword_heatmap_items)
        selected_items, dropped_items = self._select_keyword_items_for_fusion(keyword_heatmap_items)
        if not selected_items:
            return None
        keyword_heatmaps = [item['heatmap'] for item in selected_items]
        keyword_scores = [float(item['mean_attention']) for item in selected_items]
        # 将多关键词热图按配置策略融合为单张热图。
        fused_heatmap = self._fuse_keyword_heatmaps(keyword_heatmaps, keyword_scores)
        if fused_heatmap is None:
            return None


        # 组织保留/丢弃关键词的摘要信息，便于调试与可解释性分析。
        keyword_details = [
            {
                'keyword': item['keyword'],
                'token_count': item['token_count'],
                'mean_attention': item['mean_attention'],
            }
            for item in selected_items
        ]
        dropped_keyword_details = [
            {
                'keyword': item['keyword'],
                'token_count': item['token_count'],
                'mean_attention': item['mean_attention'],
                'pseudo_gt_source': item.get('pseudo_gt_source', 'unknown'),
            }
            for item in dropped_items
        ]
        visual_positions = 0
        export_result = {}
        if mode == 'mine':
            if used_default_fallback:
                # 走了“默认头兜底”路径，本次结果不能代表挖掘出的真实有效头，
                # 跳过统计导出，避免默认头污染头挖掘画像。
                print("使用了默认头兜底，跳过本次注意力头统计写入。")
            else:
                # 导出注意力头挖掘统计信息
                export_result = self._export_head_mining_statistics(
                    selected_heads=[{'head': key} for key in selected_heads],
                    head_agg_scores=head_agg_scores,
                    dataset=dataset,
                )
        # 返回融合热图及调试元信息（统计项 + 关键词级详情）。
        return {
            'heatmap': fused_heatmap,
            'visual_token_count': visual_positions,
            'keyword_token_count': int(sum(item['token_count'] for item in keyword_details)),
            'keyword_details': keyword_details,
            'dropped_keyword_details': dropped_keyword_details,
            'keyword_heatmap_items': selected_items,
            'all_keyword_heatmap_items': all_keyword_heatmap_items,
            'selected_heads': selected_heads,
            'ranked_heads': ranked_heads[:50] if ranked_heads else [],
            'head_stats_export': export_result,
            'model_signature': self._model_signature(),
            'head_selection_mode': mode,
        }

    # 功能：执行两阶段推理（阶段一提关键词，阶段二基于掩膜图生成最终回答）。
    def _generate_inner_transformers_two_stage(self, message, dataset=None):
        # 提取首张图作为两阶段注意力流程的主图输入。
        primary_image_path = self._extract_primary_image_path(message)
        if primary_image_path is None:
            # two_stage_attention 依赖图像输入；缺图时回退到单
            # 阶段生成。
            logging.warning('two_stage_attention requires at least one image; fallback to single_stage.')
            return self._generate_inner_transformers_single(message, dataset=dataset)

        # 阶段一：在原消息后追加分析指令，先生成细粒度描述文本。
        stage1_message = self._stage1_message(message)
        # print(f"stage1_message:{stage1_message}")
        if self.stage1_keyword_source == 'api':
            stage1_response = self._generate_stage1_with_api(message)
            if not stage1_response:
                # API 不可用时，回退到本地阶段一。
                print("api模型不可用，回退到本地阶段一")
                stage1_response = self._generate_inner_transformers_single(
                    stage1_message,
                    dataset=dataset,
                    generation_kwargs={'max_new_tokens': self.stage1_max_new_tokens},
                )
        else:
            stage1_response = self._generate_inner_transformers_single(
                stage1_message,
                dataset=dataset,
                generation_kwargs={'max_new_tokens': self.stage1_max_new_tokens},
            )
        # 阶段一推理完成后立即释放 KV cache 及中间产物
        torch.cuda.empty_cache()
    
        # print(f"stage1_response:{stage1_response}")
        # print("--------------------------------")
        # 从阶段一响应中提取关键词；若失败则退化为从原问题文本提取。
        keywords = self._extract_keywords(stage1_response)
        if not keywords:
            keywords = self._extract_keywords(self._extract_question_text(message))
    
        # print(f"keywords:{keywords}")
        # print("--------------------------------")
        # 基于关键词执行一次注意力前向，得到融合热图及关键词级统计信息。
        attn_result = self._forward_attention_for_keywords(message, keywords, dataset=dataset)
        # 注意力前向完成后释放 GPU 缓存，为后续掩码构建腾出空间
        torch.cuda.empty_cache()
        
        if attn_result is None:
            # 注意力抽取失败时回退原图推理，同时记录失败原因用于调试。
            logging.warning('attention extraction failed in two_stage_attention; fallback to original image.')
            final_response = self._generate_inner_transformers_single(message, dataset=dataset)
            self.last_two_stage_debug = {
                'stage1_response': stage1_response,
                'keywords': keywords,
                'fallback': 'attention_extraction_failed',
            }
            return final_response

        # LLaVA(Combined)：逐关键词显著区域并集 → 一张外接框 crop。
        combined_mask, region_debug = self._build_combined_keyword_mask(
            primary_image_path,
            attn_result.get('keyword_heatmap_items', []),
        )
        if isinstance(region_debug, dict):
            # 大数组仅用于内部 crop，不写入调试落盘。
            region_debug.pop('per_keyword_masks', None)
        if combined_mask is not None and combined_mask.any():
            masked_image_path, mask_debug = self._build_masked_image_from_mask(
                primary_image_path, combined_mask
            )
            if isinstance(mask_debug, dict):
                mask_debug['region_meta'] = region_debug
        else:
            print("逐关键词区域提取为空，回退融合热图阈值 crop")
            masked_image_path, mask_debug = self._build_masked_image(
                primary_image_path, attn_result['heatmap']
            )
        if masked_image_path is None:
            print("crop 构建失败，回退原图推理")
            final_response = self._generate_inner_transformers_single(message, dataset=dataset)
            self.last_two_stage_debug = {
                'stage1_response': stage1_response,
                'keywords': keywords,
                'fallback': 'crop_build_failed',
            }
            return final_response

        # 可选：为每个保留关键词单独保存叠加图，便于逐词检查视觉对齐效果。（待检查）
        per_keyword_overlays = []
        keyword_items_for_overlay = attn_result.get(
            'all_keyword_heatmap_items',
            attn_result.get('keyword_heatmap_items', []),
        )
        if self.save_per_keyword_overlay:
            overlay_dir = None
            pseudo_gt_dir = None
            if self.attention_debug_dir:
                overlay_dir = os.path.join(self.attention_debug_dir, 'per_keyword_overlays')
                os.makedirs(overlay_dir, exist_ok=True)
                pseudo_gt_dir = os.path.join(self.attention_debug_dir, 'per_keyword_pseudo_gt')
                os.makedirs(pseudo_gt_dir, exist_ok=True)
            for idx, item in enumerate(keyword_items_for_overlay):
                keyword = item.get('keyword', '')
                tag = self._keyword_file_tag(keyword=keyword, index=idx)
                overlay_out = (
                    os.path.join(overlay_dir, f'{tag}_overlay.png')
                    if overlay_dir
                    else None
                )
                heatmap_out = (
                    os.path.join(overlay_dir, f'{tag}_heatmap.png')
                    if overlay_dir
                    else None
                )
                heatmap_npy_out = (
                    os.path.join(overlay_dir, f'{tag}_heatmap.npy')
                    if overlay_dir
                    else None
                )
                pseudo_gt_npy_out = (
                    os.path.join(pseudo_gt_dir, f'{tag}_pseudo_gt.npy')
                    if pseudo_gt_dir
                    else None
                )
                pseudo_mask_npy_out = (
                    os.path.join(pseudo_gt_dir, f'{tag}_mask.npy')
                    if pseudo_gt_dir
                    else None
                )
                pseudo_mask_png_out = (
                    os.path.join(pseudo_gt_dir, f'{tag}_mask.png')
                    if pseudo_gt_dir
                    else None
                )
                pseudo_overlay_out = (
                    os.path.join(pseudo_gt_dir, f'{tag}_pseudo_overlay.png')
                    if pseudo_gt_dir
                    else None
                )
                overlay_path = self._save_overlay_from_heatmap(
                    primary_image_path,
                    item['heatmap'],
                    output_path=overlay_out,
                )
                heatmap_path = self._save_heatmap_png(
                    item['heatmap'],
                    output_path=heatmap_out,
                )
                if heatmap_npy_out is not None:
                    np.save(heatmap_npy_out, item['heatmap'].astype(np.float32))
                pseudo_saved = self._save_keyword_pseudo_gt_artifacts(
                    primary_image_path,
                    item.get('pseudo_gt', {}),
                    info_npy_path=pseudo_gt_npy_out,
                    mask_npy_path=pseudo_mask_npy_out,
                    mask_png_path=pseudo_mask_png_out,
                    overlay_png_path=pseudo_overlay_out,
                )
                if overlay_path is not None:
                    per_keyword_overlays.append({
                        'keyword': keyword,
                        'token_count': item['token_count'],
                        'mean_attention': item['mean_attention'],
                        'overlay_path': overlay_path,
                        'heatmap_path': heatmap_path,
                        'heatmap_npy_path': heatmap_npy_out,
                        'pseudo_gt_npy_path': pseudo_saved.get('pseudo_gt_npy_path'),
                        'pseudo_mask_npy_path': pseudo_saved.get('mask_npy_path'),
                        'pseudo_mask_png_path': pseudo_saved.get('mask_png_path'),
                        'pseudo_overlay_path': pseudo_saved.get('pseudo_overlay_path'),
                    })
        
        # 阶段二：原图 + Combined crop + 关键词，再生成最终答案。
        stage2_message = self._add_masked_image_and_keywords(message, masked_image_path, keywords)
        
        # 阶段二推理前清理 GPU 缓存，确保为最终生成腾出足够显存
        torch.cuda.empty_cache()
        # print(f"stage2_message:{stage2_message}")
        # print("--------------------------------")
        # 在完成增强提示方法后，再将之后生成的prompt补充进去，再重新进行推理
        final_response = self._generate_inner_transformers_single(stage2_message, dataset=dataset)
        # print(f"final_response:{final_response}")
        
        # 添加调试信息
        self.last_two_stage_debug = {
            'stage1_response': stage1_response,
            'keywords': keywords,
            'attention_meta': {
                'visual_token_count': attn_result['visual_token_count'],
                'keyword_token_count': attn_result['keyword_token_count'],
                'fusion_mode': self.keyword_attention_fusion,
                'keyword_top_k': int(self.keyword_top_k),
                'keyword_min_attention': float(self.keyword_min_attention),
                'model_signature': attn_result.get('model_signature', ''),
                'selected_heads': attn_result.get('selected_heads', []),
                'head_selection_mode': attn_result.get('head_selection_mode', self.head_selection_mode),
            },
            'keyword_details': attn_result.get('keyword_details', []),
            'dropped_keyword_details': attn_result.get('dropped_keyword_details', []),
            'ranked_heads': attn_result.get('ranked_heads', []),
            'head_stats_export': attn_result.get('head_stats_export', {}),
            'per_keyword_overlays': per_keyword_overlays,
            'mask_meta': mask_debug,
        }
        return final_response

    # 功能：执行单阶段 Transformers 推理并返回模型响应。（原始方法）
    def _generate_inner_transformers_single(self, message, dataset=None, generation_kwargs=None):
        # 在此处只调用了两个函数，先是按照message的格式进行prompt包装，然后进行生成。
        _, _, inputs = self._prepare_transformer_inputs(message, dataset=dataset)
        response = self._generate_with_inputs(inputs, generation_kwargs=generation_kwargs)
        # print(f"response:{response}")
        return response

    # 功能：在 Transformers 后端中按推理模式分发到单阶段或两阶段流程。
    def generate_inner_transformers(self, message, dataset=None):
        # 在此处添加我们的方法，根据reasoning_mode的值，选择不同的生成方式
        if self.reasoning_mode == 'two_stage_attention':
            # print("two_stage_attention")
            return self._generate_inner_transformers_two_stage(message, dataset=dataset)
        return self._generate_inner_transformers_single(message, dataset=dataset)

    # 功能：统一推理入口。LLaVA(HF) 两阶段依赖前向注意力，仅走 Transformers 后端。
    def generate_inner(self, message, dataset=None):
        return self.generate_inner_transformers(message, dataset=dataset)

    # 功能：处理多轮对话输入并在 Transformers 后端完成生成，返回最终回复文本。
    def chat_inner(self, message, dataset=None):
        # Multi-turn chat path for MT benchmarks (e.g., MMDU).
        assert len(message) > 0 and message[-1]['role'] == 'user'

        conversation = []
        images = []
        for turn in message:
            role = turn['role']
            if role not in ['system', 'user', 'assistant']:
                continue
            content = turn.get('content', '')
            if isinstance(content, str):
                conversation.append({'role': role, 'content': [{'type': 'text', 'text': content}]})
            else:
                turn_content, turn_images = self._prepare_content(content, dataset=dataset)
                images.extend(turn_images)
                conversation.append({'role': role, 'content': turn_content})

        text = self._apply_chat_template(conversation)
        target_dtype = next(self.model.parameters()).dtype
        inputs = self.processor(
            images=images if images else None,
            text=text,
            return_tensors='pt',
        ).to(self.model.device, target_dtype)

        return self._generate_with_inputs(inputs)
