import torchvision.transforms.functional as TF
import numpy as np
import os
import csv
import json
from scipy.ndimage import median_filter
from skimage.measure import block_reduce

from io import BytesIO
import base64

def encode_base64(image):
    """
    Encodes a PIL image to a base64 string.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def bbox_to_mask(bbox, shape):
    # 根据目标尺寸创建空白二值掩码（0 表示背景）。
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    # bbox 为空或格式非法时，直接返回全 0 掩码。
    if bbox is None or len(bbox) != 4:
        return mask
    # 将输入框坐标转为整数，便于后续切片索引。
    x1, y1, x2, y2 = [int(v) for v in bbox]
    # 将坐标裁剪到图像边界范围内，避免越界访问。
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    # 若裁剪后框无有效面积，则返回空掩码。
    if x2 <= x1 or y2 <= y1:
        return mask
    # 将框内区域置为 1，得到矩形前景掩码。
    mask[y1:y2, x1:x2] = 1
    return mask

# ! 在此处设置GT真值的阈值二值化方法
def normalize_binary_mask(mask, shape, threshold_method='constant', threshold=0.5, block_size=64):
    # 当输入掩码为空时，返回与目标 shape 对齐的全 0 掩码。
    if mask is None:
        return np.zeros(shape, dtype=np.uint8)
    # 将输入统一转为 numpy 数组，便于后续维度与尺寸处理。
    arr = np.asarray(mask)
    # 仅接受二维掩码；其他维度视为无效并回退为全 0。
    if arr.ndim != 2:
        return np.zeros(shape, dtype=np.uint8)
    # 若尺寸不匹配，先插值缩放到目标分辨率（宽, 高）。
    if arr.shape != shape:
        arr = resize_with_interpolation(arr.astype(np.float32), (shape[1], shape[0]))
    arr = arr.astype(np.float32)

    # 二值化方式可选：
    # - constant: 固定阈值
    # - otsu: 全局 Otsu 自动阈值
    # - adaptive: 自适应阈值
    # - local_otsu: 分块 Otsu 阈值
    method = str(threshold_method).strip().lower()
    if method in ('constant', 'fixed'):
        binary = constant_threshold(arr, threshold=float(threshold))
    else:
        # 自动阈值方法默认按 [0, 1] 输入设计，先做归一化以保证稳定性。
        arr_norm = min_max_scale(arr)
        if arr_norm is None:
            return np.zeros(shape, dtype=np.uint8)
        if method in ('otsu', 'auto_otsu'):
            binary = auto_otsu(arr_norm)
        elif method in ('adaptive', 'adaptive_threshold'):
            binary = adaptive_threshold(arr_norm)
        elif method in ('local_otsu', 'local'):
            binary = local_otsu(arr_norm, block_size=int(block_size))
        else:
            raise ValueError(
                f"Unsupported threshold_method: {threshold_method}. "
                "Use one of: constant, otsu, adaptive, local_otsu."
            )
    return (np.asarray(binary) > 0.5).astype(np.uint8)


def build_hybrid_pseudo_gt(image_size, detection_boxes=None, detection_scores=None, segmentation_mask=None, segmentation_score=None):
    """将检测框与分割掩码融合为统一伪 GT，供注意力头挖掘/评估使用。"""
    width, height = int(image_size[0]), int(image_size[1])
    shape = (height, width)
    if detection_boxes is None:
        detection_boxes = []
    elif isinstance(detection_boxes, np.ndarray):
        detection_boxes = detection_boxes.tolist()
    else:
        detection_boxes = list(detection_boxes)
    if detection_scores is None:
        detection_scores = []
    elif isinstance(detection_scores, np.ndarray):
        detection_scores = detection_scores.tolist()
    else:
        detection_scores = list(detection_scores)
    det_mask = np.zeros(shape, dtype=np.uint8)
    bbox_gt = None
    # 检测分支：取得分最高框，转为矩形掩码
    if len(detection_boxes) > 0:
        best_idx = 0
        if len(detection_scores) > 0 and len(detection_scores) == len(detection_boxes):
            best_idx = int(np.argmax(np.asarray(detection_scores, dtype=np.float32)))
        bbox_gt = [int(v) for v in detection_boxes[best_idx]]
        det_mask = bbox_to_mask(bbox_gt, shape)
    # 分割分支：resize 并二值化到目标尺寸
    seg_mask = normalize_binary_mask(segmentation_mask, shape)
    has_seg = bool(seg_mask.any())
    has_det = bool(det_mask.any())
    # 融合策略：分割优先，否则回退到检测矩形，均无则为全零
    if has_seg:
        fused = seg_mask
    elif has_det:
        fused = det_mask
    else:
        fused = np.zeros(shape, dtype=np.uint8)
    # 检测/分割置信度各取 50% 加权，作为伪标签整体可信度
    det_conf = (
        float(np.max(np.asarray(detection_scores, dtype=np.float32)))
        if len(detection_scores) > 0
        else (1.0 if has_det else 0.0)
    )
    seg_conf = float(segmentation_score) if segmentation_score is not None else (1.0 if has_seg else 0.0)
    q_conf = float(0.5 * det_conf + 0.5 * seg_conf) if (has_det or has_seg) else 0.0
    return {
        'mask_gt': fused.astype(np.uint8), # 融合后的掩码，分割优先，检测其次，否则全0
        'bbox_gt': bbox_gt, # 检测框坐标
        'det_mask': det_mask.astype(np.uint8), # 检测框掩码
        'seg_mask': seg_mask.astype(np.uint8), # 分割掩码
        'q_conf': q_conf, # 加权置信度
        'has_detection': has_det, # 是否存在检测框
        'has_segmentation': has_seg, # 是否存在分割掩码
    }


def normalize_attention_map_for_eval(att_map, image_size):
    """将原始注意力图规范化到评估空间：缩放、平滑、还原比例并做最小最大归一化。"""
    att_item = resize_with_interpolation(att_map, image_size)
    block_att = gaussian_filter(att_item, sigma=1, mode='reflect')
    right_att_map = square_array_to_orin(block_att, image_size)
    scaled = min_max_scale(right_att_map)
    # 极端情况下归一化失败时，返回同尺寸全 0 图，避免后续计算报错。
    if scaled is None:
        return np.zeros((image_size[1], image_size[0]), dtype=np.float32)
    return scaled.astype(np.float32)


def prepare_attention_maps_for_image(att_maps, image_size):
    """批量规范化多头注意力图，并统一 key 为字符串类型。"""
    result = {}
    for key, value in att_maps.items():
        result[str(key)] = normalize_attention_map_for_eval(value, image_size)
    return result


def _binary_from_heatmap(att_map, quantile=0.8):
    """按分位数阈值将连续热力图转为二值掩码。"""
    # 分位数限制在 [0, 1]，避免非法阈值导致异常。
    q = float(max(0.0, min(1.0, quantile)))
    threshold = float(np.quantile(att_map, q))
    return (att_map >= threshold).astype(np.uint8)


def _safe_iou(a_mask, b_mask):
    """计算两个二值掩码的 IoU；尺寸不一致时先对齐到 a_mask。"""
    if a_mask.shape != b_mask.shape:
        b_mask = resize_with_interpolation(b_mask.astype(np.float32), (a_mask.shape[1], a_mask.shape[0]))
        b_mask = (b_mask > 0.5).astype(np.uint8)
    inter = np.logical_and(a_mask == 1, b_mask == 1).sum()
    union = np.logical_or(a_mask == 1, b_mask == 1).sum()
    # 并集为 0 时返回 0，避免除零错误。
    return float(inter / union) if union > 0 else 0.0


def _pointing_score(att_map, gt_mask):
    """Pointing Game 指标：注意力峰值点是否落在目标区域内。"""
    if att_map.shape != gt_mask.shape:
        gt_mask = resize_with_interpolation(gt_mask.astype(np.float32), (att_map.shape[1], att_map.shape[0]))
        gt_mask = (gt_mask > 0.5).astype(np.uint8)
    # 无有效 GT 区域时，该指标记为 0。
    if gt_mask.sum() <= 0:
        return 0.0
    y, x = np.unravel_index(np.argmax(att_map), att_map.shape)
    return float(gt_mask[y, x] > 0)


def _entropy(att_map):
    """计算注意力分布熵，衡量注意力是否分散。"""
    data = np.asarray(att_map, dtype=np.float32).reshape(-1)
    data = np.maximum(data, 0.0)
    s = float(data.sum())
    if s <= 1e-8:
        return 0.0
    p = data / s
    p = np.clip(p, 1e-8, 1.0)
    return float(-(p * np.log(p)).sum())


def _sparsity(att_map, threshold=0.1):
    """计算稀疏度：低于阈值的像素占比。"""
    data = np.asarray(att_map, dtype=np.float32)
    return float((data <= float(threshold)).mean())


def _auc_iou(att_map, gt_mask, num_steps=7):
    """在多个阈值下计算 IoU 并取均值，近似 IoU-AUC。"""
    if gt_mask.sum() <= 0:
        return 0.0
    thresholds = np.linspace(0.3, 0.9, num_steps)
    ious = []
    for t in thresholds:
        pred = (att_map >= t).astype(np.uint8)
        ious.append(_safe_iou(pred, gt_mask))
    return float(np.mean(ious)) if ious else 0.0


def score_single_head(att_map, pseudo_gt, score_weights=None, att_quantile=0.8):
    """对单个注意力头打分，融合 IoU/Pointing/AUC/熵等指标。"""
    weights = score_weights or {
        'iou': 0.45,
        'pointing': 0.25,
        'auc': 0.20,
        'entropy': 0.10,
    }
    gt_mask = np.asarray(pseudo_gt.get('mask_gt', 0), dtype=np.uint8)
    # 无有效伪 GT 时不返回分数，交由上游忽略该头。
    if gt_mask.ndim != 2 or gt_mask.sum() <= 0:
        return None
    if gt_mask.shape != att_map.shape:
        gt_mask = resize_with_interpolation(gt_mask.astype(np.float32), (att_map.shape[1], att_map.shape[0]))
        gt_mask = (gt_mask > 0.5).astype(np.uint8)
    # ! 在此处完成注意力图的二值化，返回二值化的注意力图
    binary_pred = _binary_from_heatmap(att_map, quantile=att_quantile)
    iou = _safe_iou(binary_pred, gt_mask)
    pointing = _pointing_score(att_map, gt_mask)
    auc = _auc_iou(att_map, gt_mask)
    entropy = _entropy(att_map)
    sparsity = _sparsity(att_map)
    # 熵做尺度归一化，避免不同分辨率注意力图之间不可比。
    entropy_norm = entropy / np.log(float(att_map.size) + 1e-8)
    head_score = (
        float(weights.get('iou', 0.45)) * iou
        + float(weights.get('pointing', 0.25)) * pointing
        + float(weights.get('auc', 0.20)) * auc
        - float(weights.get('entropy', 0.10)) * entropy_norm
    )
    q_conf = float(pseudo_gt.get('q_conf', 1.0))
    return {
        'iou': iou,
        'pointing': pointing,
        'auc': auc,
        'entropy': entropy,
        'entropy_norm': entropy_norm,
        'sparsity': sparsity,
        'score': head_score,
        'score_weighted': head_score * q_conf,
        'q_conf': q_conf,
    }


def aggregate_head_scores(keyword_head_metrics):
    """跨关键词聚合各注意力头分数，并估计稳定性（均值+方差）。"""
    agg = {}
    for _, head_metrics in keyword_head_metrics.items():
        for head_key, metrics in head_metrics.items():
            if metrics is None:
                continue
            item = agg.setdefault(head_key, {'count': 0, 'sum_score': 0.0, 'sum_sq_score': 0.0})
            score_w = float(metrics.get('score_weighted', metrics.get('score', 0.0)))
            item['count'] += 1
            item['sum_score'] += score_w
            item['sum_sq_score'] += score_w * score_w
    for head_key, item in agg.items():
        count = max(1, int(item['count']))
        mean = item['sum_score'] / count
        var = max(0.0, item['sum_sq_score'] / count - mean * mean)
        item['mean_score'] = float(mean)
        item['std_score'] = float(np.sqrt(var))
        # 标准差越小稳定性越高，映射到 (0, 1] 区间。
        item['stability'] = float(1.0 / (1.0 + item['std_score']))
        item['final_score'] = float(item['mean_score'] * item['stability'])
    return agg


def _cosine_similarity(a, b):
    """计算两个向量的余弦相似度，并处理零向量边界情况。"""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-8 or nb <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def select_top_heads_with_pruning(head_agg_scores, reference_att_maps, top_k=3, similarity_threshold=0.95):
    """按综合分选头，并基于相似度剪枝以减少冗余头。"""
    ranked = sorted(
        head_agg_scores.items(),
        key=lambda kv: (float(kv[1].get('final_score', 0.0)), float(kv[1].get('mean_score', 0.0))),
        reverse=True,
    )
    selected = []
    selected_vecs = []
    for head_key, metrics in ranked:
        if head_key not in reference_att_maps:
            continue
        vec = np.asarray(reference_att_maps[head_key], dtype=np.float32).reshape(-1)
        vec = vec - vec.mean()
        if vec.size == 0:
            continue
        # 与已选头过于相似则跳过，提升多样性。
        if selected_vecs:
            sims = [_cosine_similarity(vec, old) for old in selected_vecs]
            if max(sims) >= float(similarity_threshold):
                continue
        selected.append({'head': head_key, **metrics})
        selected_vecs.append(vec)
        if int(top_k) > 0 and len(selected) >= int(top_k):
            break
    return ranked, selected


def update_model_head_profile(profile, model_name, selected_heads, head_agg_scores):
    """更新模型级注意力头画像，累计出现次数、入选次数和分数统计量。"""
    model_profile = profile.setdefault(model_name, {})
    for head_key, metrics in head_agg_scores.items():
        slot = model_profile.setdefault(
            head_key,
            {'seen': 0, 'selected': 0, 'score_sum': 0.0, 'score_sq_sum': 0.0},
        )
        score = float(metrics.get('final_score', 0.0))
        slot['seen'] += 1
        slot['score_sum'] += score
        slot['score_sq_sum'] += score * score
    selected_keys = {item['head'] for item in selected_heads}
    for head_key in selected_keys:
        if head_key in model_profile:
            model_profile[head_key]['selected'] += 1
    return profile


def export_model_head_profile(profile, model_name, output_dir, top_k=16):
    """导出模型头画像到 JSON/CSV，并返回导出路径与 Top-K 结果。"""
    if model_name not in profile:
        return {}
    os.makedirs(output_dir, exist_ok=True)
    model_profile = profile[model_name]
    rows = []
    for head_key, stats in model_profile.items():
        seen = max(1, int(stats.get('seen', 0)))
        mean_score = float(stats.get('score_sum', 0.0)) / seen
        var = max(0.0, float(stats.get('score_sq_sum', 0.0)) / seen - mean_score * mean_score)
        std_score = float(np.sqrt(var))
        rows.append({
            'head': head_key,
            'seen': seen,
            'selected': int(stats.get('selected', 0)),
            'selected_ratio': float(stats.get('selected', 0.0)) / seen,
            'mean_score': mean_score,
            'std_score': std_score,
            'stability': float(1.0 / (1.0 + std_score)),
        })
    rows = sorted(rows, key=lambda x: (x['mean_score'] * x['stability'], x['selected_ratio']), reverse=True)
    top_heads = rows[:max(1, int(top_k))]
    json_path = os.path.join(output_dir, f'{model_name}_top_heads.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(top_heads, f, ensure_ascii=False, indent=2)
    metrics_csv = os.path.join(output_dir, f'{model_name}_head_metrics.csv')
    with open(metrics_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['head', 'seen', 'selected', 'selected_ratio', 'mean_score', 'std_score', 'stability'],
        )
        writer.writeheader()
        writer.writerows(rows)
    stability_csv = os.path.join(output_dir, f'{model_name}_layer_head_stability.csv')
    with open(stability_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['layer', 'head', 'seen', 'selected_ratio', 'mean_score', 'stability'])
        for row in rows:
            layer, head = row['head'].split('_')
            writer.writerow([layer, head, row['seen'], row['selected_ratio'], row['mean_score'], row['stability']])
    return {
        'json_path': json_path,
        'head_metrics_csv': metrics_csv,
        'stability_csv': stability_csv,
        'top_heads': top_heads,
    }



def high_pass_filter(image, resolusion, km=7, kh=3, sigma=None, reduce=True, block=14):
    """
    Applies a high-pass filter to an image to highlight edges and fine details.
    
    This function resizes the image, applies a Gaussian blur to create a low-frequency version,
    subtracts it from the original to get high-frequency components, and then applies median filtering.
    
    Args:
        image: Input PIL image
        resolusion: Target resolution to resize the image to
        km: Kernel size for median filtering (default: 7)
        kh: Kernel size for Gaussian blur (default: 3)
        reduce: Whether to reduce the output size using block reduction (default: True)
        
    Returns:
        h_brightness: A 2D numpy array representing the high-frequency components of the image
    """

    image = TF.resize(image, (resolusion, resolusion))
    image = TF.to_tensor(image).unsqueeze(0)
    l = TF.gaussian_blur(image, kernel_size=(kh, kh), sigma=sigma).squeeze().detach().cpu().numpy()
    h = image.squeeze().detach().cpu().numpy() - l
    h_brightness = np.sqrt(np.square(h).sum(axis=0))
    h_brightness = median_filter(h_brightness, size=km)
    if reduce:
        h_brightness = block_reduce(h_brightness, block_size=(block, block), func=np.sum)

    return h_brightness

def bbox_from_att_image_adaptive(att_map, image_size, bbox_size=336):
    """
    Generates an adaptive bounding box for original image from an attention map.
    
    This function finds the region with the highest attention in the attention map
    and creates a bounding box around it. It tries different crop ratios and selects
    the one that produces the sharpest attention difference.
    
    Args:
        att_map: A 2D numpy array representing the attention map (e.g., 24x24 for LLaVA or 16x16 for BLIP)
        image_size: Tuple of (width, height) of the original image
        bbox_size: Base size for the bounding box (default: 336)
        
    Returns:
        tuple: (x1, y1, x2, y2) coordinates of the bounding box in the original image
    """

    # the ratios corresponds to the bounding box we are going to crop the image
    ratios = [1, 1.2, 1.4, 1.6, 1.8, 2]

    max_att_poses = []
    differences = []
    block_nums = []

    for ratio in ratios:
        # perform a bbox_size*r width and bbox_size*r height crop, where bbox_size is the size of the model's original image input resolution. (336 for LLaVA, 224 for BLIP)

        # the size of each block in the attention map, in the original image
        block_size = image_size[0] / att_map.shape[1], image_size[1] / att_map.shape[0]

        # if I want a bbox_size*r width and bbox_size*r height crop from the original image, the number of blocks I need (x, y)
        block_num = min(int(bbox_size*ratio/block_size[0]), att_map.shape[1]), min(int(bbox_size*ratio/block_size[1]), att_map.shape[0])
        if att_map.shape[1]-block_num[0] < 1 and att_map.shape[0]-block_num[1] < 1:
            if ratio == 1:
                return 0, 0, image_size[0], image_size[1]
            else:
                continue
        block_nums.append((block_num[0], block_num[1]))
        
        # attention aggregation map
        sliding_att = np.zeros((att_map.shape[0]-block_num[1]+1, att_map.shape[1]-block_num[0]+1))
        max_att = -np.inf
        max_att_pos = (0, 0)

        # sliding window to find the block with the highest attention
        for x in range(att_map.shape[1]-block_num[0]+1): 
            for y in range(att_map.shape[0]-block_num[1]+1): 
                att = att_map[y:y+block_num[1], x:x+block_num[0]].sum()
                sliding_att[y, x] = att
                if att > max_att:
                    max_att = att
                    max_att_pos = (x, y)
        
        # we have the position of max attention, we can calculate the difference between the max attention and the average of its adjacent attentions, to see if it is sharp enough, the more difference, the sharper
        # we choose the best ratio r according to their attention difference
        adjcent_atts = []
        if max_att_pos[0] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0]-1])
        if max_att_pos[0] < sliding_att.shape[1]-1:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0]+1])
        if max_att_pos[1] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1]-1, max_att_pos[0]])
        if max_att_pos[1] < sliding_att.shape[0]-1:
            adjcent_atts.append(sliding_att[max_att_pos[1]+1, max_att_pos[0]])
        difference = (max_att - np.mean(adjcent_atts)) / (block_num[0] * block_num[1])
        differences.append(difference)
        max_att_poses.append(max_att_pos)
    max_att_pos = max_att_poses[np.argmax(differences)]
    block_num = block_nums[np.argmax(differences)]
    selected_bbox_size = bbox_size * ratios[np.argmax(differences)]
    
    x_center = int(max_att_pos[0] * block_size[0] + block_size[0] * block_num[0] / 2)
    y_center = int(max_att_pos[1] * block_size[1] + block_size[1] * block_num[1] / 2)
    
    x_center = selected_bbox_size//2 if x_center < selected_bbox_size//2 else x_center
    y_center = selected_bbox_size//2 if y_center < selected_bbox_size//2 else y_center
    x_center = image_size[0] - selected_bbox_size//2 if x_center > image_size[0] - selected_bbox_size//2 else x_center
    y_center = image_size[1] - selected_bbox_size//2 if y_center > image_size[1] - selected_bbox_size//2 else y_center

    x1 = max(0, x_center - selected_bbox_size//2)
    y1 = max(0, y_center - selected_bbox_size//2)
    x2 = min(image_size[0], x_center + selected_bbox_size//2)
    y2 = min(image_size[1], y_center + selected_bbox_size//2)

    return x1, y1, x2, y2

def high_res_split_threshold(image, res_threshold=512):
    """
    Splits a high-resolution image into smaller patches.
    
    This function divides a large image into smaller patches to process them individually,
    which is useful for handling high-resolution images that might be too large for direct processing.
    
    Args:
        image: Input PIL image
        res_threshold: Maximum resolution threshold before splitting (default: 1024)
        
    Returns:
        tuple: (split_images, vertical_split, horizontal_split)
            - split_images: List of PIL image patches
            - vertical_split: Number of vertical splits
            - horizontal_split: Number of horizontal splits
    """

    vertical_split = int(np.ceil(image.size[1] / res_threshold))
    horizontal_split = int(vertical_split * image.size[0] / image.size[1])

    split_num = (horizontal_split, vertical_split)
    split_size = int(np.ceil(image.size[0] / split_num[0])), int(np.ceil(image.size[1] / split_num[1]))
    
    split_images = []
    for j in range(split_num[1]):
        for i in range(split_num[0]):
            split_image = image.crop((i*split_size[0], j*split_size[1], (i+1)*split_size[0], (j+1)*split_size[1]))
            split_images.append(split_image)
    
    return split_images, vertical_split, horizontal_split

from PIL import Image
# 原图缩放为正方形
def resize_to_square(image,image_size):
    output_size=max(image_size)
    # 按最长边等比缩放
    image.thumbnail((output_size, output_size))
    # 创建正方形画布（背景可选）
    square_img = Image.new('RGB', (output_size, output_size), (255, 255, 255))
    # # 居中粘贴缩放后的图片
    square_img.paste(image, ((output_size - image.width) // 2, (output_size - image.height) // 2))
    # square_img.save('output.jpg')
    return square_img

def square_to_orin(image, image_size):
    #image_size(102,768)
    if image_size[0]>=image_size[1]:
        side_cut=int((image_size[0]-image_size[1])/2)
        cropped_img = image.crop((0, side_cut, image_size[0],image_size[0]-side_cut))
    else:
        side_cut=int((image_size[1]-image_size[0])/2)
        cropped_img = image.crop((side_cut, 0, image_size[1]-side_cut, image_size[1]))
    return cropped_img

def square_array_to_orin(array, image_size):
    #image_size(1024,768)

    if image_size[0]==image_size[1]:
        return array
    if image_size[0]==array.shape[1] and image_size[1]==array.shape[0]:
        return array
    if  image_size[0]>image_size[1]:
        if image_size[0]%2 ==0 and image_size[1]%2 ==0:
            side_cut=int((image_size[0]-image_size[1])/2)
            cropped_arr = array[side_cut:-side_cut, :]  # 裁剪掉上下各 128 行

        elif image_size[0]%2 != 0 and image_size[1]%2 != 0:
            side_cut=int((image_size[0]-image_size[1])/2)
            cropped_arr = array[side_cut:-side_cut, :]  # 裁剪掉上下各 128 行
        else:
            side_cut=int((image_size[0]-image_size[1])/2)
            if side_cut==0:
                cropped_arr = array[1:, :] 
            else:
                cropped_arr = array[side_cut+1:-side_cut, :]  # 裁剪掉上下各 128 行
    else:
        if image_size[0]%2 ==0 and image_size[1]%2 ==0:
            side_cut=int((image_size[1]-image_size[0])/2)
            cropped_arr = array[:, side_cut:-side_cut]
        elif image_size[0]%2 != 0 and image_size[1]%2 != 0:
            side_cut=int((image_size[1]-image_size[0])/2)
            cropped_arr = array[:, side_cut:-side_cut]
        else:
            side_cut=int((image_size[1]-image_size[0])/2)
            if side_cut==0:
                cropped_arr = array[:, 1:]
            else:
                cropped_arr = array[:, side_cut+1:-side_cut]
    # 为防止出现原图边长为奇数情况，导致裁剪后边长为偶数，导致最后结果不一致
    if cropped_arr.shape[0]!=image_size[0] or cropped_arr.shape[1]!=image_size[1]:
        if cropped_arr.shape[0]>image_size[1]:
            cropped_arr=cropped_arr[:image_size[1],:]
        if cropped_arr.shape[1]>image_size[0]:
            cropped_arr=cropped_arr[:,:image_size[0]]
        else:
            cropped_arr=cropped_arr[:image_size[1],:image_size[0]]
        # print("cropped_arr.shape",cropped_arr.shape)
        # print("image_size",image_size)
    return cropped_arr

#非差值缩放返回原图

# 假设arr是2D NumPy数组，target_size是目标PIL图片的尺寸（width, height）
def resize_to_pil_size(arr, target_size):
    # 调整数组大小（注意：numpy.resize会按顺序填充数据，可能失真）
    resized_arr = np.resize(arr, (target_size[1], target_size[0]))  # (height, width)
    return resized_arr

#差值缩放返回原图
from skimage.transform import resize
def resize_with_interpolation(arr, target_size):
    # 使用skimage的resize（默认插值为双线性）
    # length=min(target_size[1], target_size[0])
    resized_arr = resize(arr, (target_size[1], target_size[0]), anti_aliasing=True)
    # 转换为0-255整数并确保类型为uint8
    # resized_arr = (resized_arr * 255).astype(np.uint8)
    return resized_arr

from scipy.ndimage import gaussian_filter
def high_res(map_func, image, prompt, general_prompt, model, processor, LAYER=None, HEAD=None, res_threshold=1024):
    """
    Applies an attention mapping function to high-resolution images by splitting and recombining.
    
    This function splits a high-resolution image into smaller patches, applies the specified
    attention mapping function to each patch, and then recombines the results into a single
    attention map.
    
    Args:
        map_func: The attention mapping function to apply to each patch
        image: Input PIL image
        prompt: Text prompt for the attention function
        general_prompt: General text prompt for baseline comparison
        model: Model instance (LLaVA or BLIP)
        processor: Processor for the corresponding model
        
    Returns:
        block_att: A 2D numpy array representing the combined attention map for the entire image
    """
    
    image=resize_to_square(image,image.size)
    split_images, num_vertical_split, num_horizontal_split = high_res_split_threshold(image,res_threshold)
    att_maps = []
    for split_image in split_images:
        if LAYER is None or HEAD is None:
            att_map = map_func(split_image, prompt, general_prompt, model, processor)
        else:
            att_map = map_func(split_image, prompt, general_prompt, model, processor, LAYER, HEAD)
        # att_map = att_map / att_map.mean()
        # 差值缩放
        att_map = resize_with_interpolation(att_map,split_image.size)
        att_maps.append(att_map)
    block_att = np.block([att_maps[j:j+num_horizontal_split] for j in range(0, num_horizontal_split * num_vertical_split, num_horizontal_split)])
    # 高斯模糊
    block_att = gaussian_filter(block_att, sigma=1, mode='reflect')
    return block_att


def min_max_scale(array):
     # 检查array是否为空
    if array is None or array.size == 0:
        print(f"警告：生成的注意力图为空，数组内容: {array}")
        return None
    array = np.asarray(array, dtype=np.float64)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    min_val = np.min(array)
    max_val = np.max(array)
    denom = max_val - min_val
    if denom <= 0:
        return np.zeros_like(array, dtype=np.float64)
    return (array - min_val) / denom

def composite_attn_map(atten_maps): # 将一个注意力map列表所有位置对应相加求和，然后再归一化
    if type(atten_maps) == dict:
        atten_map = np.sum(list(atten_maps.values()), axis=0)/len(atten_maps)
    else:
        atten_map = np.sum(atten_maps, axis=0)/len(atten_maps)
    if atten_map.size == 0:
        print("Input array is empty")
    # print("Input array shape:", atten_map.shape)  # 检查形状
    att_map = min_max_scale(atten_map)
    return att_map

class Evaluation_processor:
    def __init__(self, ref_map, att_map):
        """
        初始化评估处理器
        :param ref_map: 参考标注图(2D numpy array), 值为0或1
        :param att_map: 预测关注图(2D numpy array), 值为0或1
        """
        self.ref_map = ref_map
        self.att_map = att_map
        self._validate_inputs()
    
    def _validate_inputs(self):
        """验证输入是否为2D numpy array且值仅为0或1"""
        if self.ref_map.ndim != 2 or self.att_map.ndim != 2:
            raise ValueError("输入必须是2D numpy数组")
        
        if not np.all(np.logical_or(self.ref_map == 0, self.ref_map == 1)):
            raise ValueError("ref_map必须只包含0和1")
            
        if not np.all(np.logical_or(self.att_map == 0, self.att_map == 1)):
            raise ValueError("att_map必须只包含0和1")
    # 注意力准确度
    def calculate_Entropy(self, attention_map):
        mul = np.multiply(self.ref_map, attention_map)
        sum_mul = np.sum(mul)
        sum_att = np.sum(attention_map)
        if sum_att == 0:
            return 0
        result = sum_mul/sum_att
        return result
    def calculate_metrics(self):
        """计算精确率、召回率和F1指标"""
        # 计算混淆矩阵组件
        tp = np.sum(np.logical_and(self.ref_map == 1, self.att_map == 1))
        fp = np.sum(np.logical_and(self.ref_map == 0, self.att_map == 1))
        fn = np.sum(np.logical_and(self.ref_map == 1, self.att_map == 0))
        
        # 计算精确率、召回率和F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # return {
        #     'precision': precision,
        #     'recall': recall,
        #     'f1': f1
        # }
        return [precision,recall,f1]

from .auto_threshold import *

def norm_res(map_func, image, prompt, general_prompt, model, processor, label, LAYERS, HEADS):

    image_square=resize_to_square(image,image.size)
    att_maps = map_func(image_square, prompt, general_prompt, model, processor, LAYERS, HEADS)
    temp_result={}
    eval_result={}
    for key, value in att_maps.items():
        # 插值缩放
        att_item = resize_with_interpolation(value,image_square.size)
        # 高斯模糊
        block_att = gaussian_filter(att_item, sigma=1, mode='reflect')

        right_att_map = square_array_to_orin(block_att,image.size)
        atten_map = auto_otsu(min_max_scale(right_att_map))
        temp_result[f"{key}"]  = atten_map
        # -------------------------------------------
        # temp_result = {} # Avoid out of memory
        # -------------------------------------------
        eval_processor = Evaluation_processor(label, atten_map)
        # 计算评估指标
        # eval_result[f"{key}"]  = (eval_processor.calculate_metrics()).append(eval_processor.calculate_Entropy(right_att_map))
        ce = eval_processor.calculate_Entropy(right_att_map)
        eval_result[f"{key}"]  = (eval_processor.calculate_metrics()) + [ce]
        # print(eval_result[f"{key}"])        
    return temp_result,eval_result

def specific_norm_res(map_func, image, prompt, general_prompt, model, processor, label, LAYERS, HEADS, largest_items):

    image_square=resize_to_square(image,image.size)
    att_maps = map_func(image_square, prompt, general_prompt, model, processor, LAYERS, HEADS)
 
    att_maps_list=[]

    for key, value in att_maps.items():
        if f'{key}' in largest_items:
            # 插值缩放
            att_item = resize_with_interpolation(value,image_square.size)
            # 高斯模糊
            block_att = gaussian_filter(att_item, sigma=1, mode='reflect')

            # block_att=att_item

            right_att_map = square_array_to_orin(block_att,image.size)
            # 进行归一化 
            att_map = min_max_scale(right_att_map)
            att_maps_list.append(att_map)
        else:
            continue

    attention_map=composite_attn_map(att_maps_list)
    return attention_map

# import json
# def load_json_files(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         return json.load(f)


# def rank_guassian_filter(img, kernel_size=3):
#     """
#     Apply a rank-based Gaussian-weighted filter for robust activation map denoising.

#     Parameters:
#     img : np.ndarray
#         Input 2D grayscale image.
#     kernel_size : int
#         Size of the square kernel (must be odd).

#     Returns:
#     filtered_img : np.ndarray
#         Denoised image after applying the Gaussian weighted rank filter.

#     Note:
#         The sigma (std) of is refined to coefficient of variation for robust results
#     """

#     filtered_img = np.zeros_like(img)
#     pad_width = kernel_size // 2
#     padded_img = np.pad(img, pad_width, mode='reflect')
#     ax = np.array(range(kernel_size ** 2)) - kernel_size ** 2 // 2

#     for i in range(pad_width, img.shape[0] + pad_width):
#         for j in range(pad_width, img.shape[1] + pad_width):
#             window = padded_img[i - pad_width:i + pad_width + 1,
#                                 j - pad_width:j + pad_width + 1]

#             sorted_window = np.sort(window.flatten())
#             mean = sorted_window.mean()
#             if mean > 0:
#                 sigma = sorted_window.std() / mean # std -> cov
#                 kernel = np.exp(-(ax**2) / (2 * sigma**2))
#                 kernel = kernel / np.sum(kernel)
#                 value = (sorted_window * kernel).sum()
#             else:
#                 value = 0
#             filtered_img[i - pad_width, j - pad_width] = value
    
#     return filtered_img


# import heapq

# def norm_weight_res(map_func, image, prompt, general_prompt, model, processor, LAYERS, HEADS, head_num, param_path):
   
#     param=load_json_files(param_path)
#     largest_items = heapq.nlargest(head_num, param.items(), key=lambda item: item[1])
#     largest_items = dict(largest_items)
#     # print(largest_items)
#     image=resize_to_square(image,image.size)
#     att_maps = map_func(image, prompt, general_prompt, model, processor, LAYERS, HEADS)
#     temp_result=[]
#     for key, value in att_maps.items():
#         if f'{key}' in largest_items:
#             # 插值缩放
#             # print("yes")
#             att_item = resize_with_interpolation(value,image.size)
#             # 高斯模糊
#             # block_att = gaussian_filter(att_item, sigma=1, mode='reflect')
#             block_att = rank_guassian_filter(att_item)

#             temp_result.append(block_att*largest_items[f'{key}']) 
#         else:
#             continue
        
#     stacked = np.stack(temp_result, axis=0)  # axis=0表示在第一个维度堆叠
#     # 此时stacked的形状为 (3, 2, 3)

#     # 步骤2：在第一个维度（axis=0）上计算平均值
#     mean_arr = np.mean(stacked, axis=0) 

#     # print(mean_arr.shape)
#     return mean_arr