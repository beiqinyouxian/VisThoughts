#!/bin/bash
# 此文件用于测评 Qwen2.5-VL-Enhance-7B-Instruct 模型
# 聚焦主题：图像【细粒度推理】+【视觉问答 VQA】相关基准
# 在 vlmeval 虚拟环境中启动
#
# 说明：下列数据集均可用 `--judge exact_matching`（规则/启发式评测，无需 GPT API）。
#       可按需注释或删减任意一行来控制本次评测的范围与耗时。

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

DATA=""
# DATA="$DATA RefCOCO TreeBench"                                   # 视觉定位 / grounding（原有）
DATA="$DATA TreeBench"   
# DATA="$DATA VStarBench HRBench4K HRBench8K MME-RealWorld-Lite"   # 高分辨率 / 小目标细粒度感知
DATA="$DATA VStarBench" 
# DATA="$DATA CV-Bench-2D CV-Bench-3D BLINK MMStar MMVP"           # 视觉中心细粒度推理
DATA="$DATA BLINK MMStar MMVP"           # 视觉中心细粒度推理
# DATA="$DATA RealWorldQA"                                         # 真实世界细粒度 VQA
# DATA="$DATA TallyQA CountBenchQA CRPE_EXIST CRPE_RELATION"       # 计数 / 属性 / 空间关系
DATA="$DATA POPE HallusionBench"                                 # 细粒度幻觉 / 区分
DATA="$DATA OCRBench"                                            # OCR 细粒度

torchrun --nproc-per-node=4 --master_port=29501 run.py \
  --data $DATA \
  --model Qwen2.5-VL-7B-Instruct \
  --judge exact_matching
