#!/bin/bash
# 此文件用于测评 Qwen3-VL-8B-Instruct 模型
# 聚焦主题：图像【细粒度推理】+【视觉问答 VQA】相关基准
# 在 vlmeval 虚拟环境中启动
#
# 说明：下列数据集均可用 `--judge exact_matching`（规则/启发式评测，无需 GPT API）。
#       与 eval_qwenvl.sh 使用同一套基准，便于两个模型横向对比。
#       可按需注释或删减任意一行来控制本次评测的范围与耗时。

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

DATA=""
DATA="$DATA VStarBench"                                          # V* 高分辨率细粒度视觉搜索（原有）
DATA="$DATA RefCOCO TreeBench"                                   # 视觉定位 / grounding
DATA="$DATA HRBench4K HRBench8K MME-RealWorld-Lite"             # 高分辨率 / 小目标细粒度感知
DATA="$DATA CV-Bench-2D CV-Bench-3D BLINK MMStar MMVP"           # 视觉中心细粒度推理
DATA="$DATA RealWorldQA"                                         # 真实世界细粒度 VQA
DATA="$DATA TallyQA CountBenchQA CRPE_EXIST CRPE_RELATION"       # 计数 / 属性 / 空间关系
DATA="$DATA POPE HallusionBench"                                 # 细粒度幻觉 / 区分
DATA="$DATA OCRBench"                                            # OCR 细粒度

torchrun --nproc-per-node=2 --master_port=29502 run.py \
  --data $DATA \
  --model Qwen3-VL-8B-Instruct \
  --verbose \
  --judge exact_matching
