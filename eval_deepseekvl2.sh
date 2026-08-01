#!/bin/bash
# 此文件是用来测评DeepSeek-VL-2模型
# 在vlmeval虚拟环境中启动
# 无法使用vllm（仅支持Llama4）
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
torchrun --nproc-per-node=2 run.py --data MMDU --model deepseek_vl2_small --verbose --judge exact_matching 