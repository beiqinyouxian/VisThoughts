#!/bin/bash
# 此文件用 LMDeploy 加速推理来测评 Qwen-VL 模型（默认 Qwen2.5-VL-7B-Instruct）
# 聚焦主题：图像【细粒度推理】+【视觉问答 VQA】相关基准，与 eval_qwenvl.sh 同一套指标
# 在 vlmeval 虚拟环境中启动
#
# 启动方式参考官方文档：
#   https://github.com/open-compass/VLMEvalKit/blob/main/docs/zh-CN/EvalByLMDeploy.md
#
# 与 eval_qwenvl.sh 的区别：
#   - eval_qwenvl.sh 使用 torchrun 在本地直接加载 HF 权重做推理。
#   - 本脚本先用 `lmdeploy serve api_server` 起一个 OpenAI 兼容服务，
#     再让 run.py 通过 `--base-url` 走 LMDeployAPI 调用该服务（无需改 config.py）。
#
# 说明：下列数据集均可用 `--judge exact_matching`（规则/启发式评测，无需 GPT API）。
#       可按需注释或删减任意一行来控制本次评测的范围与耗时。

set -e

# ============ 可配置参数 ============
MODEL_NAME=${MODEL_NAME:-"Qwen2-VL-7B-Instruct"}        # run.py 的 --model，需与下面 --model-name 一致
MODEL_PATH=${MODEL_PATH:-"/mnt/data/VLM/Qwen2-VL-7B-Instruct"}  # 本地权重路径
SERVER_PORT=${SERVER_PORT:-23333}                         # LMDeploy 服务端口
TP=${TP:-4}                                               # 张量并行 / 使用的 GPU 数，模型小于7B时使用TP1
API_NPROC=${API_NPROC:-64}                                # run.py 并发请求数
GPUS=${GPUS:-"0,1,2,3"}                                         # 可见 GPU，例如 "0,1,2,3"
START_TIMEOUT=${START_TIMEOUT:-600}                       # 等待服务就绪的最长秒数
CACHE_MAX_ENTRY=${CACHE_MAX_ENTRY:-0.8}                   # KV Cache 显存占比（TurboMind：权重外剩余空闲显存的比例）
# ===================================

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=${GPUS}

API_BASE="http://0.0.0.0:${SERVER_PORT}/v1"

# RefCOCO 可选粒度：
#   全部 8 个 split:  RefCOCO
#   子数据集:        RefCOCO_only / RefCOCO+ / RefCOCOg
#   单个 split:      RefCOCO_val / RefCOCO_testA / RefCOCO_testB /
#                    RefCOCO+_val / RefCOCO+_testA / RefCOCO+_testB /
#                    RefCOCOg_val / RefCOCOg_test
DATA="RefCOCO"


# DATA="$DATA TreeBench"

# DATA="$DATA VStarBench"

# DATA="$DATA BLINK MMStar MMVP"                                  # 视觉中心细粒度推理
# DATA="$DATA RealWorldQA"                                         # 真实世界细粒度 VQA

# DATA="$DATA POPE HallusionBench"                                # 细粒度幻觉 / 区分
# DATA="$DATA OCRBench"                                           # OCR 细粒度
# DATA="$DATA MME"


# 备选
# # DATA="$DATA HRBench4K HRBench8K MME-RealWorld-Lite"   # 高分辨率 / 小目标细粒度感知
# # DATA="$DATA TallyQA CountBenchQA CRPE_EXIST CRPE_RELATION"       # 计数 / 属性 / 空间关系
# # DATA="$DATA CV-Bench-2D CV-Bench-3D"           # 视觉中心细粒度推理

# ============ 第 0 步：检查 lmdeploy ============
if ! python -c "import lmdeploy" 2>/dev/null; then
  echo "[ERROR] 未检测到 lmdeploy，请先安装：pip install lmdeploy"
  exit 1
fi

# ============ 第 1 步：启动 LMDeploy 推理服务 ============
# --model-name 必须与 run.py 的 --model 一致，VLMEvalKit 会用它选择合适的 prompt 构建策略。
echo "[INFO] 启动 LMDeploy 服务：${MODEL_PATH} (port=${SERVER_PORT}, tp=${TP}, gpus=${GPUS})"
lmdeploy serve api_server "${MODEL_PATH}" \
  --model-name "${MODEL_NAME}" \
  --server-port "${SERVER_PORT}" \
  --tp "${TP}" \
  --cache-max-entry-count "${CACHE_MAX_ENTRY}" \
  > "lmdeploy_${MODEL_NAME}.log" 2>&1 &
SERVER_PID=$!

# 退出时自动关闭服务
cleanup() {
  echo "[INFO] 关闭 LMDeploy 服务 (PID=${SERVER_PID})"
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# ============ 等待服务就绪 ============
echo "[INFO] 等待服务就绪（最长 ${START_TIMEOUT}s），日志见 lmdeploy_${MODEL_NAME}.log ..."
for ((i=0; i<START_TIMEOUT; i+=5)); do
  if curl -s "http://0.0.0.0:${SERVER_PORT}/v1/models" >/dev/null 2>&1; then
    echo "[INFO] 服务已就绪。"
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[ERROR] LMDeploy 服务进程已退出，请查看 lmdeploy_${MODEL_NAME}.log"
    exit 1
  fi
  sleep 5
done

# ============ 第 2 步：评测 ============
echo "[INFO] 开始评测：model=${MODEL_NAME}, data=${DATA}"
python run.py \
  --data $DATA \
  --model "${MODEL_NAME}" \
  --base-url "${API_BASE}" \
  --api-nproc "${API_NPROC}" \
  --verbose \
  --judge exact_matching
