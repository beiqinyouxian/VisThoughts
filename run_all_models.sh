#!/usr/bin/env bash
# ============================================================
# 多模型串行注意力头挖掘样本敏感性分析脚本
#
# 功能：依次对多个 VLMEvalKit 增强模型运行
#       run_head_mining_sample_analysis.py，
#       通过 sed 动态替换脚本中的 MODEL_NAME 实现切换。
#
# 运行方式：
#   1. 单模型运行：
#      GPU_GROUPS="0;1;2;3" torchrun --nproc_per_node=4 --master-port=29503 \
#          run_head_mining_sample_analysis.py
#
#   2. 批量运行（本脚本）：
#      bash run_all_models.sh
#      或
#      ./run_all_models.sh
#
# 配置项：
#   GPU_GROUPS       - 按分号分隔的 GPU 分组，每组数量和 nproc_per_node 一致
#   NPROC_PER_NODE   - 每节点进程数，必须等于 GPU_GROUPS 的分组数
#   MASTER_PORT      - torchrun 主节点端口（同一机器多任务需不同端口）
#
# 注意事项：
#   - 脚本退出时会自动恢复 Python 脚本中原始的 MODEL_NAME 行
#   - 任一模型运行失败（退出码非 0）会中止后续模型
# ============================================================

set -euo pipefail

# echo "脚本将在 3 小时后启动（开始等待：$(date)）"
# sleep 3h
# echo "等待结束，开始运行（$(date)）"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/run_head_mining_sample_analysis.py"

# ========== 模型列表 ==========
MODELS=(
    # "Qwen2.5-VL-Enhance-7B-Instruct"
    # "Qwen2.5-VL-Enhance-3B-Instruct"
    # "Qwen2-VL-Enhance-7B-Instruct"
    # "Qwen2-VL-Enhance-2B-Instruct"
    # "LLaVA-Enhance-1.5-7B"
    "LLaVA-Enhance-1.5-13B"
)

# ========== torchrun 配置 ==========
GPU_GROUPS="0;1;2;3"
NPROC_PER_NODE=4
MASTER_PORT=29501

# ========== 备份原始 MODEL_NAME 行 ==========
ORIG_LINE_NUM=84
ORIG_LINE=$(sed -n "${ORIG_LINE_NUM}p" "${PY_SCRIPT}")

restore_original() {
    sed -i "${ORIG_LINE_NUM}s/.*/${ORIG_LINE}/" "${PY_SCRIPT}"
}
trap restore_original EXIT

# ========== 逐模型运行 ==========
TOTAL=${#MODELS[@]}
echo "=========================================="
echo " 共 ${TOTAL} 个模型待运行"
echo "=========================================="

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    IDX=$((i + 1))

    echo ""
    echo "=========================================="
    echo " [$IDX/${TOTAL}] 开始运行模型: ${MODEL}"
    echo "=========================================="

    # 替换 MODEL_NAME 行
    sed -i "${ORIG_LINE_NUM}s/.*/MODEL_NAME = \"${MODEL}\"/" "${PY_SCRIPT}"

    echo "  -> GPU_GROUPS=${GPU_GROUPS}"
    echo "  -> torchrun --nproc_per_node=${NPROC_PER_NODE} --master-port=${MASTER_PORT}"

    GPU_GROUPS="${GPU_GROUPS}" torchrun \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master-port="${MASTER_PORT}" \
        "${PY_SCRIPT}"

    RET=$?
    if [ $RET -ne 0 ]; then
        echo " [ERROR] 模型 ${MODEL} 以退出码 ${RET} 失败，中止后续模型。"
        exit $RET
    fi

    echo " [$IDX/${TOTAL}] 模型 ${MODEL} 运行完成。"
done

echo ""
echo "=========================================="
echo " 全部 ${TOTAL} 个模型运行完毕！"
echo "=========================================="
