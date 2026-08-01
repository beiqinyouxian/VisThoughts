#!/usr/bin/env bash
# ============================================================
# 多模型 × 多数据集 串行全量推理脚本
#
# 功能：依次对每个 (MODEL, DATASET) 组合调用 run_fullset_inference.py。
#       通过 sed 动态替换脚本中的 MODEL_NAME / DATASET_NAMES。
#       默认 HEAD_SELECTION_MODE=inference，复用已固化的 top_heads
#       （MME N=200：outputs/selected_heads/mme_n200/ → temp_debug/head_stats/）。
#
# 用法：
#   bash run_all_fullset_inference.sh
#   MASTER_PORT=29521 GPU_GROUPS="4;5;6;7" bash run_all_fullset_inference.sh
#
# 并行跑 LLaVA / Qwen 两套环境时，必须同时使用不同端口 + 不重叠 GPU，例如：
#   终端 A (LLaVA):  MASTER_PORT=29511 GPU_GROUPS="0;1;2;3" bash run_all_fullset_inference.sh
#   终端 B (Qwen):   MASTER_PORT=29521 GPU_GROUPS="4;5;6;7" bash run_all_fullset_inference.sh
#
# 可选环境变量：
#   GPU_GROUPS / NPROC_PER_NODE / MASTER_PORT / REASONING_MODE
#   CONTINUE_ON_ERROR=1  某组合失败后继续后续组合（默认失败即中止）
#
# 注意：
#   - 退出时自动恢复 Python 脚本中的 MODEL_NAME / DATASET_NAMES
#   - 非 MME 数据集需先有对应 *_{dataset}_top_heads.json，否则会回退默认头
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/run_fullset_inference.py"
cd "${SCRIPT_DIR}"

# ========== 模型列表（与 head mining / selected_heads 对齐）==========
MODELS=(
    # "LLaVA-Enhance-1.5-7B"
    # "LLaVA-Enhance-1.5-13B"
    "Qwen2-VL-Enhance-7B-Instruct"
    "Qwen2.5-VL-Enhance-7B-Instruct"
    "Qwen2-VL-Enhance-2B-Instruct"
    "Qwen2.5-VL-Enhance-3B-Instruct"
)

# ========== 数据集列表（可按需增删）==========
DATASETS=(
    "MME"
    # "POPE"
)

# ========== torchrun / 推理配置 ==========
GPU_GROUPS="${GPU_GROUPS:-0;1;2;3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29511}"
REASONING_MODE="${REASONING_MODE:-two_stage}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GPU_GROUPS

# 启动前检查目标 GPU 是否已被其他任务占用（常见于 LLaVA/Qwen 并行跑时 GPU 重叠）
if command -v nvidia-smi >/dev/null 2>&1; then
    IFS=';' read -ra _GPU_GROUP_ARR <<< "${GPU_GROUPS}"
    if [[ ${#_GPU_GROUP_ARR[@]} -ne ${NPROC_PER_NODE} ]]; then
        echo "[ERROR] GPU_GROUPS 分组数 (${#_GPU_GROUP_ARR[@]}) 必须等于 NPROC_PER_NODE (${NPROC_PER_NODE})" >&2
        exit 1
    fi
    _MEM_THRESHOLD_MB=2048
    for _g in "${_GPU_GROUP_ARR[@]}"; do
        _g="${_g// /}"
        _first="${_g%%,*}"
        _used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${_first}" 2>/dev/null | tr -d ' ')
        if [[ -n "${_used}" && "${_used}" -gt "${_MEM_THRESHOLD_MB}" ]]; then
            echo "[WARN] GPU ${_first} 已占用 ${_used} MiB（>${_MEM_THRESHOLD_MB} MiB）。并行任务可能 GPU 冲突导致 OOM。" >&2
            echo "       请确认 LLaVA/Qwen 使用不重叠 GPU，例如 0-3 vs 4-7。" >&2
        fi
    done
fi

# ========== 备份并在退出时恢复 MODEL_NAME / DATASET_NAMES ==========
ORIG_MODEL_LINE=$(grep -n '^MODEL_NAME = ' "${PY_SCRIPT}" | head -1)
ORIG_DATASET_LINE=$(grep -n '^DATASET_NAMES = ' "${PY_SCRIPT}" | head -1)
ORIG_REASONING_LINE=$(grep -n '^REASONING_MODE = ' "${PY_SCRIPT}" | head -1)

if [[ -z "${ORIG_MODEL_LINE}" || -z "${ORIG_DATASET_LINE}" ]]; then
    echo "[ERROR] 无法在 ${PY_SCRIPT} 中定位 MODEL_NAME / DATASET_NAMES" >&2
    exit 1
fi

MODEL_LINE_NUM="${ORIG_MODEL_LINE%%:*}"
DATASET_LINE_NUM="${ORIG_DATASET_LINE%%:*}"
REASONING_LINE_NUM="${ORIG_REASONING_LINE%%:*}"
ORIG_MODEL_TEXT="${ORIG_MODEL_LINE#*:}"
ORIG_DATASET_TEXT="${ORIG_DATASET_LINE#*:}"
ORIG_REASONING_TEXT="${ORIG_REASONING_LINE#*:}"

restore_original() {
    sed -i "${MODEL_LINE_NUM}s|.*|${ORIG_MODEL_TEXT}|" "${PY_SCRIPT}"
    sed -i "${DATASET_LINE_NUM}s|.*|${ORIG_DATASET_TEXT}|" "${PY_SCRIPT}"
    if [[ -n "${REASONING_LINE_NUM}" ]]; then
        sed -i "${REASONING_LINE_NUM}s|.*|${ORIG_REASONING_TEXT}|" "${PY_SCRIPT}"
    fi
}
trap restore_original EXIT

TOTAL_MODELS=${#MODELS[@]}
TOTAL_DATASETS=${#DATASETS[@]}
TOTAL_COMBOS=$((TOTAL_MODELS * TOTAL_DATASETS))
FAILED=0
DONE=0

echo "=========================================="
echo " 全量推理矩阵：${TOTAL_MODELS} 模型 × ${TOTAL_DATASETS} 数据集 = ${TOTAL_COMBOS} 组合"
echo " GPU_GROUPS=${GPU_GROUPS}  nproc=${NPROC_PER_NODE}  port=${MASTER_PORT}"
echo " REASONING_MODE=${REASONING_MODE}"
echo " PY_SCRIPT=${PY_SCRIPT}"
echo "=========================================="

for mi in "${!MODELS[@]}"; do
    MODEL="${MODELS[$mi]}"
    for di in "${!DATASETS[@]}"; do
        DATASET="${DATASETS[$di]}"
        DONE=$((DONE + 1))

        echo ""
        echo "=========================================="
        echo " [${DONE}/${TOTAL_COMBOS}] MODEL=${MODEL}  DATASET=${DATASET}"
        echo "=========================================="

        sed -i "${MODEL_LINE_NUM}s|.*|MODEL_NAME = \"${MODEL}\"|" "${PY_SCRIPT}"
        sed -i "${DATASET_LINE_NUM}s|.*|DATASET_NAMES = [\"${DATASET}\"]|" "${PY_SCRIPT}"
        if [[ -n "${REASONING_LINE_NUM}" ]]; then
            sed -i "${REASONING_LINE_NUM}s|.*|REASONING_MODE = \"${REASONING_MODE}\"|" "${PY_SCRIPT}"
        fi

        set +e
        GPU_GROUPS="${GPU_GROUPS}" torchrun \
            --nproc_per_node="${NPROC_PER_NODE}" \
            --master-port="${MASTER_PORT}" \
            "${PY_SCRIPT}"
        RET=$?
        set -e

        if [[ $RET -ne 0 ]]; then
            echo " [ERROR] ${MODEL} + ${DATASET} 失败 (exit=${RET})"
            FAILED=$((FAILED + 1))
            if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
                echo " 中止后续组合（设 CONTINUE_ON_ERROR=1 可跳过失败继续）。"
                exit "${RET}"
            fi
        else
            echo " [OK] ${MODEL} + ${DATASET} 完成"
        fi
    done
done

echo ""
echo "=========================================="
if [[ ${FAILED} -eq 0 ]]; then
    echo " 全部 ${TOTAL_COMBOS} 个组合运行完毕。"
else
    echo " 完成，但有 ${FAILED}/${TOTAL_COMBOS} 个组合失败。"
    exit 1
fi
echo "=========================================="
