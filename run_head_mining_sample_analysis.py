"""注意力头挖掘样本规模敏感性实验。

Layer 1 - Head Selection Stability:
  比较不同挖掘样本数 N 下选出的 top-k 头集合与排名，相对 REFERENCE_N 的稳定性。

Layer 2 - Downstream Performance Stability:
  固定各 N 挖掘出的 heads，在分层采样的固定评测子集上推理并评测（MME 14 类齐全）。

清除历史数据（完全重跑时使用）：
  本脚本所有产出均在 ./outputs/head_mining_analysis/ 下，包含三处：
    head_stats/  - Layer1 各 N 挖掘统计、top heads、挖掘行索引缓存
    infer/       - Layer2 推理预测文件（含多进程 rank 分片）
    reports/     - Layer1/Layer2 分析报告、全局挖掘/评测划分与评测子集缓存
  一键清除全部历史结果：
    rm -rf ./outputs/head_mining_analysis
  注意：修改挖掘池策略后需清除旧缓存（MINE_ROW_LIST_VERSION / MINING_EVAL_SPLIT_VERSION 变更时）

运行方式：
  单进程多卡（推荐，一份模型分片到全部 GPU）：
    CUDA_VISIBLE_DEVICES=0,1,2,3 python run_head_mining_sample_analysis.py

  多进程多卡（每进程 1 或多卡；mine/eval 均在同一 N 内按样本切分到各 rank，负载均衡）：
    # 4 卡，2 进程 × 2 卡
    GPU_GROUPS="0,1;2,3" torchrun --nproc_per_node=2 --master-port=29501 run_head_mining_sample_analysis.py
    # 4 卡，4 进程 × 1 卡
    GPU_GROUPS="0;1;2;3" torchrun --nproc_per_node=4 --master-port=29502 run_head_mining_sample_analysis.py
    # 同一机器并行跑多个 torchrun 任务时，必须为每个任务指定不同的 master 端口（默认 29500）
    GPU_GROUPS="0;1;2;3" torchrun --nproc_per_node=4 --master-port=29503 run_head_mining_sample_analysis.py
    # 8 卡，4 进程 × 2 卡
    GPU_GROUPS="0,1;2,3;4,5;6,7" torchrun --nproc_per_node=4 run_head_mining_sample_analysis.py
    # 8 卡，8 进程 × 1 卡
    GPU_GROUPS="0;1;2;3;4;5;6;7" torchrun --nproc_per_node=8 run_head_mining_sample_analysis.py
    # nproc_per_node 必须等于 GPU_GROUPS 的分组数
    # mine 仅单进程时也可用：CUDA_VISIBLE_DEVICES=0,1,2,3 python ...（单进程多卡，模型自动分片）

    多卡 barrier 超时：见脚本顶部 DIST_INIT_TIMEOUT_SEC / TORCH_NCCL_BLOCKING_WAIT
"""
from __future__ import annotations

import csv
import gc
import json
import math
import os
from datetime import timedelta
from pathlib import Path

# 多卡 collective 超时（秒）。慢 rank 推理过久时，其他 rank 在 barrier 等待；PyTorch 默认 600 秒易超时。
DIST_INIT_TIMEOUT_SEC = 2100
# 阻塞等待 NCCL 完成（布尔开关，仅接受 0/1；勿把超时秒数写在这里）
TORCH_NCCL_BLOCKING_WAIT = "1"

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", TORCH_NCCL_BLOCKING_WAIT)


def _setup_gpu_binding():
    """双进程四卡：每个进程只可见 2 张卡，避免 device_map=auto 跨进程占满全部 GPU。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    groups = os.environ.get("GPU_GROUPS", "0,1;2,3").split(";")
    if local_rank < len(groups):
        os.environ["CUDA_VISIBLE_DEVICES"] = groups[local_rank].strip()
        print(f"[dist] LOCAL_RANK={local_rank} -> CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")


_setup_gpu_binding()

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from scipy.stats import kendalltau, spearmanr
from tabulate import tabulate
from tqdm.auto import tqdm

from vlmeval.config import supported_VLM
from vlmeval.dataset import build_dataset
from vlmeval.smp import dump, get_pred_file_path, load

# =============== 参数区（按需修改）===============
MODEL_NAME = "LLaVA-Enhance-1.5-13B"
MODEL_SLUG = MODEL_NAME.lower().replace(".", "_").replace("-", "_").replace(" ", "_")
DATASET_NAMES = ["MME"]
SAMPLE_SIZES = [20, 50, 100, 150, 200, 300, 500]
REFERENCE_N = 500
AUTO_HEAD_TOP_K = 3

HEAD_STATS_ROOT = "./outputs/head_mining_analysis/head_stats"
WORK_DIR = "./outputs/head_mining_analysis/infer"
ANALYSIS_DIR = "./outputs/head_mining_analysis/reports"

# 固定评测子集（与挖掘样本不重叠；全局分层划分时由 split 自动保证）
EVAL_START_INDEX = 500              # 仅 EVAL_STRATIFIED=False 时用于 iloc 后缀切分
EVAL_INDEX_LIST: list | None = None  # 非空时优先，跳过自动构造

# Layer2 分层评测集：每类全局预留若干 image 作评测，其余行进入挖掘池
EVAL_STRATIFIED = True
EVAL_IMAGES_PER_CATEGORY = 10       # 每类抽多少张图（每图通常 2 条问答，保留完整 image_path 组）
EVAL_RANDOM_SEED = 42
EVAL_SAVE_INDEX_LIST = True         # 保存到 ANALYSIS_DIR，保证多次运行评测集一致
EVAL_USE_INDEX_CACHE = True         # 若已保存则直接加载
STRATIFIED_EVAL_VERSION = 3         # 评测集构造逻辑版本，变更后自动重建缓存

# MME 官方评测要求的 14 个子类（与 vlmeval/dataset/utils/yorn.py 一致）
MME_CATEGORIES = [
    "OCR", "artwork", "celebrity", "color", "count", "existence",
    "landmark", "position", "posters", "scene",
    "code_reasoning", "commonsense_reasoning", "numerical_calculation", "text_translation",
]

RUN_PHASES = ["mine", "analyze", "eval"]  # 可分阶段：["mine", "analyze", "eval"]
SKIP_EXISTING_MINING = True

# 第一阶段挖掘：按子类在全局挖掘池内分层抽样（不再使用 iloc[:N] 前缀池）
MINE_STRATIFIED = True
MINE_STRATIFY_MODE = "uniform"       # uniform=每类均分 | proportional=按类内样本量比例
MINE_RANDOM_SEED = 42
MINE_SAVE_ROW_LIST = True            # 保存每个 N 的挖掘行 index，便于复现
MINE_ROW_LIST_VERSION = 2            # 挖掘行列表版本（全局池策略变更后需递增）
MINING_EVAL_SPLIT_VERSION = 1        # 挖掘/评测全局划分缓存版本

# Layer1 稳定判据
LAYER1_JACCARD_THRESH = 0.9
LAYER1_KENDALL_THRESH = 0.95

# Layer2 稳定判据：主指标相对 REFERENCE_N 绝对变化小于该阈值（百分点）
LAYER2_DELTA_THRESH = 0.5
LAYER2_PRIMARY_METRICS = ["perception", "reasoning"]  # MME；其他数据集自动取数值列

JUDGE_KWARGS = {"nproc": 4, "verbose": False}


def init_distributed() -> tuple[int, int]:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(seconds=DIST_INIT_TIMEOUT_SEC),
        )
    return rank, world_size


def barrier(world_size: int):
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.barrier()


def partition_by_rank(items: list, rank: int, world_size: int) -> list:
    return [item for i, item in enumerate(items) if i % world_size == rank]


def _set_head_mode(model, mode: str):
    if hasattr(model, "head_selection_mode"):
        model.head_selection_mode = mode
    if hasattr(model, "auto_head_mining"):
        model.auto_head_mining = mode == "mine"


def _reset_head_profile(model):
    if hasattr(model, "head_mining_profile"):
        model.head_mining_profile = {}
    if hasattr(model, "_loaded_head_profile_scopes"):
        model._loaded_head_profile_scopes = set()


def build_model(head_stats_dir: str | None = None):
    if int(os.environ.get("RANK", 0)) == 0:
        print(f"Loading model: {MODEL_NAME}")
    # 避免 transformers 在 torchrun 下误启 TP，与 VLMEvalKit inference 保持一致。
    ws_bak = os.environ.pop("WORLD_SIZE", None)
    model = supported_VLM[MODEL_NAME](verbose=False)
    if ws_bak:
        os.environ["WORLD_SIZE"] = ws_bak

    if hasattr(model, "reasoning_mode"):
        model.reasoning_mode = "two_stage_attention"

    for attr in ("save_heatmap_overlay", "save_per_keyword_overlay", "save_attention_debug"):
        if hasattr(model, attr):
            setattr(model, attr, False)
    if hasattr(model, "attention_debug_dir"):
        model.attention_debug_dir = None

    if hasattr(model, "auto_head_top_k"):
        model.auto_head_top_k = int(AUTO_HEAD_TOP_K)
    if hasattr(model, "auto_head_stats_dir"):
        model.auto_head_stats_dir = head_stats_dir or HEAD_STATS_ROOT

    return model


def get_scope_name(model, dataset_name: str) -> str:
    if hasattr(model, "_profile_scope_name"):
        return model._profile_scope_name(dataset_name)
    # fallback
    slug = str(dataset_name).strip().lower()
    return f"qwen2_7b__{slug}"


def _eval_index_cache_path(dataset_name: str) -> Path:
    return Path(ANALYSIS_DIR) / f"{MODEL_SLUG}_{dataset_name}_stratified_eval_indices.json"


def _mining_eval_split_cache_path(dataset_name: str) -> Path:
    return Path(ANALYSIS_DIR) / f"{MODEL_SLUG}_{dataset_name}_mining_eval_split.json"


def _required_eval_categories(dataset_name: str) -> list[str] | None:
    name = str(dataset_name).upper()
    if name == "MME" or (name.endswith("MME") and "MMESCI" not in name):
        return list(MME_CATEGORIES)
    return None


def _attach_row_positions(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["_row_pos"] = np.arange(len(out))
    return out


def _sample_images_from_pool(
    cat_pool: pd.DataFrame,
    rng: np.random.Generator,
    n_images: int,
) -> tuple[list, int]:
    image_paths = cat_pool["image_path"].unique()
    n_sample = min(int(n_images), len(image_paths))
    chosen_paths = rng.choice(image_paths, size=n_sample, replace=False)
    rows = cat_pool[cat_pool["image_path"].isin(chosen_paths)]
    return rows["index"].tolist(), n_sample


def _fallback_category_holdout_images(
    cat_all: pd.DataFrame,
    rng: np.random.Generator,
    n_images: int,
    mining_cutoff: int,
) -> tuple[list, int, str]:
    """类别整体落在挖掘前缀区时，从该类靠后的 image 中留出评测（可能与 iloc[:N] 挖掘重叠）。"""
    in_mining_zone = cat_all[cat_all["_row_pos"] < mining_cutoff]
    if in_mining_zone.empty:
        return [], 0, "empty"

    img_min_pos = in_mining_zone.groupby("image_path")["_row_pos"].min().sort_values()
    paths = img_min_pos.index.to_numpy()
    # 优先从类别后半段 image 中随机抽，降低与 iloc[:小N] 挖掘的重叠概率
    tail_start = max(0, len(paths) // 2)
    tail_paths = paths[tail_start:]
    if len(tail_paths) == 0:
        tail_paths = paths
    n_sample = min(int(n_images), len(tail_paths))
    chosen_paths = rng.choice(tail_paths, size=n_sample, replace=False)
    rows = cat_all[cat_all["image_path"].isin(chosen_paths)]
    return rows["index"].tolist(), n_sample, "category_tail_holdout"


def _build_global_mining_eval_split(
    data: pd.DataFrame,
    dataset_name: str,
) -> tuple[set, set, dict]:
    """按类别全局划分：每类预留评测 image，其余 QA 行进入挖掘池（与 row_pos 无关）。"""
    data = _attach_row_positions(data)
    required_cats = _required_eval_categories(dataset_name)
    if not required_cats:
        if "category" in data.columns:
            required_cats = sorted(data["category"].dropna().unique().tolist())
        else:
            required_cats = []

    if not required_cats or "category" not in data.columns:
        eval_row_start = int(max(EVAL_START_INDEX, max(SAMPLE_SIZES)))
        eval_rows = data[data["_row_pos"] >= eval_row_start]
        mining_rows = data[data["_row_pos"] < eval_row_start]
        return (
            set(eval_rows["index"].tolist()),
            set(mining_rows["index"].tolist()),
            {"mode": "prefix_fallback", "eval_row_start": eval_row_start},
        )

    if "image_path" not in data.columns:
        raise ValueError("Global stratified split requires 'image_path' column in dataset.")

    rng = np.random.default_rng(EVAL_RANDOM_SEED)
    eval_indices: set = set()
    mining_indices: set = set()
    per_cat_stats = {}

    for cat in required_cats:
        cat_all = data[data["category"] == cat]
        if cat_all.empty:
            raise ValueError(f"Category '{cat}' not found in dataset.")

        image_paths = cat_all["image_path"].unique()
        shuffled = rng.permutation(image_paths)
        n_eval_images = min(int(EVAL_IMAGES_PER_CATEGORY), len(shuffled))
        eval_paths = set(shuffled[:n_eval_images])
        eval_rows = cat_all[cat_all["image_path"].isin(eval_paths)]
        mining_rows = cat_all[~cat_all["image_path"].isin(eval_paths)]

        if mining_rows.empty:
            raise ValueError(
                f"Category '{cat}' has no mining rows after reserving "
                f"{n_eval_images} eval images; lower EVAL_IMAGES_PER_CATEGORY."
            )

        eval_indices.update(eval_rows["index"].tolist())
        mining_indices.update(mining_rows["index"].tolist())
        per_cat_stats[cat] = {
            "eval_images": int(n_eval_images),
            "eval_rows": len(eval_rows),
            "mining_images": int(mining_rows["image_path"].nunique()),
            "mining_rows": len(mining_rows),
        }

    return eval_indices, mining_indices, {
        "mode": "global_stratified",
        "images_per_category": int(EVAL_IMAGES_PER_CATEGORY),
        "random_seed": int(EVAL_RANDOM_SEED),
        "per_category": per_cat_stats,
    }


def resolve_mining_eval_split(dataset, dataset_name: str = "") -> tuple[set, set, dict]:
    """解析或构建全局挖掘/评测划分（结果缓存到 ANALYSIS_DIR）。"""
    cache_path = _mining_eval_split_cache_path(dataset_name)
    if EVAL_USE_INDEX_CACHE and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as fin:
            payload = json.load(fin)
        if payload.get("version") == MINING_EVAL_SPLIT_VERSION:
            eval_indices = set(payload.get("eval_indices", []))
            mining_indices = set(payload.get("mining_indices", []))
            stats = payload.get("stats", {})
            if eval_indices and mining_indices:
                print(
                    f"[split] loaded global mining/eval split from cache: {cache_path} "
                    f"(eval={len(eval_indices)}, mining_pool={len(mining_indices)})"
                )
                return eval_indices, mining_indices, stats
        print(f"[split] ignore stale mining/eval split cache (version mismatch): {cache_path}")

    eval_indices, mining_indices, stats = _build_global_mining_eval_split(dataset.data, dataset_name)
    if EVAL_SAVE_INDEX_LIST:
        os.makedirs(ANALYSIS_DIR, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fout:
            json.dump({
                "version": MINING_EVAL_SPLIT_VERSION,
                "dataset": dataset_name,
                "eval_indices": sorted(eval_indices),
                "mining_indices": sorted(mining_indices),
                "stats": stats,
            }, fout, indent=2, ensure_ascii=False)
        print(f"[split] saved global mining/eval split: {cache_path}")

    print(
        f"[split] global mining/eval split: eval={len(eval_indices)} rows, "
        f"mining_pool={len(mining_indices)} rows ({stats.get('mode', 'unknown')})"
    )
    for cat, cat_stat in stats.get("per_category", {}).items():
        print(
            f"  - {cat}: eval {cat_stat['eval_images']} images / {cat_stat['eval_rows']} rows, "
            f"mining pool {cat_stat['mining_images']} images / {cat_stat['mining_rows']} rows"
        )
    return eval_indices, mining_indices, stats


def _build_stratified_eval_indices(
    data: pd.DataFrame,
    required_cats: list[str],
    eval_row_start: int,
    mining_cutoff: int,
) -> tuple[list, dict]:
    """Legacy：按 row_pos 后缀构造评测集（EVAL_STRATIFIED=False 或非 MME 时备用）。"""
    data = _attach_row_positions(data)
    rng = np.random.default_rng(EVAL_RANDOM_SEED)
    selected_idx_set: set = set()
    per_cat_stats = {}

    for cat in required_cats:
        cat_all = data[data["category"] == cat]
        if cat_all.empty:
            raise ValueError(f"Category '{cat}' not found in dataset.")

        disjoint_pool = cat_all[cat_all["_row_pos"] >= eval_row_start]
        if not disjoint_pool.empty:
            idxs, n_images = _sample_images_from_pool(disjoint_pool, rng, EVAL_IMAGES_PER_CATEGORY)
            source = "disjoint_tail"
        else:
            idxs, n_images, source = _fallback_category_holdout_images(
                cat_all, rng, EVAL_IMAGES_PER_CATEGORY, mining_cutoff,
            )
            if not idxs:
                raise ValueError(f"Category '{cat}' has no usable samples for eval.")
            print(
                f"[eval][warn] category={cat}: no samples at row_pos>={eval_row_start}, "
                f"using holdout from category tail ({source}); may overlap with mining prefix."
            )

        if n_images < int(EVAL_IMAGES_PER_CATEGORY):
            print(
                f"[eval][warn] category={cat}: only {n_images} images available "
                f"(< {EVAL_IMAGES_PER_CATEGORY})"
            )

        selected_idx_set.update(idxs)
        per_cat_stats[cat] = {
            "images": int(n_images),
            "rows": len(idxs),
            "source": source,
        }

    ordered = data[data["index"].isin(selected_idx_set)]["index"].tolist()
    return ordered, per_cat_stats


def resolve_eval_indices(dataset, dataset_name: str = "") -> list:
    data = dataset.data
    if EVAL_INDEX_LIST:
        indices = [idx for idx in EVAL_INDEX_LIST if idx in set(data["index"])]
        if not indices:
            raise ValueError("EVAL_INDEX_LIST is empty or not found in dataset.")
        return indices

    required_cats = _required_eval_categories(dataset_name)
    use_global_split = (
        EVAL_STRATIFIED
        and required_cats
        and "category" in data.columns
        and "image_path" in data.columns
    )

    if use_global_split:
        cache_path = _eval_index_cache_path(dataset_name)
        if EVAL_USE_INDEX_CACHE and cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as fin:
                payload = json.load(fin)
            if payload.get("version") == STRATIFIED_EVAL_VERSION:
                indices = payload.get("indices", [])
                if indices:
                    print(f"[eval] loaded stratified eval indices from cache: {cache_path} (n={len(indices)})")
                    return indices
            print(f"[eval] ignore stale eval cache (version mismatch): {cache_path}")

        eval_indices, _, split_stats = resolve_mining_eval_split(dataset, dataset_name)
        ordered = data[data["index"].isin(eval_indices)]["index"].tolist()
        print(f"[eval] stratified eval set: {len(ordered)} rows from {len(required_cats)} categories")

        if EVAL_SAVE_INDEX_LIST:
            os.makedirs(ANALYSIS_DIR, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as fout:
                json.dump({
                    "version": STRATIFIED_EVAL_VERSION,
                    "dataset": dataset_name,
                    "split_mode": split_stats.get("mode", "global_stratified"),
                    "images_per_category": EVAL_IMAGES_PER_CATEGORY,
                    "random_seed": EVAL_RANDOM_SEED,
                    "per_category": split_stats.get("per_category", {}),
                    "indices": ordered,
                }, fout, indent=2, ensure_ascii=False)
            print(f"[eval] saved stratified eval indices: {cache_path}")
        return ordered

    if EVAL_START_INDEX >= len(data):
        raise ValueError(
            f"EVAL_START_INDEX={EVAL_START_INDEX} >= dataset size {len(data)}; "
            "lower EVAL_START_INDEX or use EVAL_INDEX_LIST."
        )

    mining_cutoff = int(max(SAMPLE_SIZES))
    eval_row_start = int(max(EVAL_START_INDEX, mining_cutoff))
    if EVAL_START_INDEX < mining_cutoff:
        print(
            f"[eval][warn] EVAL_START_INDEX={EVAL_START_INDEX} < max(SAMPLE_SIZES)={mining_cutoff}; "
            "using prefix split — consider enabling global stratified split for MME."
        )

    cache_path = _eval_index_cache_path(dataset_name)
    if EVAL_USE_INDEX_CACHE and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as fin:
            payload = json.load(fin)
        if payload.get("version") == STRATIFIED_EVAL_VERSION:
            indices = payload.get("indices", [])
            if indices:
                print(f"[eval] loaded stratified eval indices from cache: {cache_path} (n={len(indices)})")
                return indices
        print(f"[eval] ignore stale eval cache (version mismatch): {cache_path}")

    if EVAL_STRATIFIED and required_cats and "category" in data.columns and "image_path" in data.columns:
        ordered, per_cat_stats = _build_stratified_eval_indices(
            data, required_cats, eval_row_start, mining_cutoff,
        )
        print(f"[eval] stratified eval set: {len(ordered)} rows from {len(required_cats)} categories")
        for cat, stat in per_cat_stats.items():
            print(
                f"  - {cat}: {stat['images']} images, {stat['rows']} QA rows "
                f"({stat['source']})"
            )

        if EVAL_SAVE_INDEX_LIST:
            os.makedirs(ANALYSIS_DIR, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as fout:
                json.dump({
                    "version": STRATIFIED_EVAL_VERSION,
                    "dataset": dataset_name,
                    "eval_start_index": EVAL_START_INDEX,
                    "mining_cutoff": mining_cutoff,
                    "eval_row_start": eval_row_start,
                    "images_per_category": EVAL_IMAGES_PER_CATEGORY,
                    "random_seed": EVAL_RANDOM_SEED,
                    "per_category": per_cat_stats,
                    "indices": ordered,
                }, fout, indent=2, ensure_ascii=False)
            print(f"[eval] saved stratified eval indices: {cache_path}")

        return ordered

    pool = data.iloc[eval_row_start:].copy()
    return pool["index"].tolist()


def _mining_row_cache_path(n: int, scope: str) -> Path:
    return Path(HEAD_STATS_ROOT) / f"n_{n}" / f"{scope}_mining_rows.json"


def _allocate_uniform_quotas(n: int, categories: list[str]) -> dict[str, int]:
    base, rem = divmod(int(n), len(categories))
    return {cat: base + (1 if i < rem else 0) for i, cat in enumerate(categories)}


def _allocate_proportional_quotas(n: int, pool: pd.DataFrame, categories: list[str]) -> dict[str, int]:
    counts = {cat: int((pool["category"] == cat).sum()) for cat in categories}
    total = sum(counts.values())
    if total <= 0:
        return {cat: 0 for cat in categories}
    raw = {cat: int(n) * counts[cat] / total for cat in categories}
    quotas = {cat: int(math.floor(raw[cat])) for cat in categories}
    leftover = int(n) - sum(quotas.values())
    if leftover > 0:
        order = sorted(categories, key=lambda c: (raw[c] - quotas[c], counts[c]), reverse=True)
        for cat in order:
            if leftover <= 0:
                break
            quotas[cat] += 1
            leftover -= 1
    return quotas


def _build_stratified_mining_df(
    data: pd.DataFrame,
    dataset_name: str,
    n: int,
    mining_pool_indices: set,
) -> tuple[pd.DataFrame, dict]:
    """在全局挖掘池内按子类配额抽样，保证各类别均可参与挖掘。"""
    data = _attach_row_positions(data)
    pool = data[data["index"].isin(mining_pool_indices)]
    categories = _required_eval_categories(dataset_name)
    if not categories:
        categories = sorted(pool["category"].dropna().unique().tolist()) if "category" in pool.columns else []

    if not MINE_STRATIFIED or not categories or "category" not in pool.columns:
        rows_df = pool.iloc[: int(n)].copy()
        return rows_df, {
            "mode": "global_pool_prefix",
            "pool_mode": "global_stratified",
            "pool_rows": len(pool),
            "rows": len(rows_df),
        }

    rng = np.random.default_rng(MINE_RANDOM_SEED + int(n))
    if MINE_STRATIFY_MODE == "proportional":
        quotas = _allocate_proportional_quotas(n, pool, categories)
        mode = "proportional"
    else:
        quotas = _allocate_uniform_quotas(n, categories)
        mode = "uniform"

    selected_indices: list = []
    per_cat_stats = {}
    for cat in categories:
        quota = int(quotas.get(cat, 0))
        cat_pool = pool[pool["category"] == cat]
        if quota <= 0 or cat_pool.empty:
            per_cat_stats[cat] = {"quota": quota, "rows": 0, "pool_rows": len(cat_pool)}
            continue
        n_take = min(quota, len(cat_pool))
        chosen = cat_pool.sample(n=n_take, random_state=int(rng.integers(0, 2**31 - 1)))
        idxs = chosen["index"].tolist()
        selected_indices.extend(idxs)
        per_cat_stats[cat] = {"quota": quota, "rows": len(idxs), "pool_rows": len(cat_pool)}

    rows_df = data[data["index"].isin(set(selected_indices))].copy()
    rows_df = rows_df.sort_values("_row_pos")
    stats = {
        "mode": mode,
        "pool_mode": "global_stratified",
        "target_rows": int(n),
        "actual_rows": len(rows_df),
        "pool_rows": len(pool),
        "per_category": per_cat_stats,
    }
    return rows_df, stats


def _load_or_build_mining_rows(
    data: pd.DataFrame,
    dataset_name: str,
    scope: str,
    n: int,
    mining_pool_indices: set,
) -> tuple[pd.DataFrame, dict]:
    cache_path = _mining_row_cache_path(n, scope)
    if MINE_SAVE_ROW_LIST and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as fin:
            payload = json.load(fin)
        if payload.get("version") == MINE_ROW_LIST_VERSION:
            indices = payload.get("indices", [])
            if indices:
                index_order = {idx: i for i, idx in enumerate(indices)}
                rows_df = data[data["index"].isin(indices)].copy()
                rows_df["_sort"] = rows_df["index"].map(index_order)
                rows_df = rows_df.sort_values("_sort").drop(columns="_sort")
                print(f"[mine] loaded stratified mining rows from cache: {cache_path} (n={len(rows_df)})")
                return rows_df, payload.get("stats", {})

    rows_df, stats = _build_stratified_mining_df(data, dataset_name, n, mining_pool_indices)
    if MINE_SAVE_ROW_LIST:
        os.makedirs(cache_path.parent, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fout:
            json.dump({
                "version": MINE_ROW_LIST_VERSION,
                "indices": rows_df["index"].tolist(),
                "stats": stats,
            }, fout, indent=2, ensure_ascii=False)
    return rows_df, stats


def _build_struct(model, dataset, dataset_name, row):
    if getattr(dataset, "force_use_dataset_prompt", False):
        return dataset.build_prompt(row)
    if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(dataset_name):
        return model.build_prompt(row, dataset=dataset_name)
    return dataset.build_prompt(row)


def partition_rows_df(rows_df: pd.DataFrame, rank: int, world_size: int) -> pd.DataFrame:
    """将挖掘行按 index 均匀切分到各 rank（保持各 rank 负载接近）。"""
    if world_size <= 1:
        return rows_df
    indices = rows_df["index"].tolist()
    my_indices = set(partition_by_rank(indices, rank, world_size))
    return rows_df[rows_df["index"].isin(my_indices)].copy()


def mine_heads_on_rows(model, dataset, dataset_name, rows_df, rank: int = 0, world_size: int = 1):
    _set_head_mode(model, "mine")
    n = len(rows_df)
    desc = f"Mining {dataset_name}"
    if world_size > 1:
        desc = f"Mining {dataset_name} rank {rank}/{world_size}"
    print(f"[mine] mining on {n} samples for {dataset_name}" + (f" (rank {rank})" if world_size > 1 else ""))
    for i in tqdm(range(n), desc=desc, unit="sample"):
        struct = _build_struct(model, dataset, dataset_name, rows_df.iloc[i])
        try:
            model.generate(message=struct, dataset=dataset_name)
        except Exception as err:
            idx = rows_df.iloc[i].get("index", i)
            print(f"[mine][warn] sample index={idx} failed: {type(err).__name__}: {err}")
            gc.collect()
        torch.cuda.empty_cache()


def top_heads_path(n: int, scope: str) -> Path:
    return Path(HEAD_STATS_ROOT) / f"n_{n}" / f"{scope}_top_heads.json"


def head_metrics_path(n: int, scope: str) -> Path:
    return Path(HEAD_STATS_ROOT) / f"n_{n}" / f"{scope}_head_metrics.csv"


def _write_merged_head_exports(n: int, scope: str, rows: list[dict]) -> None:
    """将合并后的 head 统计写入 n_{n}/{scope}_* 主文件。"""
    base_dir = Path(HEAD_STATS_ROOT) / f"n_{n}"
    os.makedirs(base_dir, exist_ok=True)
    rows = sorted(rows, key=lambda x: (x["mean_score"] * x["stability"], x["selected_ratio"]), reverse=True)
    top_k = max(8, int(AUTO_HEAD_TOP_K))
    top_heads = rows[:top_k]

    json_path = base_dir / f"{scope}_top_heads.json"
    with open(json_path, "w", encoding="utf-8") as fout:
        json.dump(top_heads, fout, ensure_ascii=False, indent=2)

    metrics_csv = base_dir / f"{scope}_head_metrics.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=["head", "seen", "selected", "selected_ratio", "mean_score", "std_score", "stability"],
        )
        writer.writeheader()
        writer.writerows(rows)

    stability_csv = base_dir / f"{scope}_layer_head_stability.csv"
    with open(stability_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["layer", "head", "seen", "selected_ratio", "mean_score", "stability"])
        for row in rows:
            layer, head = row["head"].split("_", 1)
            writer.writerow([layer, head, row["seen"], row["selected_ratio"], row["mean_score"], row["stability"]])


def merge_rank_mining_exports(n: int, scope: str, world_size: int) -> None:
    """合并各 rank 在 n_{n}/rank_* 下的挖掘统计到主目录。"""
    base_dir = Path(HEAD_STATS_ROOT) / f"n_{n}"
    merged_slots: dict[str, dict[str, float]] = {}

    for r in range(world_size):
        csv_path = base_dir / f"rank_{r}" / f"{scope}_head_metrics.csv"
        if not csv_path.exists():
            print(f"[mine][warn] missing rank {r} metrics for N={n}: {csv_path}")
            continue
        with open(csv_path, "r", encoding="utf-8", newline="") as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                head = str(row.get("head", "")).strip()
                if not head:
                    continue
                seen = int(float(row.get("seen", 0) or 0))
                if seen <= 0:
                    continue
                mean_score = float(row.get("mean_score", 0.0) or 0.0)
                std_score = float(row.get("std_score", 0.0) or 0.0)
                slot = merged_slots.setdefault(
                    head,
                    {"seen": 0.0, "selected": 0.0, "score_sum": 0.0, "score_sq_sum": 0.0},
                )
                slot["seen"] += seen
                slot["selected"] += float(row.get("selected", 0) or 0)
                slot["score_sum"] += mean_score * seen
                slot["score_sq_sum"] += seen * (std_score ** 2 + mean_score ** 2)

    if not merged_slots:
        print(f"[mine][warn] no rank metrics to merge for N={n}")
        return

    rows = []
    for head_key, stats in merged_slots.items():
        seen = max(1, int(stats["seen"]))
        mean_score = float(stats["score_sum"]) / seen
        var = max(0.0, float(stats["score_sq_sum"]) / seen - mean_score ** 2)
        std_score = float(np.sqrt(var))
        rows.append({
            "head": head_key,
            "seen": seen,
            "selected": int(stats["selected"]),
            "selected_ratio": float(stats["selected"]) / seen,
            "mean_score": mean_score,
            "std_score": std_score,
            "stability": float(1.0 / (1.0 + std_score)),
        })

    _write_merged_head_exports(n, scope, rows)
    print(f"[mine] merged rank exports for N={n}: {top_heads_path(n, scope)}")


def load_top_head_keys(n: int, scope: str) -> list[str]:
    path = top_heads_path(n, scope)
    if not path.exists():
        raise FileNotFoundError(f"Missing top heads file: {path}")
    with open(path, "r", encoding="utf-8") as fin:
        payload = json.load(fin)
    return [str(item["head"]) for item in payload if isinstance(item, dict) and "head" in item]


def load_head_ranking(metrics_csv: Path) -> list[str]:
    if not metrics_csv.exists():
        return []
    rows = []
    with open(metrics_csv, "r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            head = str(row.get("head", "")).strip()
            if not head:
                continue
            mean_score = float(row.get("mean_score", 0.0) or 0.0)
            stability = float(row.get("stability", 1.0) or 1.0)
            rows.append((head, mean_score * stability))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [head for head, _ in rows]


def overlap_jaccard(ref_heads: list[str], cmp_heads: list[str], k: int) -> tuple[float, float]:
    ref_set = set(ref_heads[:k])
    cmp_set = set(cmp_heads[:k])
    inter = len(ref_set & cmp_set)
    union = len(ref_set | cmp_set)
    overlap = inter / k if k > 0 else 0.0
    jaccard = inter / union if union > 0 else 0.0
    return float(overlap), float(jaccard)


def rank_correlation(ref_rank: list[str], cmp_rank: list[str]) -> tuple[float, float]:
    common = set(ref_rank) & set(cmp_rank)
    if len(common) < 2:
        return float("nan"), float("nan")
    ref_pos = {h: i for i, h in enumerate(ref_rank)}
    cmp_pos = {h: i for i, h in enumerate(cmp_rank)}
    x = [ref_pos[h] for h in common]
    y = [cmp_pos[h] for h in common]
    tau, _ = kendalltau(x, y)
    rho, _ = spearmanr(x, y)
    return float(tau), float(rho)


def mine_all_sample_sizes(
    dataset,
    dataset_name: str,
    scope: str,
    mining_pool_indices: set,
    rank: int = 0,
    world_size: int = 1,
):
    os.makedirs(HEAD_STATS_ROOT, exist_ok=True)
    if rank == 0:
        print(f"[mine] SAMPLE_SIZES={SAMPLE_SIZES}, world_size={world_size}")
        print(
            f"[mine] stratified={MINE_STRATIFIED} mode={MINE_STRATIFY_MODE} "
            f"pool_mode=global_stratified pool_rows={len(mining_pool_indices)}"
        )
        if world_size > 1:
            print("[mine] multi-GPU: each rank mines a disjoint sample shard per N, then rank0 merges")

    for n in sorted(SAMPLE_SIZES):
        out_path = top_heads_path(n, scope)
        if SKIP_EXISTING_MINING and out_path.exists():
            print(f"[mine] skip N={n}, exists: {out_path}")
            continue

        print(f"\n[mine] === N={n} (rank {rank}/{world_size}) ===")
        rows_df, mine_stats = _load_or_build_mining_rows(
            dataset.data, dataset_name, scope, n, mining_pool_indices,
        )
        if rank == 0:
            print(
                f"[mine] selected {mine_stats.get('actual_rows', len(rows_df))} rows "
                f"(target={n}, mode={mine_stats.get('mode', 'unknown')})"
            )
            for cat, stat in mine_stats.get("per_category", {}).items():
                print(f"  - {cat}: quota={stat.get('quota', 0)}, rows={stat.get('rows', 0)}")

        if world_size > 1:
            my_indices = partition_by_rank(rows_df["index"].tolist(), rank, world_size)
            local_rows = rows_df[rows_df["index"].isin(my_indices)].copy()
            stats_dir = str(Path(HEAD_STATS_ROOT) / f"n_{n}" / f"rank_{rank}")
        else:
            local_rows = rows_df
            stats_dir = str(Path(HEAD_STATS_ROOT) / f"n_{n}")

        print(f"[mine] rank {rank}/{world_size} local shard: {len(local_rows)}/{len(rows_df)} samples")

        if len(local_rows) > 0:
            os.makedirs(stats_dir, exist_ok=True)
            model = build_model(head_stats_dir=stats_dir)
            _reset_head_profile(model)
            if hasattr(model, "set_dump_image"):
                model.set_dump_image(dataset.dump_image)

            mine_heads_on_rows(
                model, dataset, dataset_name, local_rows,
                rank=rank, world_size=world_size,
            )

            del model
            gc.collect()
            torch.cuda.empty_cache()

        if world_size > 1:
            barrier(world_size)
            if rank == 0:
                merge_rank_mining_exports(n, scope, world_size)
            barrier(world_size)


def analyze_head_selection_stability(dataset_name: str, scope: str):
    if REFERENCE_N not in SAMPLE_SIZES:
        raise ValueError(f"REFERENCE_N={REFERENCE_N} must be in SAMPLE_SIZES")

    ref_heads = load_top_head_keys(REFERENCE_N, scope)
    ref_rank = load_head_ranking(head_metrics_path(REFERENCE_N, scope))
    if not ref_rank:
        ref_rank = ref_heads

    records = []
    detail = {}
    k = int(AUTO_HEAD_TOP_K)

    for n in sorted(SAMPLE_SIZES):
        heads = load_top_head_keys(n, scope)
        rank = load_head_ranking(head_metrics_path(n, scope)) or heads
        overlap, jaccard = overlap_jaccard(ref_heads, heads, k)
        tau, rho = rank_correlation(ref_rank, rank)
        records.append({
            "dataset": dataset_name,
            "N": n,
            f"overlap@{k}": overlap,
            f"jaccard@{k}": jaccard,
            "kendall_tau": tau,
            "spearman_rho": rho,
            "selected_heads": ",".join(heads[:k]),
        })
        detail[str(n)] = {"heads": heads[:k], "rank_top10": rank[:10]}

    df = pd.DataFrame(records)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    csv_path = Path(ANALYSIS_DIR) / f"{MODEL_SLUG}_{dataset_name}_layer1_stability.csv"
    json_path = Path(ANALYSIS_DIR) / f"{MODEL_SLUG}_{dataset_name}_layer1_stability.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as fout:
        json.dump({"reference_N": REFERENCE_N, "scope": scope, "metrics": records, "detail": detail}, fout, indent=2)

    print(f"\n[layer1] saved: {csv_path}")
    print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".4f"))

    stable_ns = df[
        (df[f"jaccard@{k}"] >= LAYER1_JACCARD_THRESH)
        & (df["kendall_tau"] >= LAYER1_KENDALL_THRESH)
    ]["N"].tolist()
    min_stable = min(stable_ns) if stable_ns else None
    print(
        f"[layer1] stable N (jaccard@{k}>={LAYER1_JACCARD_THRESH}, "
        f"kendall>={LAYER1_KENDALL_THRESH}): {min_stable}"
    )
    return df


def _flatten_eval_metrics(eval_results) -> dict[str, float]:
    if eval_results is None:
        return {}
    if isinstance(eval_results, dict):
        out = {}
        for key, val in eval_results.items():
            try:
                out[str(key)] = float(val)
            except (TypeError, ValueError):
                continue
        return out
    if isinstance(eval_results, pd.DataFrame):
        df = eval_results.T if len(eval_results) < len(eval_results.columns) else eval_results
        out = {}
        for col in df.columns:
            try:
                out[str(col)] = float(df[col].iloc[0])
            except (TypeError, ValueError, IndexError):
                continue
        return out
    return {}


def infer_eval_subset(
    model,
    dataset,
    dataset_name: str,
    eval_df: pd.DataFrame,
    out_file: str,
    rank: int = 0,
    world_size: int = 1,
):
    if hasattr(model, "set_dump_image"):
        model.set_dump_image(dataset.dump_image)
    _set_head_mode(model, "inference")

    full_eval_df = eval_df.copy()
    local_df = eval_df.iloc[rank::world_size].copy() if world_size > 1 else eval_df

    predictions = {}
    desc = f"Infer {dataset_name} rank {rank}/{world_size}"
    for i in tqdm(range(len(local_df)), desc=desc, unit="sample"):
        row = local_df.iloc[i]
        idx = row["index"]
        struct = _build_struct(model, dataset, dataset_name, row)
        try:
            predictions[idx] = model.generate(message=struct, dataset=dataset_name)
        except Exception as err:
            predictions[idx] = f"[ERROR] {type(err).__name__}: {err}"
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()

    if world_size > 1:
        partial_path = out_file.rsplit(".", 1)[0] + f"_rank{rank}.pkl"
        dump(predictions, partial_path)
        barrier(world_size)
        if rank == 0:
            merged = {}
            base = out_file.rsplit(".", 1)[0]
            for r in range(world_size):
                part_file = f"{base}_rank{r}.pkl"
                if os.path.exists(part_file):
                    merged.update(load(part_file))
            out_data = full_eval_df.copy()
            out_data["prediction"] = [merged[idx] for idx in out_data["index"]]
            os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
            dump(out_data, out_file)
            for r in range(world_size):
                part_file = f"{base}_rank{r}.pkl"
                if os.path.exists(part_file):
                    os.remove(part_file)
        barrier(world_size)
        return

    out_data = full_eval_df.copy()
    out_data["prediction"] = [predictions[idx] for idx in out_data["index"]]
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    dump(out_data, out_file)


def eval_downstream_stability(
    dataset,
    dataset_name: str,
    eval_indices: list,
    scope: str,
    rank: int = 0,
    world_size: int = 1,
):
    eval_df = dataset.data[dataset.data["index"].isin(eval_indices)].copy()
    if rank == 0:
        print(f"[layer2] eval subset size={len(eval_df)} (fixed)")
        if world_size > 1:
            print(
                f"[layer2] multi-GPU: world_size={world_size}, "
                f"each rank infers ~{len(eval_df) // world_size} samples per N"
            )

    model = build_model()
    if hasattr(model, "set_dump_image"):
        model.set_dump_image(dataset.dump_image)

    records = []
    os.makedirs(WORK_DIR, exist_ok=True)

    # 所有 rank 共同处理每个 N；评测样本在 infer_eval_subset 内按 rank 切分并行。
    for n in sorted(SAMPLE_SIZES):
        heads = load_top_head_keys(n, scope)[: int(AUTO_HEAD_TOP_K)]
        model.inference_selected_heads = heads
        _set_head_mode(model, "inference")

        model_name = f"{MODEL_NAME}_two_stage_n{n}"
        result_file = get_pred_file_path(WORK_DIR, model_name, dataset_name)
        if rank == 0:
            print(f"\n[layer2] N={n} heads={heads} -> {result_file}")

        infer_eval_subset(
            model, dataset, dataset_name, eval_df, result_file,
            rank=rank, world_size=world_size,
        )
        if rank == 0:
            eval_results = dataset.evaluate(result_file, **dict(JUDGE_KWARGS))
            metrics = _flatten_eval_metrics(eval_results)
            row = {"dataset": dataset_name, "N": n, "heads": ",".join(heads), **metrics}
            records.append(row)
        barrier(world_size)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    if rank != 0:
        return None

    df = pd.DataFrame(records)

    ref_row = df[df["N"] == REFERENCE_N]
    if ref_row.empty:
        print(f"[layer2][warn] REFERENCE_N={REFERENCE_N} not in results; skip delta.")
    else:
        ref_metrics = ref_row.iloc[0]
        metric_cols = [
            c for c in df.columns
            if c not in {"dataset", "N", "heads"} and pd.api.types.is_numeric_dtype(df[c])
        ]
        for col in metric_cols:
            ref_val = float(ref_metrics[col])
            df[f"delta_{col}"] = df[col].astype(float) - ref_val

    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    csv_path = Path(ANALYSIS_DIR) / f"{MODEL_SLUG}_{dataset_name}_layer2_downstream.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[layer2] saved: {csv_path}")
    print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".4f"))

    primary_cols = [c for c in LAYER2_PRIMARY_METRICS if c in df.columns]
    if not primary_cols:
        primary_cols = [
            c for c in df.columns
            if c.startswith("delta_") is False
            and c not in {"dataset", "N", "heads"}
            and pd.api.types.is_numeric_dtype(df[c])
        ][:2]

    stable_ns = []
    for n in sorted(SAMPLE_SIZES):
        row = df[df["N"] == n]
        if row.empty:
            continue
        ok = True
        for col in primary_cols:
            delta_col = f"delta_{col}"
            if delta_col in df.columns and abs(float(row.iloc[0][delta_col])) > LAYER2_DELTA_THRESH:
                ok = False
                break
        if ok:
            stable_ns.append(n)

    min_stable = min(stable_ns) if stable_ns else None
    print(
        f"[layer2] stable N (|delta| <= {LAYER2_DELTA_THRESH} on {primary_cols}): {min_stable}"
    )

    return df


def main():
    rank, world_size = init_distributed()
    if rank == 0:
        os.makedirs(ANALYSIS_DIR, exist_ok=True)
        os.makedirs(WORK_DIR, exist_ok=True)
        os.makedirs(HEAD_STATS_ROOT, exist_ok=True)
    barrier(world_size)

    for dataset_name in DATASET_NAMES:
        if rank == 0:
            print(f"\n{'=' * 80}\nDataset: {dataset_name}\n{'=' * 80}")
        dataset = build_dataset(dataset_name)
        if dataset is None:
            if rank == 0:
                print(f"[skip] invalid dataset: {dataset_name}")
            continue

        probe = build_model()
        scope = get_scope_name(probe, dataset_name)
        del probe
        gc.collect()
        torch.cuda.empty_cache()
        if rank == 0:
            print(f"profile scope: {scope}")

        _, mining_pool_indices, _ = resolve_mining_eval_split(dataset, dataset_name)
        eval_indices = resolve_eval_indices(dataset, dataset_name)
        if rank == 0:
            print(f"eval subset: {len(eval_indices)} samples (fixed)")
            print(f"mining pool: {len(mining_pool_indices)} samples (global stratified)")

        if "mine" in RUN_PHASES:
            mine_all_sample_sizes(
                dataset, dataset_name, scope, mining_pool_indices,
                rank=rank, world_size=world_size,
            )
        barrier(world_size)

        if "analyze" in RUN_PHASES and rank == 0:
            analyze_head_selection_stability(dataset_name, scope)
        barrier(world_size)

        if "eval" in RUN_PHASES:
            eval_downstream_stability(
                dataset, dataset_name, eval_indices, scope,
                rank=rank, world_size=world_size,
            )
        barrier(world_size)

    if rank == 0:
        print(f"\nDone. Reports under: {ANALYSIS_DIR}")
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
