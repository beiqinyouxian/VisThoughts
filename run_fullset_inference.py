"""大规模全量推理脚本。

复用评测框架的 infer_data + 文件信号协调，完成整套数据集推理与指标计算，
关闭所有注意力可视化落盘，并支持在传统单阶段推理与两阶段注意力推理之间自由切换。

两阶段流程（mine_then_inference）：
  第一阶段：在指定的 MINE_SAMPLE_COUNT 个样本上做注意力头挖掘（mine），导出每个数据集作用域的 top_heads。
  第二阶段：切换到 inference，直接复用挖掘出的注意力头对全量样本推理，不再重新挖掘。

当前默认 HEAD_SELECTION_MODE="inference"：跳过挖掘，从 HEAD_STATS_DIR 加载已固化的 top_heads
（MME N=200 权威副本：outputs/selected_heads/mme_n200/）。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
单卡：python run_fullset_inference.py
多卡：GPU_GROUPS="0;1;2;3" torchrun --nproc_per_node=4 run_fullset_inference.py
      （每个 rank 必须只绑定一张 GPU，否则多进程会挤在同一张卡上 OOM）

多进程协调：使用文件信号（rank N 写入 .done 文件）替代 NCCL barrier，
避免各 rank 推理速度不均导致超时。
"""
import gc
import json
import os
import os.path as osp
import time

# 需在导入 torch 之前设置，缓解显存碎片导致的 OOM。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _setup_gpu_binding():
    """torchrun 多进程时，每个 rank 只可见一张物理 GPU（与 run_head_mining_sample_analysis 一致）。

    未绑定时 4 个 rank 都会看到全部 GPU，device_map=\"auto\" 会在同一张卡上叠多个进程，导致 OOM。
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    groups = os.environ.get("GPU_GROUPS", "0;1;2;3").split(";")
    if local_rank < len(groups):
        os.environ["CUDA_VISIBLE_DEVICES"] = groups[local_rank].strip()
        print(
            f"[dist] LOCAL_RANK={local_rank} -> CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}",
            flush=True,
        )


_setup_gpu_binding()

import torch
from tabulate import tabulate
from tqdm.auto import tqdm

from vlmeval.config import supported_VLM
from vlmeval.dataset import build_dataset
from vlmeval.inference import infer_data
from vlmeval.smp import dump, get_pred_file_path, get_rank_and_world_size, load

# =============== 参数区（按需修改）===============
MODEL_NAME = "Qwen2.5-VL-Enhance-7B-Instruct"
DATASET_NAMES = ["MME"]
REASONING_MODE = "two_stage"

# 两阶段头选择流程（仅 REASONING_MODE="two_stage" 时生效）：
# - "mine_then_inference": 先在 MINE_SAMPLE_COUNT 个样本上挖掘注意力头，再对全量样本 inference
# - "inference":           跳过挖掘，直接加载已有 top_heads 对全量样本 inference
# - "mine":                仅做注意力头挖掘（不做全量推理/评测）
# 默认 inference：MME 的 N=200 头已固化到 HEAD_STATS_DIR（权威副本见 outputs/selected_heads/mme_n200/）
HEAD_SELECTION_MODE = "inference"
MINE_SAMPLE_COUNT = 200                 # 第一阶段用于挖掘注意力头的样本数（仅 mine / mine_then_inference 时用）

WORK_DIR = "./outputs/fullset_infer"

# 头挖掘/加载目录（mine 导出、inference 加载共用同一目录）
# MME N=200 各模型 * __mme_top_heads.json 已安装于此；权威副本：./outputs/selected_heads/mme_n200/
HEAD_STATS_DIR = "./temp_debug/head_stats"
AUTO_HEAD_TOP_K = 3

# 评测参数；需要 LLM judge 的数据集（如 MMVet/MathVista）在此追加 "model": "gpt-4o-mini"
JUDGE_KWARGS = {"nproc": 4, "verbose": False}

_REASONING_MAP = {"single": "single_stage", "two_stage": "two_stage_attention"}


def _set_head_mode(model, mode):
    """在同一模型实例上切换头选择模式（mine / inference），避免重复加载权重。"""
    if hasattr(model, "head_selection_mode"):
        model.head_selection_mode = mode
    if hasattr(model, "auto_head_mining"):
        model.auto_head_mining = mode == "mine"


def build_model():
    if REASONING_MODE not in _REASONING_MAP:
        raise ValueError(f"REASONING_MODE must be one of {list(_REASONING_MAP)}")
    if REASONING_MODE == "two_stage" and HEAD_SELECTION_MODE not in {"mine_then_inference", "inference", "mine"}:
        raise ValueError("HEAD_SELECTION_MODE must be one of: mine_then_inference / inference / mine")

    print(f"Loading model: {MODEL_NAME} | reasoning_mode={REASONING_MODE}")
    model = supported_VLM[MODEL_NAME](verbose=False)

    # 推理方法切换：传统单阶段 vs 两阶段注意力
    if hasattr(model, "reasoning_mode"):
        model.reasoning_mode = _REASONING_MAP[REASONING_MODE]

    # 关闭全部注意力可视化落盘（掩码图仍走临时文件，不影响两阶段推理）
    for attr in ("save_heatmap_overlay", "save_per_keyword_overlay", "save_attention_debug"):
        if hasattr(model, attr):
            setattr(model, attr, False)
    if hasattr(model, "attention_debug_dir"):
        model.attention_debug_dir = None

    # 两阶段头挖掘相关目录/参数（各数据集作用域自动区分）
    if REASONING_MODE == "two_stage":
        if hasattr(model, "auto_head_top_k"):
            model.auto_head_top_k = int(AUTO_HEAD_TOP_K)
        if hasattr(model, "auto_head_stats_dir"):
            model.auto_head_stats_dir = HEAD_STATS_DIR

    return model


def _build_struct(model, dataset, dataset_name, row):
    """与 vlmeval.inference.infer_data 保持一致的 prompt 构建逻辑。"""
    if getattr(dataset, "force_use_dataset_prompt", False):
        return dataset.build_prompt(row)
    if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(dataset_name):
        return model.build_prompt(row, dataset=dataset_name)
    return dataset.build_prompt(row)


def mine_heads(model, dataset, dataset_name, n):
    """第一阶段：在前 n 个样本上以 mine 模式运行两阶段前向，挖掘并导出注意力头。"""
    _set_head_mode(model, "mine")
    data = dataset.data
    n = min(int(n), len(data))
    print(f"[mine] mining attention heads on {n} samples for {dataset_name}")
    for i in tqdm(range(n), desc=f"Mining {dataset_name}", unit="sample"):
        struct = _build_struct(model, dataset, dataset_name, data.iloc[i])
        try:
            model.generate(message=struct, dataset=dataset_name)
        except Exception as err:
            print(f"[mine][warn] sample index={data.iloc[i].get('index', i)} failed: {type(err).__name__}: {err}")
            gc.collect()
            torch.cuda.empty_cache()


def print_eval_results(dataset_name, eval_results):
    print("=" * 100)
    print(f"Evaluation results | dataset={dataset_name}")
    if eval_results is None:
        print("(dataset.evaluate returned None)")
    elif isinstance(eval_results, dict):
        print(json.dumps(eval_results, indent=4, ensure_ascii=False))
    else:  # pandas.DataFrame
        df = eval_results
        if len(df) < len(df.columns):
            df = df.T
        print(tabulate(df, headers="keys"))
    print("=" * 100)


def _file_barrier_signal(rank, world_size, work_dir, tag):
    """写入本 rank 完成标记文件，rank 0 轮询等待所有其他 rank 完成后再继续。

    替代 NCCL dist.barrier()，避免各 rank 推理耗时不均导致的 NCCL 超时。
    """
    signal_file = osp.join(work_dir, f'.signal_{tag}_rank{rank}')
    with open(signal_file, 'w') as f:
        f.write('done')
    if rank == 0:
        for r in range(1, world_size):
            other = osp.join(work_dir, f'.signal_{tag}_rank{r}')
            while not osp.exists(other):
                time.sleep(3)


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    model_name = f"{MODEL_NAME}_{REASONING_MODE}"
    rank, world_size = get_rank_and_world_size()

    # 与 vlmeval.inference.infer_data 一致：建模型前临时去掉 WORLD_SIZE，
    # 避免 transformers 在 device_map=auto 下误启 TP，与 torchrun 多实例冲突。
    ws_bak = os.environ.pop("WORLD_SIZE", None)
    model = build_model()
    if ws_bak is not None:
        os.environ["WORLD_SIZE"] = ws_bak

    two_stage = REASONING_MODE == "two_stage"

    for dataset_name in DATASET_NAMES:
        print(f"\nLoading dataset: {dataset_name}")
        dataset = build_dataset(dataset_name)
        if dataset is None:
            print(f"[skip] dataset not found or invalid: {dataset_name}")
            continue
        if hasattr(model, "set_dump_image"):
            model.set_dump_image(dataset.dump_image)

        # 第一阶段：注意力头挖掘（仅两阶段且需要挖掘时）。
        # 仅 rank 0 挖掘，避免多进程重复计算与 top_heads.json 写入竞争。
        if two_stage and HEAD_SELECTION_MODE in {"mine", "mine_then_inference"}:
            if rank == 0:
                mine_heads(model, dataset, dataset_name, MINE_SAMPLE_COUNT)
            if world_size > 1:
                _file_barrier_signal(rank, world_size, WORK_DIR, f"mine_{dataset_name}")
            if HEAD_SELECTION_MODE == "mine":
                if rank == 0:
                    print(f"[mine] finished mining for {dataset_name}; skip full inference (HEAD_SELECTION_MODE='mine').")
                continue

        # 第二阶段：切到 inference，复用已挖掘的注意力头对全量样本推理
        if two_stage:
            _set_head_mode(model, "inference")

        # ---- 全量推理（每 rank 独立写入自己的 pkl 文件，无需 NCCL 同步） ----
        dataset_dname = dataset.dataset_name
        result_file = get_pred_file_path(WORK_DIR, model_name, dataset_dname)
        prev_file = f'{WORK_DIR}/{model_name}_{dataset_dname}_PREV.pkl'

        # 加载断点续跑结果（仅 rank 0 负责从 result_file 恢复）
        if rank == 0 and osp.exists(result_file):
            data = load(result_file)
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
            dump(results, prev_file)
        elif osp.exists(prev_file):
            # 其他 rank 如果有 prev_file 也加载（之前 rank 0 写入的）
            pass

        # 每 rank 输出到各自文件
        out_file = osp.join(WORK_DIR, f'{rank}_{world_size}_{dataset_dname}.pkl')
        infer_data(
            model=model,
            work_dir=WORK_DIR,
            model_name=model_name,
            dataset=dataset,
            out_file=out_file,
            verbose=False,
            api_nproc=4,
        )

        gc.collect()
        torch.cuda.empty_cache()

        # 所有 rank 推理完成后，rank 0 负责聚合结果并写入最终文件
        if world_size > 1:
            _file_barrier_signal(rank, world_size, WORK_DIR, f"infer_{dataset_dname}")

        if rank == 0:
            # 聚合各 rank 结果
            data_all = {}
            for i in range(world_size):
                fpath = osp.join(WORK_DIR, f'{i}_{world_size}_{dataset_dname}.pkl')
                if osp.exists(fpath):
                    data_all.update(load(fpath))
            data = dataset.data
            for x in data['index']:
                assert x in data_all, f"Missing result for index {x}"
            data['prediction'] = [str(data_all[x]) for x in data['index']]
            if 'image' in data:
                data.pop('image')
            dump(data, result_file)

            # 清理临时文件
            for i in range(world_size):
                fpath = osp.join(WORK_DIR, f'{i}_{world_size}_{dataset_dname}.pkl')
                if osp.exists(fpath):
                    os.remove(fpath)
            checkpoint_file = f'{WORK_DIR}/{model_name}_{dataset_dname}_checkpoint.pkl'
            if osp.exists(checkpoint_file):
                os.remove(checkpoint_file)
            if osp.exists(prev_file):
                os.remove(prev_file)
            # 清理信号文件
            for i in range(world_size):
                sig = osp.join(WORK_DIR, f'.signal_infer_{dataset_dname}_rank{i}')
                if osp.exists(sig):
                    os.remove(sig)
            if world_size > 1:
                mine_sig = osp.join(WORK_DIR, f'.signal_mine_{dataset_dname}_rank0')
                if osp.exists(mine_sig):
                    for i in range(world_size):
                        msig = osp.join(WORK_DIR, f'.signal_mine_{dataset_dname}_rank{i}')
                        if osp.exists(msig):
                            os.remove(msig)

            # 指标评测
            eval_results = dataset.evaluate(result_file, **JUDGE_KWARGS)
            print_eval_results(dataset_dname, eval_results)

    if rank == 0:
        print(f"\nDone. Predictions and metrics saved under: {WORK_DIR}")


if __name__ == "__main__":
    main()
