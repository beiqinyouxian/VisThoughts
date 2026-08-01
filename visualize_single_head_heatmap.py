"""Visualize a single attention-head heatmap for one keyword on one image.

Inspired by ``debug_prompt_mode_switch.py`` (top-level parameter block +
mine / inference-style head selection) and reuses the enhance-model attention
pipeline without modifying library code.

Typical usage:
  conda activate vlmeval_llava
  CUDA_VISIBLE_DEVICES=0 python visualize_single_head_heatmap.py

Defaults target ``photo.png`` with keyword ``phone``.
For 13B full-head mine, prefer multi-GPU or HEAD_MODE=\"mine_candidates\".
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from vlmeval.config import supported_VLM

# =============== 参数区（按需修改）===============
# 与 debug_prompt_mode_switch.py 对齐可用 13B；单卡全头扫描建议 7B。
MODEL_NAME = "LLaVA-Enhance-1.5-7B"
IMAGE_PATH = "./photo.png"
KEYWORD = "phone"

# 选头模式:
# - "mine": 全头扫描后对本图关键词打分，取 top-1（显存大）
# - "mine_candidates": 仅对候选头（TOP_HEADS / CANDIDATE_HEADS）打分，取 top-1（省显存，适合 13B）
# - "manual": 使用 MANUAL_HEAD（若 MULTI_HEADS 非空则批量导出这些头）
# - "batch": 批量导出 MULTI_HEADS（或 mining 选出的多头列表）
# - "top_file": 从已挖掘的 top_heads.json 取第 1 个（不打分）
HEAD_MODE = "mine"
MANUAL_HEAD = "17_2"  # layer_head，例如 "17_2"
# batch / manual 批量列表；为空时 batch 会回退到 MANUAL_HEAD
MULTI_HEADS: list[str] = [
    "20_10",
    "22_8",
    "19_15",
    "14_24",
    "11_17",
    "14_5",
]
TOP_HEADS_PATH = "./temp_debug/head_stats/llava_1_5_7b__mme_top_heads.json"
# mine_candidates 时最多评估前 N 个候选；也可显式指定 CANDIDATE_HEADS
CANDIDATE_TOP_N = 16
CANDIDATE_HEADS: list[str] = []  # 例如 ["10_29", "14_24", "11_17"]

# 注意力类型与辅助 prompt（与 enhance 管线一致）
ATTENTION_TYPE = "rel"  # "rel" | "orin"
GENERAL_PROMPT = (
    "Write a general description of the image. "
    "Answer the question using a single word or phrase."
)

# 分析图边长上限（LLaVA 固定 336 方图，此项主要留给 Qwen）
ANALYSIS_MAX_SIDE = 1024

OUT_DIR = Path("./outputs/single_head_heatmap")
SAVE_ALL_HEAD_RANKING = True  # mine 时额外导出打分排序
DPI = 160
# jet 热力图叠原图透明度：越大颜色越实、原图越淡；建议 0.45~0.85
HEATMAP_ON_IMAGE_ALPHA = 0.78
# ================================================


def _normalize_head_key(head: str) -> str:
    return str(head).strip().replace("-", "_")


def _parse_head_key(head: str) -> tuple[int, int]:
    key = _normalize_head_key(head)
    if "_" not in key:
        raise ValueError(f"Invalid head key {head!r}; expected 'layer_head'.")
    layer_s, head_s = key.split("_", 1)
    return int(layer_s), int(head_s)


def _safe_tag(text: str) -> str:
    raw = str(text or "").strip().lower()
    safe = re.sub(r"[^0-9a-zA-Z_-]+", "_", raw).strip("_")
    return safe or "kw"


def build_model(model_name: str):
    if model_name not in supported_VLM:
        raise KeyError(
            f"Unknown model {model_name!r}. Enhance entries: "
            f"{[k for k in supported_VLM if 'Enhance' in k]}"
        )
    print(f"[info] loading model: {model_name}")
    model = supported_VLM[model_name](verbose=False)
    if hasattr(model, "attention_type"):
        model.attention_type = ATTENTION_TYPE
    return model


def _prepare_analysis_image(current_image: Image.Image, analysis_max_side: int) -> Image.Image:
    """Match enhance pipeline: square-pad, then optionally downscale."""
    from vlmeval.vlm.llava.utils import resize_to_square

    # Qwen / LLaVA enhance both expose compatible resize helpers; LLaVA one is fine.
    try:
        from vlmeval.vlm.qwen2_vl.utils import resize_to_square as qwen_resize

        analysis_image = qwen_resize(current_image, current_image.size)
    except Exception:
        analysis_image = resize_to_square(current_image, current_image.size)

    max_side = int(max(1, analysis_max_side))
    w, h = analysis_image.size
    if max(w, h) > max_side:
        analysis_image = analysis_image.resize((max_side, max_side), Image.BILINEAR)
        print(f"[info] analysis image resized {w}x{h} -> {max_side}x{max_side}")
    return analysis_image


def _prepare_attention_helpers(model):
    """Pick prepare_attention_maps / utils module matching the enhance class."""
    model_cls = type(model).__name__.lower()
    if "llava" in model_cls:
        from vlmeval.vlm.llava.utils import (
            normalize_attention_map_for_eval,
            prepare_attention_maps_for_image,
        )
        from vlmeval.vlm.llava.model_methods import llava_methods as methods

        manual_rel = getattr(methods, "manual_param_rel_attention_llava", None)
        manual_orin = getattr(methods, "manual_param_orin_attention_llava", None)
    else:
        from vlmeval.vlm.qwen2_vl.utils import (
            normalize_attention_map_for_eval,
            prepare_attention_maps_for_image,
        )

        manual_rel = None
        manual_orin = None

    return {
        "prepare_attention_maps_for_image": prepare_attention_maps_for_image,
        "normalize_attention_map_for_eval": normalize_attention_map_for_eval,
        "manual_rel": manual_rel,
        "manual_orin": manual_orin,
    }


def extract_all_head_maps(
    model,
    image_path: str,
    keyword: str,
    analysis_max_side: int = 1024,
) -> tuple[dict[str, np.ndarray], dict[str, Any], tuple[int, int]]:
    """Scan all heads for one keyword (same path as mine round-1)."""
    helpers = _prepare_attention_helpers(model)
    current_image = Image.open(image_path).convert("RGB")
    analysis_image = _prepare_analysis_image(current_image, analysis_max_side)
    image_size = current_image.size

    scan_cfg = model._resolve_attention_scan_config()
    map_func = model._resolve_attention_map_func()
    print(
        f"[extract] keyword={keyword!r} scanning layers={scan_cfg['layers']} "
        f"heads={scan_cfg['heads']} ..."
    )
    raw = map_func(
        analysis_image,
        str(keyword),
        GENERAL_PROMPT,
        model.model,
        model.processor,
        scan_cfg["layers"],
        scan_cfg["heads"],
    )
    head_maps = helpers["prepare_attention_maps_for_image"](raw, image_size=image_size)
    head_maps = {_normalize_head_key(k): v for k, v in head_maps.items()}
    print(f"[extract] got {len(head_maps)} head maps")
    return head_maps, scan_cfg, image_size


def extract_single_head_map(
    model,
    image_path: str,
    keyword: str,
    head_key: str,
    analysis_max_side: int = 1024,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Prefer manual single-head API when available; else scan-all and pick."""
    helpers = _prepare_attention_helpers(model)
    current_image = Image.open(image_path).convert("RGB")
    analysis_image = _prepare_analysis_image(current_image, analysis_max_side)
    image_size = current_image.size
    layer, head = _parse_head_key(head_key)

    use_rel = str(getattr(model, "attention_type", ATTENTION_TYPE)).lower() != "orin"
    manual = helpers["manual_rel"] if use_rel else helpers["manual_orin"]
    if manual is not None:
        print(f"[extract] single-head manual API: {head_key}")
        raw = manual(
            analysis_image,
            str(keyword),
            GENERAL_PROMPT,
            model.model,
            model.processor,
            layer,
            head,
        )
        heat = helpers["normalize_attention_map_for_eval"](raw, image_size)
        return heat.astype(np.float32), image_size

    print(f"[extract] fallback: scan all heads then pick {head_key}")
    head_maps, _, _ = extract_all_head_maps(
        model, image_path, keyword, analysis_max_side=analysis_max_side
    )
    key = _normalize_head_key(head_key)
    if key not in head_maps:
        raise KeyError(f"Head {key!r} not found in scanned maps.")
    return np.asarray(head_maps[key], dtype=np.float32), image_size


def load_heads_from_file(path: str | Path, top_n: int | None = None) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TOP_HEADS_PATH not found: {path}")
    with open(path, "r", encoding="utf-8") as fin:
        payload = json.load(fin)

    items: list[Any] = []
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        for key in ("selected_heads", "heads", "top_heads"):
            value = payload.get(key)
            if isinstance(value, list) and value:
                items = value
                break
    heads: list[str] = []
    for item in items:
        if isinstance(item, dict) and "head" in item:
            heads.append(_normalize_head_key(item["head"]))
        elif isinstance(item, str):
            heads.append(_normalize_head_key(item))
    # dedupe keep order
    seen = set()
    ordered = []
    for h in heads:
        if h and h not in seen:
            seen.add(h)
            ordered.append(h)
    if not ordered:
        raise ValueError(f"Cannot parse heads from {path}")
    if top_n is not None:
        ordered = ordered[: max(1, int(top_n))]
    return ordered


def load_top_head_from_file(path: str | Path) -> str:
    return load_heads_from_file(path, top_n=1)[0]


def resolve_candidate_heads() -> list[str]:
    if CANDIDATE_HEADS:
        return [_normalize_head_key(h) for h in CANDIDATE_HEADS if str(h).strip()]
    return load_heads_from_file(TOP_HEADS_PATH, top_n=CANDIDATE_TOP_N)


def mine_best_head(
    model,
    image_path: str,
    keyword: str,
    head_maps: dict[str, np.ndarray],
    scan_cfg: dict[str, Any],
    image_size,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """Score heads with hybrid pseudo-GT; return best head + mining meta."""
    print(f"[mine] building hybrid pseudo-GT for {keyword!r} ...")
    pseudo_gt = model._build_hybrid_pseudo_gt_for_keyword(
        image_path=image_path,
        keyword=keyword,
        image_size=image_size,
        fallback_heatmap=None,
    )
    src = pseudo_gt.get("source", "none")
    has_mask = int(np.asarray(pseudo_gt.get("mask_gt", np.zeros((1, 1)))).sum()) > 0
    print(f"[mine] source={src} has_mask={has_mask}")

    mining = model._mine_heads_for_keywords(
        {keyword: head_maps},
        {keyword: pseudo_gt},
        scan_cfg,
    )
    selected = [
        _normalize_head_key(h) for h in (mining.get("selected_head_keys") or [])
    ]

    ranking: list[dict[str, Any]] = []
    # Prefer aggregated ranked list; fall back to per-keyword metrics.
    ranked = mining.get("ranked_heads") or []
    for item in ranked:
        # select_top_heads_with_pruning returns List[Tuple[head_key, metrics_dict]]
        if isinstance(item, tuple) and len(item) == 2:
            head_key, mets = item
            row = {"head": _normalize_head_key(head_key)}
            if isinstance(mets, dict):
                for k, v in mets.items():
                    if isinstance(v, (int, float, np.floating)):
                        row[k] = float(v)
                    else:
                        row[k] = v
            if row["head"]:
                ranking.append(row)
            continue
        if isinstance(item, dict):
            row = {"head": _normalize_head_key(item.get("head", ""))}
            for k, v in item.items():
                if k == "head":
                    continue
                if isinstance(v, (int, float, np.floating)):
                    row[k] = float(v)
                else:
                    row[k] = v
            if row["head"]:
                ranking.append(row)

    if not ranking:
        metrics = (mining.get("keyword_head_metrics") or {}).get(keyword) or {}
        for head_key, mets in metrics.items():
            if not isinstance(mets, dict):
                continue
            row = {"head": _normalize_head_key(head_key)}
            for k, v in mets.items():
                if isinstance(v, (int, float, np.floating)):
                    row[k] = float(v)
                else:
                    row[k] = v
            ranking.append(row)
        ranking.sort(
            key=lambda x: float(x.get("score", x.get("final_score", x.get("mean_score", 0.0)))),
            reverse=True,
        )

    if selected:
        best = selected[0]
    elif ranking:
        best = ranking[0]["head"]
    else:
        fallback = scan_cfg.get("default_items") or ["14_13"]
        best = _normalize_head_key(fallback[0])
        print(f"[mine] no scored head; fallback to {best}")

    print(f"[mine] best head for {keyword!r}: {best}")
    return best, mining, ranking


def normalize_heat(heat: np.ndarray) -> np.ndarray:
    heat = np.asarray(heat, dtype=np.float32)
    heat = heat - float(heat.min())
    denom = float(heat.max()) + 1e-8
    return (heat / denom).astype(np.float32)


def _resize_heat_to_image(heat: np.ndarray, img_arr: np.ndarray) -> np.ndarray:
    heat = np.clip(np.asarray(heat, dtype=np.float32), 0.0, 1.0)
    if heat.shape[:2] == img_arr.shape[:2]:
        return heat
    heat_u8 = (heat * 255).astype(np.uint8)
    return np.asarray(
        Image.fromarray(heat_u8).resize((img_arr.shape[1], img_arr.shape[0]), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0


def build_heatmap_overlay(img_arr: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Enhance-style pseudo-color overlay (red-high / blue-low)."""
    alpha = float(max(0.0, min(1.0, alpha)))
    img = np.asarray(img_arr, dtype=np.float32)
    heat = _resize_heat_to_image(heat, img_arr)
    red = heat * 255.0
    green = np.sqrt(heat) * 180.0
    blue = (1.0 - heat) * 220.0
    color = np.stack([red, green, blue], axis=-1)
    overlay = img * (1.0 - alpha) + color * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def build_jet_heatmap_on_image(
    img_arr: np.ndarray,
    heat: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """Jet colormap blended over the original image so the photo remains visible."""
    alpha = float(max(0.0, min(1.0, alpha)))
    img = np.asarray(img_arr, dtype=np.float32)
    heat = _resize_heat_to_image(heat, img_arr)
    # Boost mid/high responses so hotspots read more clearly on the photo.
    heat_vis = np.power(np.clip(heat, 0.0, 1.0), 0.75)
    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heat_vis)[..., :3] * 255.0
    # Stronger attention-weighted blend; low-attention still keeps some original.
    blend = np.clip(
        heat_vis * alpha + (1.0 - heat_vis) * (alpha * 0.55),
        0.0,
        1.0,
    )[..., None]
    out = img * (1.0 - blend) + heat_rgb * blend
    return np.clip(out, 0, 255).astype(np.uint8)


def render_figure(
    img_arr: np.ndarray,
    heat_on_image: np.ndarray,
    overlay: np.ndarray,
    out_base: Path,
    title: str,
    dpi: int = 160,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    panels = [
        (img_arr, "Original"),
        (heat_on_image, "Heatmap on image"),
        (overlay, "Overlay"),
    ]
    for ax, (data, label) in zip(axes, panels):
        ax.imshow(data)
        ax.set_title(label, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_png = Path(f"{out_base}.png")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png


def save_one_head(
    model,
    image_path: str,
    keyword: str,
    head_key: str,
    heat: np.ndarray,
    img_arr: np.ndarray,
    out_dir: Path,
    mode: str,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize, overlay, and write artifacts for one head."""
    heat = normalize_heat(heat)
    alpha = float(getattr(model, "heatmap_overlay_alpha", 0.45))
    heat_for_overlay = _resize_heat_to_image(heat, img_arr)
    if hasattr(model, "_build_heatmap_overlay"):
        overlay = model._build_heatmap_overlay(img_arr.astype(np.float32), heat_for_overlay)
    else:
        overlay = build_heatmap_overlay(img_arr, heat_for_overlay, alpha=alpha)

    # Jet heatmap blended on the original photo (faint original remains visible).
    heat_on_image = build_jet_heatmap_on_image(
        img_arr, heat_for_overlay, alpha=HEATMAP_ON_IMAGE_ALPHA
    )

    stem = Path(image_path).stem or "image"
    tag = _safe_tag(keyword)
    base = out_dir / f"{stem}_{tag}_head_{head_key}"

    heat_path = Path(f"{base}_heatmap.npy")
    overlay_path = Path(f"{base}_overlay.png")
    heat_png = Path(f"{base}_heatmap.png")
    np.save(heat_path, heat)
    Image.fromarray(overlay).save(overlay_path)
    Image.fromarray(heat_on_image).save(heat_png)

    fig_path = render_figure(
        img_arr=img_arr,
        heat_on_image=heat_on_image,
        overlay=overlay,
        out_base=base,
        title=f"{MODEL_NAME} | keyword={keyword!r} | head={head_key} | mode={mode}",
        dpi=DPI,
    )

    meta = {
        "model": MODEL_NAME,
        "image": image_path,
        "keyword": keyword,
        "head_mode": mode,
        "selected_head": head_key,
        "attention_type": ATTENTION_TYPE,
        "overlay_alpha": alpha,
        "artifacts": {
            "figure": str(fig_path),
            "overlay": str(overlay_path),
            "heatmap_png": str(heat_png),
            "heatmap_npy": str(heat_path),
        },
    }
    if extra_meta:
        meta.update(extra_meta)
    meta_path = Path(f"{base}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fout:
        json.dump(meta, fout, ensure_ascii=False, indent=2)
    print(f"[save] head={head_key} -> {fig_path}")
    return meta


def main():
    image_path = str(Path(IMAGE_PATH).resolve())
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    keyword = str(KEYWORD).strip()
    if not keyword:
        raise ValueError("KEYWORD must be non-empty.")

    out_dir = OUT_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] image={image_path}")
    print(f"[info] keyword={keyword!r}")
    print(f"[info] HEAD_MODE={HEAD_MODE}")
    print(f"[info] output_dir={out_dir}")

    model = build_model(MODEL_NAME)
    img_arr = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)

    ranking: list[dict[str, Any]] = []
    mining_meta: dict[str, Any] = {}
    head_maps: dict[str, np.ndarray] = {}
    heads_to_save: list[str] = []

    mode = str(HEAD_MODE).lower().strip()
    if mode in {"manual", "batch"}:
        if MULTI_HEADS:
            heads_to_save = [_normalize_head_key(h) for h in MULTI_HEADS if str(h).strip()]
        else:
            heads_to_save = [_normalize_head_key(MANUAL_HEAD)]
        print(f"[{mode}] exporting {len(heads_to_save)} heads: {heads_to_save}")
        for hk in heads_to_save:
            heat_i, _ = extract_single_head_map(
                model, image_path, keyword, hk, analysis_max_side=ANALYSIS_MAX_SIDE
            )
            head_maps[hk] = heat_i
    elif mode == "top_file":
        head_key = load_top_head_from_file(TOP_HEADS_PATH)
        heads_to_save = [head_key]
        print(f"[top_file] using head={head_key} from {TOP_HEADS_PATH}")
        heat_i, _ = extract_single_head_map(
            model, image_path, keyword, head_key, analysis_max_side=ANALYSIS_MAX_SIDE
        )
        head_maps[head_key] = heat_i
    elif mode in {"mine", "mine_candidates"}:
        scan_cfg = model._resolve_attention_scan_config()
        if mode == "mine":
            head_maps, scan_cfg, image_size = extract_all_head_maps(
                model, image_path, keyword, analysis_max_side=ANALYSIS_MAX_SIDE
            )
        else:
            candidates = resolve_candidate_heads()
            print(f"[mine_candidates] evaluating {len(candidates)} heads: {candidates}")
            head_maps = {}
            image_size = Image.open(image_path).convert("RGB").size
            for hk in candidates:
                heat_i, image_size = extract_single_head_map(
                    model,
                    image_path,
                    keyword,
                    hk,
                    analysis_max_side=ANALYSIS_MAX_SIDE,
                )
                head_maps[hk] = heat_i
        head_key, mining_meta, ranking = mine_best_head(
            model, image_path, keyword, head_maps, scan_cfg, image_size
        )
        # Prefer exporting all mining-selected heads when available.
        selected = [
            _normalize_head_key(h) for h in (mining_meta.get("selected_head_keys") or [])
        ]
        heads_to_save = selected if selected else [head_key]
        if MULTI_HEADS:
            extra = [_normalize_head_key(h) for h in MULTI_HEADS if str(h).strip()]
            for h in extra:
                if h not in heads_to_save:
                    heads_to_save.append(h)
                if h not in head_maps:
                    heat_i, _ = extract_single_head_map(
                        model,
                        image_path,
                        keyword,
                        h,
                        analysis_max_side=ANALYSIS_MAX_SIDE,
                    )
                    head_maps[h] = heat_i
    else:
        raise ValueError(
            "HEAD_MODE must be one of: mine / mine_candidates / manual / batch / top_file"
        )

    stem = Path(image_path).stem or "image"
    tag = _safe_tag(keyword)
    shared_meta: dict[str, Any] = {}
    if ranking and SAVE_ALL_HEAD_RANKING:
        rank_path = out_dir / f"{stem}_{tag}_head_ranking.json"
        with open(rank_path, "w", encoding="utf-8") as fout:
            json.dump(ranking[:50], fout, ensure_ascii=False, indent=2)
        shared_meta["ranking_path"] = str(rank_path)
        shared_meta["top5"] = ranking[:5]
    if mining_meta:
        shared_meta["mining_selected_heads"] = [
            _normalize_head_key(h) for h in (mining_meta.get("selected_head_keys") or [])
        ]
        shared_meta["used_default_fallback"] = bool(mining_meta.get("used_default_fallback"))

    for hk in heads_to_save:
        if hk not in head_maps:
            raise KeyError(f"Missing attention map for head {hk}")
        save_one_head(
            model=model,
            image_path=image_path,
            keyword=keyword,
            head_key=hk,
            heat=np.asarray(head_maps[hk], dtype=np.float32),
            img_arr=img_arr,
            out_dir=out_dir,
            mode=mode,
            extra_meta=shared_meta,
        )

    print(f"[done] exported heads={heads_to_save}")


if __name__ == "__main__":
    main()
