"""Visualize per-keyword attention region extraction stages (reference-figure style).

Reuses the existing Qwen2.5-VL enhance attention / dominant-region pipeline without
modifying any library code under vlmeval/vlm/qwen2_vl.

Pipeline columns (per keyword row):
  A_k -> Binarize -> 8-connected components -> Select dominant region
      -> Hole filling + elliptical closing -> R*_k

Usage:
  CUDA_VISIBLE_DEVICES=0 python visualize_keyword_region_stages.py
  CUDA_VISIBLE_DEVICES=0,1,2,3 python visualize_keyword_region_stages.py \\
      --image /home/user/Pictures/sofa.png \\
      --keywords chair floor book blanket \\
      --model Qwen2-VL-Enhance-7B-Instruct
"""
from __future__ import annotations

import argparse
import colorsys
import os
import re
from pathlib import Path
from typing import Any, Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes

from vlmeval.config import supported_VLM
from vlmeval.vlm.qwen2_vl.auto_threshold import auto_otsu
from vlmeval.vlm.qwen2_vl.utils import (
    composite_attn_map,
    prepare_attention_maps_for_image,
    resize_to_square,
)

DEFAULT_IMAGE = "/home/user/Pictures/laptop.png"
DEFAULT_KEYWORDS = ["cup", "laptop"]
DEFAULT_MODEL = "Qwen2-VL-Enhance-7B-Instruct"
DEFAULT_OUTPUT = Path("./outputs/keyword_region_stages")
GENERAL_PROMPT = (
    "Write a general description of the image. "
    "Answer the question using a single word or phrase."
)

STAGE_KEYS = (
    "A_k",
    "binarize",
    "components",
    "select",
    "morph",
    "final",
)
STAGE_TITLES = {
    "A_k": r"$A_k$",
    "binarize": "Binarize",
    "components": "8-connected components",
    "select": "Select dominant region\n(peak-containing or largest attention mass)",
    "morph": "Convex hull +\nsmooth closing",
    "final": r"$R_k^\ast$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-keyword attention region stage visualization."
    )
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE, help="Input image path.")
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=DEFAULT_KEYWORDS,
        help="Keywords to visualize (space-separated).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="VLMEvalKit model name in supported_VLM.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for stage figures and per-keyword artifacts.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
        help="Grid figure formats.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="DPI for raster outputs.")
    parser.add_argument(
        "--head-fallback-top-n",
        type=int,
        default=4,
        help="If selected heads miss a keyword, use that keyword's top-N heads by mean mass.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1280 * 28 * 28,
        help=(
            "Processor max_pixels for attention extraction. "
            "Lower this on OOM; default is conservative for large sofa images."
        ),
    )
    parser.add_argument(
        "--analysis-max-side",
        type=int,
        default=1024,
        help=(
            "Max side length of the square analysis image before attention forward. "
            "Heatmaps are still remapped to the original image size."
        ),
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip model inference and redraw stages from existing per_keyword/*/heatmap.npy.",
    )
    return parser.parse_args()


def _safe_tag(text: str, index: int = 0) -> str:
    raw = str(text or "").strip().lower()
    safe = re.sub(r"[^0-9a-zA-Z_-]+", "_", raw).strip("_")
    if not safe:
        safe = f"kw_{index:02d}"
    return f"{index:02d}_{safe}"


def build_model(model_name: str, max_pixels: int | None = None):
    if model_name not in supported_VLM:
        raise KeyError(
            f"Unknown model {model_name!r}. Available enhance entries include: "
            f"{[k for k in supported_VLM if 'Enhance' in k]}"
        )
    print(f"[info] loading model: {model_name}")
    kwargs = {"verbose": False}
    if max_pixels is not None:
        kwargs["max_pixels"] = int(max_pixels)
        print(f"[info] override max_pixels={max_pixels}")
    model = supported_VLM[model_name](**kwargs)
    return model


def _prepare_analysis_image(current_image: Image.Image, analysis_max_side: int) -> Image.Image:
    """Square-pad then optionally downscale for attention memory budget."""
    analysis_image = resize_to_square(current_image, current_image.size)
    max_side = int(max(1, analysis_max_side))
    w, h = analysis_image.size
    if max(w, h) > max_side:
        analysis_image = analysis_image.resize((max_side, max_side), Image.BILINEAR)
        print(f"[info] analysis image resized {w}x{h} -> {max_side}x{max_side}")
    return analysis_image


def extract_keyword_attention_maps(
    model,
    image_path: str,
    keywords: list[str],
    analysis_max_side: int = 1024,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], tuple[int, int]]:
    """Round-1 attention extraction aligned with _forward_attention_for_keywords."""
    current_image = Image.open(image_path).convert("RGB")
    analysis_image = _prepare_analysis_image(current_image, analysis_max_side)
    image_size = current_image.size  # (W, H)

    scan_cfg = model._resolve_attention_scan_config()
    map_func = model._resolve_attention_map_func()

    keyword_att_maps: dict[str, dict[str, np.ndarray]] = {}
    for kw in keywords:
        print(f"[extract] keyword={kw!r} scanning all heads ...")
        raw = map_func(
            analysis_image,
            str(kw),
            GENERAL_PROMPT,
            model.model,
            model.processor,
            scan_cfg["layers"],
            scan_cfg["heads"],
        )
        keyword_att_maps[kw] = prepare_attention_maps_for_image(raw, image_size=image_size)
        print(f"[extract] keyword={kw!r} heads={len(keyword_att_maps[kw])}")

    return keyword_att_maps, scan_cfg, image_size


def mine_selected_heads(
    model,
    image_path: str,
    keywords: list[str],
    keyword_att_maps: dict[str, dict[str, np.ndarray]],
    scan_cfg: dict[str, Any],
    image_size,
) -> list[str]:
    """Mine shared head set via hybrid pseudo-GT (OWLv2/SAM) when available."""
    pseudo_gts: dict[str, dict[str, Any]] = {}
    for kw in keywords:
        print(f"[mine] building hybrid pseudo-GT for {kw!r} ...")
        pseudo_gts[kw] = model._build_hybrid_pseudo_gt_for_keyword(
            image_path=image_path,
            keyword=kw,
            image_size=image_size,
            fallback_heatmap=None,
        )
        src = pseudo_gts[kw].get("source", "none")
        has_mask = int(np.asarray(pseudo_gts[kw].get("mask_gt", np.zeros((1, 1)))).sum()) > 0
        print(f"[mine] keyword={kw!r} source={src} has_mask={has_mask}")

    mining = model._mine_heads_for_keywords(keyword_att_maps, pseudo_gts, scan_cfg)
    selected = list(mining.get("selected_head_keys") or [])
    if not selected or mining.get("used_default_fallback"):
        fallback = scan_cfg["default_items"][: max(1, int(getattr(model, "head_num", 4)))]
        # default_items may use "layer-head"; normalize to "layer_head"
        selected = [str(h).replace("-", "_") for h in fallback]
        print(f"[mine] using default/fallback heads: {selected}")
    else:
        selected = [str(h).replace("-", "_") for h in selected]
        print(f"[mine] selected heads: {selected}")
    return selected


def composite_keyword_heatmaps(
    keywords: list[str],
    keyword_att_maps: dict[str, dict[str, np.ndarray]],
    selected_heads: list[str],
    head_fallback_top_n: int = 4,
) -> dict[str, np.ndarray]:
    """Fuse selected heads per keyword; fall back to local top-N if miss."""
    out: dict[str, np.ndarray] = {}
    for kw in keywords:
        maps = keyword_att_maps.get(kw, {})
        # Normalize map keys too
        maps_norm = {str(k).replace("-", "_"): v for k, v in maps.items()}
        chosen = [maps_norm[h] for h in selected_heads if h in maps_norm]
        if not chosen:
            ranked = sorted(
                maps_norm.items(),
                key=lambda kv: float(np.asarray(kv[1], dtype=np.float32).mean()),
                reverse=True,
            )
            chosen = [v for _, v in ranked[: max(1, int(head_fallback_top_n))]]
            print(
                f"[compose] keyword={kw!r} missed selected heads; "
                f"fallback top-{len(chosen)} by mean mass"
            )
        heatmap = composite_attn_map(chosen)
        heat = np.asarray(heatmap, dtype=np.float32)
        heat = heat - float(heat.min())
        denom = float(heat.max()) + 1e-8
        out[kw] = (heat / denom).astype(np.float32)
    return out


def binarize_heatmap(
    heat: np.ndarray,
    method: str = "otsu",
    quantile: float = 0.8,
) -> np.ndarray:
    heat = np.asarray(heat, dtype=np.float32)
    if method == "quantile":
        thr = float(np.quantile(heat, float(max(0.0, min(1.0, quantile)))))
        return (heat >= thr).astype(np.uint8)
    return auto_otsu(heat).astype(np.uint8)


def _component_palette(num_labels: int) -> np.ndarray:
    """Distinct RGB colors for labels 1..N-1; label 0 stays black."""
    colors = np.zeros((max(1, num_labels), 3), dtype=np.uint8)
    for i in range(1, num_labels):
        hue = ((i - 1) * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
        colors[i] = (int(r * 255), int(g * 255), int(b * 255))
    return colors


def default_dominant_selector(
    labels: np.ndarray,
    stats: np.ndarray,
    heat: np.ndarray,
    criterion: str = "integrated",
) -> int:
    """Default criterion mirroring _extract_dominant_region_mask."""
    num_labels = int(labels.max()) + 1
    if num_labels <= 1:
        return -1

    best_label = -1
    if criterion == "peak":
        peak_idx = int(np.argmax(heat))
        py, px = np.unravel_index(peak_idx, heat.shape)
        peak_label = int(labels[py, px])
        best_label = peak_label if peak_label > 0 else -1

    if best_label <= 0:
        best_score = -1.0
        for label in range(1, num_labels):
            comp = labels == label
            if criterion == "area":
                score = float(stats[label, cv2.CC_STAT_AREA])
            else:
                score = float(heat[comp].sum())
            if score > best_score:
                best_score = score
                best_label = label
    return int(best_label)


def select_dominant_region(
    labels: np.ndarray,
    stats: np.ndarray,
    heat: np.ndarray,
    criterion: str = "integrated",
    selector: Callable[..., int] | None = None,
) -> int:
    """Select dominant connected component.

    ``selector`` is an optional hook for API / external VLM-assisted ranking.
    It must accept (labels, stats, heat) and return a label id (>0) or -1.
    """
    if selector is not None:
        return int(selector(labels=labels, stats=stats, heat=heat))
    return default_dominant_selector(labels, stats, heat, criterion=criterion)


def draw_star_marker(rgb: np.ndarray, cy: float, cx: float, radius: int = 14) -> np.ndarray:
    """Draw a filled white 5-point star near (cy, cx) on an RGB uint8 image."""
    out = rgb.copy()
    h, w = out.shape[:2]
    cx_i, cy_i = float(cx), float(cy)
    # Build star polygon in image coords
    pts = []
    for i in range(10):
        ang = -np.pi / 2 + i * np.pi / 5
        r = radius if i % 2 == 0 else radius * 0.45
        x = cx_i + r * np.cos(ang)
        y = cy_i + r * np.sin(ang)
        pts.append([int(round(x)), int(round(y))])
    pts_arr = np.asarray(pts, dtype=np.int32)
    cv2.fillPoly(out, [pts_arr], (255, 255, 255))
    cv2.polylines(out, [pts_arr], isClosed=True, color=(40, 40, 40), thickness=1)
    # Clamp draw area safety: if polygon outside, just put a circle
    if pts_arr[:, 0].min() < 0 or pts_arr[:, 1].min() < 0 or pts_arr[:, 0].max() >= w or pts_arr[:, 1].max() >= h:
        cx_c = int(np.clip(cx_i, 0, w - 1))
        cy_c = int(np.clip(cy_i, 0, h - 1))
        cv2.drawMarker(out, (cx_c, cy_c), (255, 255, 255), markerType=cv2.MARKER_STAR, markerSize=radius * 2, thickness=2)
    return out


def build_heatmap_overlay(img_arr: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Local copy of the existing lightweight heatmap overlay for reuse-only mode."""
    alpha = float(max(0.0, min(1.0, alpha)))
    img = np.asarray(img_arr, dtype=np.float32)
    heat = np.clip(np.asarray(heat, dtype=np.float32), 0.0, 1.0)
    red = heat * 255.0
    green = np.sqrt(heat) * 180.0
    blue = (1.0 - heat) * 220.0
    color = np.stack([red, green, blue], axis=-1)
    overlay = img * (1.0 - alpha) + color * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def convex_hull_smooth_mask(mask: np.ndarray) -> np.ndarray:
    """Return the minimal convex outer mask with smoothed edges.

    The convex hull removes tiny boundary spikes and holes by wrapping the
    selected component. A light elliptical close plus blur-threshold pass then
    rounds staircase artifacts introduced by rasterization.
    """
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    if mask_u8.sum() <= 0:
        return mask_u8.astype(bool)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = [cnt.reshape(-1, 2) for cnt in contours if cnt.size >= 6]
    if not points:
        return mask_u8.astype(bool)

    all_points = np.concatenate(points, axis=0).astype(np.int32)
    hull = cv2.convexHull(all_points)
    hull_mask = np.zeros_like(mask_u8)
    cv2.fillConvexPoly(hull_mask, hull, 1)

    min_side = min(mask_u8.shape[:2])
    close_ksize = max(9, (min_side // 90) | 1)
    smooth_ksize = max(11, (min_side // 70) | 1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    hull_mask = cv2.morphologyEx(hull_mask, cv2.MORPH_CLOSE, close_kernel)

    blurred = cv2.GaussianBlur((hull_mask * 255).astype(np.uint8), (smooth_ksize, smooth_ksize), 0)
    smoothed = (blurred >= 96).astype(np.uint8)

    # Keep the result as an outer envelope after smoothing.
    smoothed = np.logical_or(smoothed > 0, hull_mask > 0)
    return smoothed.astype(bool)


def run_region_stages(
    heat: np.ndarray,
    img_arr: np.ndarray,
    model,
    selector: Callable[..., int] | None = None,
) -> dict[str, np.ndarray]:
    """Produce all stage visualizations for one keyword heatmap."""
    h, w = img_arr.shape[:2]
    heat_u8 = (np.clip(heat, 0.0, 1.0) * 255).astype(np.uint8)
    heat_resized = np.asarray(
        Image.fromarray(heat_u8).resize((w, h), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0

    # Stage 1: A_k overlay
    if hasattr(model, "_build_heatmap_overlay"):
        overlay = model._build_heatmap_overlay(img_arr.astype(np.float32), heat_resized)
    else:
        overlay = build_heatmap_overlay(
            img_arr.astype(np.float32),
            heat_resized,
            alpha=float(getattr(model, "heatmap_overlay_alpha", 0.45)),
        )

    # Stage 2: binarize
    method = str(getattr(model, "keyword_region_binarize", "otsu")).lower()
    quantile = float(getattr(model, "attention_threshold_quantile", 0.8))
    binary = binarize_heatmap(heat_resized, method=method, quantile=quantile)

    # Stage 3: 8-connected components
    if binary.sum() <= 0:
        labels = np.zeros_like(binary, dtype=np.int32)
        stats = np.zeros((1, 5), dtype=np.int32)
        num_labels = 1
    else:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    palette = _component_palette(num_labels)
    components_rgb = palette[labels]

    # Stage 4: select dominant
    criterion = str(getattr(model, "keyword_region_criterion", "integrated")).lower()
    best_label = select_dominant_region(
        labels=labels,
        stats=stats,
        heat=heat_resized,
        criterion=criterion,
        selector=selector,
    )
    select_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    if best_label > 0:
        select_rgb[labels == best_label] = (220, 50, 50)
        # dim other components for context
        other = (labels > 0) & (labels != best_label)
        select_rgb[other] = (80, 80, 80)
        # star at peak inside selected region (or overall peak if fallthrough)
        region = labels == best_label
        masked_heat = np.where(region, heat_resized, -1.0)
        peak_idx = int(np.argmax(masked_heat))
        py, px = np.unravel_index(peak_idx, masked_heat.shape)
        star_r = max(10, min(h, w) // 40)
        select_rgb = draw_star_marker(select_rgb, float(py), float(px), radius=star_r)

    # Stage 5 + 6: convex outer envelope + final
    final_mask = np.zeros((h, w), dtype=bool)
    morph_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    if best_label > 0:
        mask = labels == best_label
        if bool(getattr(model, "keyword_region_morph", True)):
            mask = binary_fill_holes(mask)
            mask = convex_hull_smooth_mask(mask)
        final_mask = mask
        morph_rgb[mask] = (255, 255, 255)

    # Stage 6 visualization: white-mask pixels are transparent (original image
    # remains unchanged); black-mask pixels retain only a faint background.
    background_alpha = float(
        np.clip(getattr(model, "mask_background_alpha", 0.2), 0.0, 1.0)
    )
    final_rgb = np.where(
        final_mask[..., None],
        img_arr,
        img_arr.astype(np.float32) * background_alpha,
    )
    final_rgb = np.clip(final_rgb, 0, 255).astype(np.uint8)

    return {
        "A_k": overlay,
        "binarize": np.stack([binary * 255] * 3, axis=-1).astype(np.uint8),
        "components": components_rgb.astype(np.uint8),
        "select": select_rgb,
        "morph": morph_rgb,
        "final": final_rgb.astype(np.uint8),
        "heat": heat_resized,
        "binary": binary,
        "labels": labels,
        "best_label": np.asarray([best_label]),
        "final_mask": final_mask.astype(np.uint8),
    }


def save_keyword_artifacts(
    out_dir: Path,
    keyword: str,
    index: int,
    stages: dict[str, np.ndarray],
) -> None:
    tag = _safe_tag(keyword, index)
    kw_dir = out_dir / "per_keyword" / tag
    kw_dir.mkdir(parents=True, exist_ok=True)
    for key in STAGE_KEYS:
        Image.fromarray(stages[key]).save(kw_dir / f"{key}.png")
    np.save(kw_dir / "heatmap.npy", stages["heat"])
    np.save(kw_dir / "binary.npy", stages["binary"])
    np.save(kw_dir / "labels.npy", stages["labels"])
    np.save(kw_dir / "final_mask.npy", stages["final_mask"])


def render_grid(
    keywords: list[str],
    all_stages: dict[str, dict[str, np.ndarray]],
    out_base: Path,
    formats: list[str],
    dpi: int,
) -> None:
    n_rows = len(keywords)
    n_cols = len(STAGE_KEYS)
    fig_w = 2.6 * n_cols + 1.2
    fig_h = 2.4 * n_rows + 0.8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, kw in enumerate(keywords):
        stages = all_stages[kw]
        for c, key in enumerate(STAGE_KEYS):
            ax = axes[r, c]
            ax.imshow(stages[key])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(STAGE_TITLES[key], fontsize=9, pad=6)
            if c == 0:
                ax.set_ylabel(kw, fontsize=11, rotation=0, labelpad=28, va="center", fontweight="bold")

    fig.suptitle(
        "Optional per-keyword dominant region extraction",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=[0.02, 0.01, 1.0, 0.97])
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_base.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[save] {path}")
    plt.close(fig)


def merge_final_masks(
    keywords: list[str],
    all_stages: dict[str, dict[str, np.ndarray]],
    img_arr: np.ndarray,
    background_alpha: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Union all keyword final masks and overlay them on the original image."""
    h, w = img_arr.shape[:2]
    combined = np.zeros((h, w), dtype=bool)
    for kw in keywords:
        mask = np.asarray(all_stages[kw].get("final_mask", []), dtype=np.uint8) > 0
        if mask.size == 0:
            continue
        if mask.shape != (h, w):
            mask = np.asarray(
                Image.fromarray((mask.astype(np.uint8) * 255)).resize((w, h), Image.NEAREST)
            ) > 0
        combined |= mask

    alpha = float(np.clip(background_alpha, 0.0, 1.0))
    overlay = np.where(
        combined[..., None],
        img_arr,
        img_arr.astype(np.float32) * alpha,
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay, combined.astype(np.uint8)


def render_final_mask_summary(
    keywords: list[str],
    all_stages: dict[str, dict[str, np.ndarray]],
    img_arr: np.ndarray,
    out_base: Path,
    formats: list[str],
    dpi: int,
    background_alpha: float = 0.2,
) -> np.ndarray:
    """Render one fused overlay with all keyword final masks on the original image."""
    fused, combined = merge_final_masks(
        keywords,
        all_stages,
        img_arr,
        background_alpha=background_alpha,
    )

    out_base.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(fused).save(out_base.with_suffix(".png"))
    print(f"[save] {out_base.with_suffix('.png')}")
    np.save(out_base.with_name(out_base.name + "_mask.npy"), combined)
    print(f"[save] {out_base.with_name(out_base.name + '_mask.npy')}")

    # Optional PDF/SVG via matplotlib (single-panel figure).
    if "pdf" in formats or "svg" in formats:
        fig, ax = plt.subplots(figsize=(8.0, 5.5))
        ax.imshow(fused)
        ax.set_title("Fused final masks on original image", fontsize=13, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout()
        for fmt in formats:
            if fmt == "png":
                continue
            path = out_base.with_suffix(f".{fmt}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            print(f"[save] {path}")
        plt.close(fig)

    return combined


class ReuseStageConfig:
    """Minimal config object for redrawing stages from saved heatmaps."""

    keyword_region_binarize = "otsu"
    attention_threshold_quantile = 0.8
    keyword_region_criterion = "integrated"
    keyword_region_morph = True
    heatmap_overlay_alpha = 0.45
    mask_background_alpha = 0.2


def main():
    args = parse_args()
    image_path = os.path.abspath(args.image)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    keywords = [str(k).strip() for k in args.keywords if str(k).strip()]
    if not keywords:
        raise ValueError("At least one keyword is required.")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] image={image_path}")
    print(f"[info] keywords={keywords}")
    print(f"[info] output_dir={out_dir}")

    if args.reuse_existing:
        print("[info] reuse-existing mode: loading per_keyword/*/heatmap.npy")
        model = ReuseStageConfig()
        selected_heads = []
        heatmaps = {}
        for idx, kw in enumerate(keywords):
            heat_path = out_dir / "per_keyword" / _safe_tag(kw, idx) / "heatmap.npy"
            if not heat_path.exists():
                raise FileNotFoundError(f"Missing saved heatmap: {heat_path}")
            heatmaps[kw] = np.load(heat_path).astype(np.float32)
    else:
        model = build_model(args.model, max_pixels=args.max_pixels)

        keyword_att_maps, scan_cfg, image_size = extract_keyword_attention_maps(
            model, image_path, keywords, analysis_max_side=args.analysis_max_side
        )
        selected_heads = mine_selected_heads(
            model, image_path, keywords, keyword_att_maps, scan_cfg, image_size
        )
        heatmaps = composite_keyword_heatmaps(
            keywords,
            keyword_att_maps,
            selected_heads,
            head_fallback_top_n=args.head_fallback_top_n,
        )

    img_arr = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    all_stages: dict[str, dict[str, np.ndarray]] = {}
    for idx, kw in enumerate(keywords):
        print(f"[stages] keyword={kw!r}")
        stages = run_region_stages(heatmaps[kw], img_arr, model, selector=None)
        all_stages[kw] = stages
        save_keyword_artifacts(out_dir, kw, idx, stages)

    stem = Path(image_path).stem or "image"
    render_grid(
        keywords,
        all_stages,
        out_base=out_dir / f"{stem}_keyword_region_stages",
        formats=args.formats,
        dpi=args.dpi,
    )
    render_final_mask_summary(
        keywords,
        all_stages,
        img_arr=img_arr,
        out_base=out_dir / f"{stem}_keyword_final_masks",
        formats=args.formats,
        dpi=args.dpi,
        background_alpha=float(getattr(model, "mask_background_alpha", 0.2)),
    )

    meta = {
        "image": image_path,
        "keywords": keywords,
        "model": args.model,
        "selected_heads": selected_heads,
        "keyword_region_binarize": getattr(model, "keyword_region_binarize", None),
        "keyword_region_criterion": getattr(model, "keyword_region_criterion", None),
        "keyword_region_morph": getattr(model, "keyword_region_morph", None),
        "final_region_postprocess": "convex_hull_smooth_outer_envelope",
    }
    meta_path = out_dir / f"{stem}_meta.json"
    import json

    with open(meta_path, "w", encoding="utf-8") as fout:
        json.dump(meta, fout, ensure_ascii=False, indent=2)
    print(f"[save] {meta_path}")
    print("[done] keyword region stage visualization complete.")


if __name__ == "__main__":
    main()
