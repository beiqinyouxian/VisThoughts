"""Visualize head mining sample-size sensitivity results and export one LaTeX table.

Reads outputs under ./outputs/head_mining_analysis/ produced by run_head_mining_sample_analysis.py:
  reports/   - Layer1/Layer2 CSV & JSON summaries
  infer/     - per-N MME score breakdowns (optional enrichment)

Produces:
  figures/{dataset}_overview.{pdf,png}  — 3-panel figure (τ+Δτ, Jaccard, |Δ|%)
  reports/{dataset}_summary_table.tex   — one consolidated LaTeX table

Usage:
  python visualize_head_mining_analysis.py
  python visualize_head_mining_analysis.py --output-dir ./outputs/head_mining_analysis
  python visualize_head_mining_analysis.py --datasets MME --formats pdf png
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator

# Defaults aligned with run_head_mining_sample_analysis.py
DEFAULT_ROOT = Path("./outputs/head_mining_analysis")

# Strict reference thresholds (near-identical to full-set mining)
LAYER1_JACCARD_THRESH = 0.9
LAYER1_KENDALL_THRESH = 0.95
LAYER2_DELTA_THRESH = 0.5

# Practical operating point used in the paper / downstream experiments.
# N=200 is where ranking correlation plateaus (~0.8) and relative downstream
# deviation typically stays within a few percent of the N=500 reference.
# Jaccard on the discrete top-k set is shown for reference but is noisier,
# so the practical "OK" criterion emphasizes ranking + relative score error.
RECOMMENDED_N = 200
PRACTICAL_JACCARD = 0.5        # informative reference (≥2/3 top heads)
PRACTICAL_KENDALL = 0.79       # ranking already strongly aligned (elbow)
PRACTICAL_REL_DELTA_PCT = 3.1  # Δ_rel ≤ ε (%); ε used in figure (c) / Stage-2 criterion

# Short display names for figures / tables
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "llava_enhance_1_5_7b": "LLaVA-1.5-7B",
    "llava_enhance_1_5_13b": "LLaVA-1.5-13B",
    "qwen2_vl_enhance_2b_instruct": "Qwen2-VL-2B",
    "qwen2_vl_enhance_7b_instruct": "Qwen2-VL-7B",
    "qwen2_5_vl_enhance_3b_instruct": "Qwen2.5-VL-3B",
    "qwen2_5_vl_enhance_7b_instruct": "Qwen2.5-VL-7B",
}

# Consistent colors across panels (colorblind-friendly)
MODEL_COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # magenta
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize head mining analysis results.")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_ROOT,
        help="Root directory of head_mining_analysis outputs.",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Dataset names to process (default: auto-discover from reports/).",
    )
    parser.add_argument(
        "--formats", nargs="+", default=["pdf", "png"],
        choices=["pdf", "png", "svg"],
        help="Figure file formats to save.",
    )
    parser.add_argument(
        "--dpi", type=int, default=220, help="DPI for raster figure formats.",
    )
    parser.add_argument(
        "--recommended-n", type=int, default=RECOMMENDED_N,
        help="Mining sample size to highlight as the chosen operating point.",
    )
    return parser.parse_args()


def display_name(model_slug: str) -> str:
    if model_slug in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[model_slug]
    return model_slug.replace("_instruct", "").replace("_", "-")


def _normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def discover_groups(reports_dir: Path) -> dict[str, list[str]]:
    """Discover all {model_slug}_{dataset}_layer1_stability.csv and group by dataset."""
    groups: dict[str, list[str]] = {}
    for p in reports_dir.glob("*_layer1_stability.csv"):
        stem = p.name.replace("_layer1_stability.csv", "")
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        model_slug, dataset = parts[0], parts[1]
        groups.setdefault(dataset, []).append(model_slug)
    if not groups:
        raise FileNotFoundError(f"No *_layer1_stability.csv found under {reports_dir}")
    for ds in groups:
        groups[ds] = sorted(groups[ds], key=display_name)
    return groups


def load_layer1(reports_dir: Path, dataset: str, model_slug: str) -> tuple[pd.DataFrame, dict]:
    csv_path = reports_dir / f"{model_slug}_{dataset}_layer1_stability.csv"
    json_path = reports_dir / f"{model_slug}_{dataset}_layer1_stability.json"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing Layer1 report: {csv_path}")

    df = pd.read_csv(csv_path).sort_values("N")
    meta: dict = {"reference_N": None, "detail": {}, "top_k": 3, "scope": None}
    if json_path.exists():
        with open(json_path, encoding="utf-8") as fin:
            payload = json.load(fin)
        meta["reference_N"] = payload.get("reference_N")
        meta["detail"] = payload.get("detail", {})
        meta["scope"] = payload.get("scope")
        for col in df.columns:
            m = re.fullmatch(r"overlap@(\d+)", col)
            if m:
                meta["top_k"] = int(m.group(1))
                break
    return df, meta


def _find_infer_score(infer_dir: Path, model_slug: str, n: int, dataset: str) -> Path | None:
    """Match infer score CSV to the given model slug."""
    pattern = f"*_two_stage_n{n}_{dataset}_score.csv"
    candidates = sorted(infer_dir.glob(pattern))
    if not candidates:
        candidates = sorted(infer_dir.glob(f"*_two_stage_n{n}_*_score.csv"))
    if not candidates:
        return None

    target = _normalize_key(model_slug)
    best: Path | None = None
    best_score = -1
    for path in candidates:
        prefix = path.name.split("_two_stage_n", 1)[0]
        key = _normalize_key(prefix)
        if key == target:
            return path
        score = 0
        for a, b in zip(key, target):
            if a != b:
                break
            score += 1
        if target in key or key in target:
            score += 50
        if score > best_score:
            best_score = score
            best = path
    return best if best_score > 0 else candidates[0]


def load_layer2(reports_dir: Path, infer_dir: Path, dataset: str, model_slug: str) -> pd.DataFrame:
    csv_path = reports_dir / f"{model_slug}_{dataset}_layer2_downstream.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing Layer2 report: {csv_path}")

    df = pd.read_csv(csv_path).sort_values("N")
    enriched_rows = []
    for _, row in df.iterrows():
        n = int(row["N"])
        rec = row.to_dict()
        score_file = _find_infer_score(infer_dir, model_slug, n, dataset)
        if score_file is not None:
            scores = pd.read_csv(score_file)
            if len(scores) > 0:
                for col in scores.columns:
                    try:
                        rec[col] = float(scores[col].iloc[0])
                    except (TypeError, ValueError):
                        continue
                if "perception" in rec and "reasoning" in rec:
                    rec["total"] = float(rec["perception"]) + float(rec["reasoning"])
        enriched_rows.append(rec)

    out = pd.DataFrame(enriched_rows).sort_values("N")
    score_col = _primary_score_column(out)
    if not out.empty and score_col in out.columns:
        ref_n = int(out["N"].max())
        ref_val = float(out.loc[out["N"] == ref_n, score_col].iloc[0])
        out[f"delta_{score_col}"] = out[score_col].astype(float) - ref_val
    return out


def _primary_score_column(df: pd.DataFrame) -> str:
    for col in ("total", "perception", "reasoning"):
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            return col
    skip = {"dataset", "N", "heads"}
    numeric = [
        c for c in df.columns
        if c not in skip and not str(c).startswith("delta_")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return numeric[0] if numeric else "score"


def _relative_delta_pct(layer2: pd.DataFrame, score_col: str) -> pd.Series:
    """|Δ score| / reference score × 100, aligned to layer2 rows."""
    delta_col = f"delta_{score_col}"
    if score_col not in layer2.columns or delta_col not in layer2.columns:
        return pd.Series(np.nan, index=layer2.index)
    ref_n = int(layer2["N"].max())
    ref_val = float(layer2.loc[layer2["N"] == ref_n, score_col].iloc[0])
    if abs(ref_val) < 1e-9:
        return pd.Series(np.nan, index=layer2.index)
    return layer2[delta_col].astype(float).abs() / abs(ref_val) * 100.0


def is_practical_stable_row(
    jaccard: float | None,
    kendall: float | None,
    rel_delta_pct: float | None,
) -> bool:
    """Practical stability — ranking + error decisive; Jaccard recorded but not required."""
    del jaccard
    ken_ok = kendall is not None and not pd.isna(kendall) and float(kendall) >= PRACTICAL_KENDALL
    l2_ok = (
        rel_delta_pct is not None
        and not pd.isna(rel_delta_pct)
        and float(rel_delta_pct) <= PRACTICAL_REL_DELTA_PCT
    )
    return ken_ok and l2_ok


def _setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.titlepad": 10,
        "axes.labelpad": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save_fig(fig: plt.Figure, base_path: Path, formats: list[str], dpi: int):
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(base_path.with_suffix(f".{fmt}"), bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def _model_color_map(model_slugs: list[str]) -> dict[str, str]:
    return {slug: MODEL_COLORS[i % len(MODEL_COLORS)] for i, slug in enumerate(model_slugs)}


# ---------------------------------------------------------------------------
# Main visualization
# ---------------------------------------------------------------------------
def plot_overview(
    plot_data: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict]],
    dataset: str,
    out_base: Path,
    formats: list[str],
    dpi: int,
    recommended_n: int = RECOMMENDED_N,
):
    """One figure: (a) Kendall τ + marginal Δτ, (b) Jaccard, (c) |Δ|% heatmap."""
    if not plot_data:
        return

    model_slugs = list(plot_data.keys())
    colors = _model_color_map(model_slugs)
    first_meta = next(iter(plot_data.values()))[2]
    k = first_meta.get("top_k", 3)
    jac_col = f"jaccard@{k}"

    # Discrete, equal-spaced x positions (not proportional to raw N values).
    all_ns = sorted({int(n) for l1, l2, _ in plot_data.values() for n in list(l1["N"]) + list(l2["N"])})
    n_to_x = {n: i for i, n in enumerate(all_ns)}
    xs = np.arange(len(all_ns), dtype=float)
    rec_x = n_to_x.get(recommended_n)

    def _set_n_ticks(ax: plt.Axes):
        ax.set_xlim(-0.4, len(all_ns) - 0.6)
        ax.xaxis.set_major_locator(FixedLocator(list(xs)))
        ax.xaxis.set_major_formatter(FixedFormatter([str(n) for n in all_ns]))
        ax.set_autoscalex_on(False)

    def _n_to_xpos(series_n) -> np.ndarray:
        return np.asarray([n_to_x[int(n)] for n in series_n], dtype=float)

    # Top: ranking + overlap; bottom: narrower centered heatmap (room for side labels).
    fig = plt.figure(figsize=(14.0, 10.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 1.0], hspace=0.40, wspace=0.34)
    ax_ken = fig.add_subplot(gs[0, 0])
    ax_jac = fig.add_subplot(gs[0, 1])
    gs_hm = gs[1, :].subgridspec(1, 3, width_ratios=[0.18, 0.64, 0.18], wspace=0.08)
    ax_hm = fig.add_subplot(gs_hm[0, 1])

    # ---- (a) Kendall τ per model + mean marginal Δτ bars (merged former (d)) ----
    ken_matrix = np.full((len(model_slugs), len(all_ns)), np.nan)
    for i, slug in enumerate(model_slugs):
        l1, _, _ = plot_data[slug]
        x = _n_to_xpos(l1["N"].values)
        y = l1["kendall_tau"].values
        ax_ken.plot(x, y, "o-", color=colors[slug], linewidth=2.0, markersize=5.0, alpha=0.9, zorder=4)
        hit = l1.loc[l1["N"] == recommended_n]
        if len(hit) and rec_x is not None:
            ax_ken.plot(
                [rec_x], hit["kendall_tau"].values, "o", color=colors[slug],
                markersize=11, markeredgecolor="black", markeredgewidth=0.8, zorder=6,
            )
        for j, n in enumerate(all_ns):
            rows = l1.loc[l1["N"] == n, "kendall_tau"]
            if len(rows):
                ken_matrix[i, j] = float(rows.iloc[0])

    mean_ken = np.nanmean(ken_matrix, axis=0)

    # Marginal gain bars from cross-model mean τ (drawn on twin axis).
    mg_x, mg_heights = [], []
    for a_idx in range(len(all_ns) - 1):
        n1 = all_ns[a_idx]
        n2 = all_ns[a_idx + 1]
        delta_n = n2 - n1
        dtau = mean_ken[a_idx + 1] - mean_ken[a_idx]
        mg_per100 = dtau / delta_n * 100.0
        if np.isfinite(mg_per100):
            mg_x.append(float(a_idx) + 0.5)
            mg_heights.append(max(0.0, float(mg_per100)))

    raw_max = max(mg_heights) if mg_heights else 0.2
    ymax_mg = max(0.2, float(np.ceil(raw_max / 0.2) * 0.2))

    ax_mg = ax_ken.twinx()
    bar_container = ax_mg.bar(
        mg_x, mg_heights, width=0.42, color="#B0B0B0",
        edgecolor="white", linewidth=0.5, alpha=0.45, zorder=1, align="center",
        label=r"$\Delta\tau$ / 100 samples",
    )
    ax_mg.set_ylim(0.0, ymax_mg * 1.35)
    ax_mg.set_yticks(np.arange(0.0, ymax_mg + 1e-9, 0.2))
    ax_mg.set_ylabel(r"$\Delta\tau$ / 100 samples", color="#D55E00", fontsize=14)
    ax_mg.tick_params(axis="y", labelcolor="#D55E00", labelsize=11)
    ax_mg.spines["right"].set_color("#D55E00")
    ax_mg.xaxis.set_visible(False)
    # Keep categorical x ticks owned by the primary axis.
    ax_mg.set_xlim(-0.4, len(all_ns) - 0.6)

    ax_ken.set_ylim(0.30, 1.12)
    ax_ken.set_zorder(ax_mg.get_zorder() + 1)
    ax_ken.patch.set_visible(False)
    if rec_x is not None:
        ax_ken.axvline(rec_x, color="#C0392B", linestyle="--", linewidth=1.6, alpha=0.85, zorder=3)

    # Plateau callout anchored to mean τ at recommended N.
    if rec_x is not None and int(rec_x) + 1 < len(all_ns):
        next_n = all_ns[int(rec_x) + 1]
        dtau_after = mean_ken[int(rec_x) + 1] - mean_ken[int(rec_x)]
        delta_n_after = next_n - recommended_n
        ax_ken.annotate(
            rf"$\Delta\tau$={dtau_after:.3f} over next {delta_n_after}"
            + "\n"
            + r"(plateau)",
            xy=(rec_x, float(mean_ken[int(rec_x)])),
            xytext=(max(rec_x - 1.65, 0.15), min(float(mean_ken[int(rec_x)]) + 0.18, 1.05)),
            fontsize=14,
            color="#C0392B",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.2),
            bbox=dict(
                boxstyle="round,pad=0.28", facecolor="#fff5f5",
                edgecolor="#C0392B", alpha=0.95,
            ),
            zorder=10,
        )

    _set_n_ticks(ax_ken)
    ax_ken.set_xlabel("Mining sample size $N$", fontsize=16)
    ax_ken.set_ylabel(r"Kendall $\tau$", fontsize=16)
    legend_handles = [
        Line2D([0], [0], color=colors[s], marker="o", linewidth=2.0, markersize=5.5,
               label=display_name(s))
        for s in model_slugs
    ]
    legend_handles.append(
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="#B0B0B0", edgecolor="white", alpha=0.55,
            label=r"$\Delta\tau$ / 100 samples",
        )
    )
    ax_ken.legend(
        handles=legend_handles, loc="lower right", ncol=2,
        bbox_to_anchor=(1.0, 0.16),
        frameon=True, fancybox=False, framealpha=0.92,
        edgecolor="#cccccc", fontsize=11,
        columnspacing=0.9, handletextpad=0.35, handlelength=1.6,
        borderpad=0.35,
    )
    # Silence unused-var lint for bar_container when only drawn for side effect.
    _ = bar_container

    # ---- (b) Jaccard@k — clean mean trend (no per-model spaghetti) ----
    jac_matrix = np.full((len(model_slugs), len(all_ns)), np.nan)
    for i, slug in enumerate(model_slugs):
        l1, _, _ = plot_data[slug]
        if jac_col not in l1.columns:
            continue
        for j, n in enumerate(all_ns):
            rows = l1.loc[l1["N"] == n, jac_col]
            if len(rows):
                jac_matrix[i, j] = float(rows.iloc[0])

    mean_jac = np.nanmean(jac_matrix, axis=0)
    std_jac = np.nanstd(jac_matrix, axis=0)
    ax_jac.fill_between(
        xs, np.clip(mean_jac - std_jac, 0, 1), np.clip(mean_jac + std_jac, 0, 1),
        color="#0072B2", alpha=0.18, linewidth=0, zorder=2, label=r"$\pm$1 std",
    )
    ax_jac.plot(
        xs, mean_jac, "o-", color="#0072B2", linewidth=2.6, markersize=7,
        label=rf"Mean Jaccard@{k}", zorder=4,
    )
    if rec_x is not None:
        ax_jac.plot(
            [rec_x], [mean_jac[int(rec_x)]], "o", color="#0072B2",
            markersize=12, markeredgecolor="black", markeredgewidth=0.9, zorder=5,
        )
        ax_jac.axvline(rec_x, color="#C0392B", linestyle="--", linewidth=1.6, alpha=0.85, zorder=3)
        ax_jac.annotate(
            rf"$N={recommended_n}$: mean={mean_jac[int(rec_x)]:.2f}",
            xy=(rec_x, mean_jac[int(rec_x)]),
            xytext=(max(rec_x - 1.35, 0.4), min(mean_jac[int(rec_x)] + 0.16, 1.05)),
            fontsize=14, color="#C0392B",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.1),
            bbox=dict(boxstyle="round,pad=0.30", facecolor="#fff5f5",
                      edgecolor="#C0392B", alpha=0.95),
            zorder=6,
        )
    # Horizontal guide through mean Jaccard at recommended N (exact polyline point).
    if rec_x is not None and np.isfinite(mean_jac[int(rec_x)]):
        ax_jac.axhline(
            float(mean_jac[int(rec_x)]), color="#555555",
            linestyle=":", linewidth=1.2, zorder=3,
        )
    ax_jac.set_ylim(-0.05, 1.15)
    _set_n_ticks(ax_jac)
    ax_jac.set_xlabel("Mining sample size $N$", fontsize=16)
    ax_jac.set_ylabel(f"Jaccard@{k}", fontsize=16)
    ax_jac.legend(
        loc="lower right", bbox_to_anchor=(1.0, 0.12),
        frameon=True, fancybox=False,
        framealpha=0.75, facecolor="white", edgecolor="#cccccc",
        fontsize=14, borderaxespad=0.0,
    )

    # ---- (c) Relative |Δ|% heatmap (omit reference-N column; it is identically 0) ----
    ref_ns = {
        int(m.get("reference_N") or l1["N"].max())
        for l1, _, m in plot_data.values()
    }
    heat_ns = [n for n in all_ns if n not in ref_ns]
    mat_rel = np.full((len(model_slugs), len(heat_ns)), np.nan)
    for i, slug in enumerate(model_slugs):
        _, l2, _ = plot_data[slug]
        sc = _primary_score_column(l2)
        rel = _relative_delta_pct(l2, sc)
        for j, n in enumerate(heat_ns):
            rows = rel.loc[l2["N"] == n]
            if len(rows):
                mat_rel[i, j] = float(rows.iloc[0])

    finite = mat_rel[np.isfinite(mat_rel)]
    vmax = max(
        float(np.nanpercentile(finite, 95)) if finite.size else 5.0,
        PRACTICAL_REL_DELTA_PCT,
    )
    im = ax_hm.imshow(mat_rel, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=vmax)
    ax_hm.set_xticks(range(len(heat_ns)))
    ax_hm.set_xticklabels([str(n) for n in heat_ns])
    ax_hm.set_yticks(range(len(model_slugs)))
    # Hide default y tick labels; draw tilted ones manually (more reliable than tick props).
    ax_hm.set_yticklabels([])
    ax_hm.tick_params(axis="y", length=4, pad=2)
    for _i, _slug in enumerate(model_slugs):
        ax_hm.text(
            -0.03, _i - 0.18, display_name(_slug),
            transform=ax_hm.get_yaxis_transform(),  # x=axes fraction, y=data
            ha="right", va="center",
            rotation=55, rotation_mode="anchor",
            fontsize=12, fontfamily="Times New Roman",
            clip_on=False,
        )
    ax_hm.set_xlabel("Mining sample size $N$", fontsize=16)
    ax_hm.grid(False)

    if recommended_n in heat_ns:
        j_rec = heat_ns.index(recommended_n)
        ax_hm.add_patch(plt.Rectangle(
            (j_rec - 0.5, -0.5), 1.0, len(model_slugs),
            fill=False, edgecolor="#C0392B", linewidth=2.5, zorder=6,
        ))

    for i in range(mat_rel.shape[0]):
        for j in range(mat_rel.shape[1]):
            val = mat_rel[i, j]
            if not np.isfinite(val):
                continue
            is_rec = heat_ns[j] == recommended_n
            ax_hm.text(
                j, i, f"{val:.1f}", ha="center", va="center", fontsize=11,
                color="white" if val >= vmax * 0.55 else "#222222",
                fontweight="bold" if is_rec else "normal",
                fontfamily="Times New Roman",
            )
    # Colorbar is attached later (after layout settle) so label spacing is not reset by draw().
    im_for_cbar = im

    # Panel captions below each subplot (figure coords, after colorbar/twins settle).
    fig.canvas.draw()
    _caption_fs = 17
    _captions = [
        (
            ax_ken,
            f"(a) Head ranking correlation  (elbow at $N=${recommended_n})",
            0.078,
        ),
        (ax_jac, f"(b) Top-head set overlap  (mean Jaccard@{k})", 0.078),
        (
            ax_hm,
            rf"(c) Downstream relative error $\Delta_{{\mathrm{{rel}}}}$ (\%)  "
            rf"($\varepsilon{{=}}{PRACTICAL_REL_DELTA_PCT:g}$)",
            0.070,
        ),
    ]
    for _ax, _txt, _gap in _captions:
        _bbox = _ax.get_position()
        fig.text(
            _bbox.x0 + 0.5 * _bbox.width,
            _bbox.y0 - _gap,
            _txt,
            ha="center", va="top",
            fontsize=_caption_fs,
            fontfamily="Times New Roman",
        )

    # Place a free colorbar axis to the right of the heatmap with room for ticks + title.
    _hm_pos = ax_hm.get_position()
    cax = fig.add_axes([
        _hm_pos.x1 + 0.012,
        _hm_pos.y0,
        0.016,
        _hm_pos.height,
    ])
    cbar = fig.colorbar(im_for_cbar, cax=cax)
    cbar.set_label(r"$\Delta_{\mathrm{rel}}$ (\%)", fontsize=13)
    cbar.ax.tick_params(labelsize=12, pad=2)
    # Thin cax: labelpad is nearly ineffective; place title in axes coords
    # well beyond the tick labels (right edge of cax is x=1).
    cbar.ax.yaxis.set_label_coords(2.8, 0.5)

    _save_fig(fig, out_base, formats, dpi)


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------
def _latex_escape(text: str) -> str:
    repl = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}",
    }
    out = str(text)
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def _fmt_num(val, digits=3) -> str:
    if pd.isna(val):
        return "---"
    try:
        f = float(val)
    except (TypeError, ValueError):
        return _latex_escape(str(val))
    if abs(f - round(f)) < 1e-9 and abs(f) >= 10:
        return str(int(round(f)))
    return f"{f:.{digits}f}"


def build_unified_latex_table(
    plot_data: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict]],
    dataset: str,
    recommended_n: int = RECOMMENDED_N,
) -> str:
    """One table covering all models; recommended N rows are highlighted."""
    first_meta = next(iter(plot_data.values()))[2]
    k = first_meta.get("top_k", 3)
    first_l2 = next(iter(plot_data.values()))[1]
    score_col = _primary_score_column(first_l2)
    ref_ns = sorted({
        int(m.get("reference_N") or l1["N"].max())
        for l1, _, m in plot_data.values()
    })
    ref_n_str = ",".join(map(str, ref_ns))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Head-mining sample-size sensitivity on {dataset} "
        rf"(reference $N={ref_n_str}$, top-${k}$). "
        rf"We adopt $N={recommended_n}$: ranking correlation saturates "
        rf"(Kendall $\tau\geq{PRACTICAL_KENDALL}$), relative downstream "
        rf"deviation $\leq{PRACTICAL_REL_DELTA_PCT}\%$, and marginal "
        rf"gain drops to near-zero beyond this point.}}",
        rf"\label{{tab:head_mining_{dataset.lower()}_all}}",
        r"\small",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\begin{tabular}{l r ccc c cc c}",
        r"\toprule",
        rf"Model & $N$ & Overlap@{k} & Jaccard@{k} & Kendall $\tau$ "
        rf"& {_latex_escape(score_col)} & $\Delta$ & $|\Delta|$\% & OK \\",
        r"\midrule",
    ]

    for mi, (slug, (layer1, layer2, meta)) in enumerate(plot_data.items()):
        kk = meta.get("top_k", k)
        sc = _primary_score_column(layer2)
        dcol = f"delta_{sc}"
        jcol = f"jaccard@{kk}"
        ocol = f"overlap@{kk}"
        name = _latex_escape(display_name(slug))
        rel_pct = _relative_delta_pct(layer2, sc)
        layer2_tmp = layer2.copy()
        layer2_tmp["_rel_pct"] = rel_pct

        merged = layer1.merge(
            layer2_tmp[["N", sc, dcol, "_rel_pct"]].rename(
                columns={sc: "_score", dcol: "_delta"}
            ),
            on="N",
            how="left",
        ).sort_values("N")
        n_rows = len(merged)

        for ri, (_, row) in enumerate(merged.iterrows()):
            n = int(row["N"])
            overlap = _fmt_num(row.get(ocol), 2)
            jaccard = _fmt_num(row.get(jcol), 2)
            tau = _fmt_num(row.get("kendall_tau"), 3)
            score = _fmt_num(row.get("_score"), 1)
            delta = _fmt_num(row.get("_delta"), 1)
            rel = _fmt_num(row.get("_rel_pct"), 1)

            ok = is_practical_stable_row(
                row.get(jcol), row.get("kendall_tau"), row.get("_rel_pct"),
            )
            mark = r"\checkmark" if ok else "---"
            model_cell = rf"\multirow{{{n_rows}}}{{*}}{{{name}}}" if ri == 0 else ""

            if n == recommended_n:
                prefix = r"\rowcolor{blue!10} "
                n_cell = rf"\textbf{{{n}}}$^\star$"
            else:
                ref_n_model = int(meta.get("reference_N") or -1)
                prefix = r"\rowcolor{green!6} " if ok and n != ref_n_model else ""
                n_cell = str(n)

            lines.append(
                f"{prefix}{model_cell} & {n_cell} & {overlap} & {jaccard} "
                f"& {tau} & {score} & {delta} & {rel} & {mark} \\\\"
            )

        if mi < len(plot_data) - 1:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1mm}\\",
        rf"\footnotesize{{Requires \usepackage{{booktabs,multirow,colortbl,amssymb}}. "
        rf"Blue: chosen $N={recommended_n}$ ($^\star$). "
        rf"Green: satisfies Kendall$\geq{PRACTICAL_KENDALL}$ "
        rf"and $|\Delta|$\%$\leq{PRACTICAL_REL_DELTA_PCT}$.}}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def export_latex(
    plot_data: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict]],
    dataset: str,
    reports_dir: Path,
    recommended_n: int = RECOMMENDED_N,
):
    tex = build_unified_latex_table(plot_data, dataset, recommended_n=recommended_n)
    path = reports_dir / f"{dataset}_summary_table.tex"
    header = (
        f"% Auto-generated by visualize_head_mining_analysis.py\n"
        f"% Dataset: {dataset}, recommended N={recommended_n}, "
        f"models: {', '.join(display_name(s) for s in plot_data)}\n\n"
    )
    path.write_text(header + tex + "\n", encoding="utf-8")
    print(f"[latex] saved: {path}")


def process_dataset(
    dataset: str,
    model_slugs: list[str],
    root: Path,
    formats: list[str],
    dpi: int,
    recommended_n: int = RECOMMENDED_N,
):
    reports_dir = root / "reports"
    infer_dir = root / "infer"
    figures_dir = root / "figures"

    plot_data: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict]] = {}
    for model_slug in model_slugs:
        try:
            layer1, meta = load_layer1(reports_dir, dataset, model_slug)
            layer2 = load_layer2(reports_dir, infer_dir, dataset, model_slug)
            plot_data[model_slug] = (layer1, layer2, meta)
        except FileNotFoundError as exc:
            print(f"[warn] skip {model_slug}/{dataset}: {exc}")

    if not plot_data:
        print(f"[warn] no data loaded for dataset={dataset}")
        return

    _setup_style()
    plot_overview(
        plot_data,
        dataset,
        figures_dir / f"{dataset}_overview",
        formats,
        dpi,
        recommended_n=recommended_n,
    )
    export_latex(plot_data, dataset, reports_dir, recommended_n=recommended_n)
    print(f"[figures] saved: {figures_dir / (dataset + '_overview')}")


def main():
    args = parse_args()
    root = args.output_dir.resolve()
    reports_dir = root / "reports"
    if not reports_dir.is_dir():
        raise FileNotFoundError(f"Reports directory not found: {reports_dir}")

    groups = discover_groups(reports_dir)
    if args.datasets:
        groups = {k: v for k, v in groups.items() if k in args.datasets}

    print(f"[info] recommended N={args.recommended_n}")
    print(f"[info] processing groups: {{{', '.join(f'{k}: {len(v)}' for k, v in groups.items())}}}")
    for dataset, model_slugs in sorted(groups.items()):
        names = [display_name(s) for s in model_slugs]
        print(f"\n=== {dataset} ({len(model_slugs)} models: {names}) ===")
        process_dataset(
            dataset, model_slugs, root, args.formats, args.dpi,
            recommended_n=args.recommended_n,
        )
    print("\n[done] visualization complete.")


if __name__ == "__main__":
    main()
