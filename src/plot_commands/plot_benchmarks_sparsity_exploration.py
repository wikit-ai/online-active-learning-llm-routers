from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset  # type: ignore
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import mannwhitneyu  # type: ignore
from sklearn.manifold import TSNE

from plot_commands.plotting_function.utils import setup_paper_rcparams
from logging_config import logger

TOLERANCE = 0.05
TOP_N = 10
MIN_PIE_PCT = 2.0

CMAP_DIVERGENCE = LinearSegmentedColormap.from_list(
    "teal_amber_red",
    [
        "#5DCAA5",
        "#1D9E75",
        "#E8C9A0",
        "#FAC775",
        "#EF9F27",
        "#E24B4A",
        "#A32D2D",
        "#701818",
    ],
)


def kl_batch(perf: np.ndarray) -> np.ndarray:
    """KL(row || Uniform) for a (n, m) performance matrix."""
    eps = 1e-10
    p = np.clip(perf, 0, None).astype(float)
    totals = p.sum(axis=1, keepdims=True)
    valid = totals.squeeze(1) > 0
    p = np.where(totals > 0, p / np.where(totals > 0, totals, 1), 0)
    p = np.where(p == 0, eps, p)
    H = -np.sum(p * np.log(p), axis=1)
    kls = np.log(perf.shape[1]) - H
    kls[~valid] = 0.0
    return kls


def _multi_palette_colors(n: int) -> list[Any]:
    """Build a list of n visually distinct colors drawn from multiple palettes."""
    palettes = ["tab20", "tab20b", "tab20c", "Set1", "Set2", "Set3", "gist_rainbow"]
    all_colors: list[Any] = []
    for p in palettes:
        cm = plt.colormaps[p]
        n_s = cm.N if hasattr(cm, "N") and cm.N <= 20 else 20
        all_colors.extend([cm(i) for i in np.linspace(0, 1, n_s)])
    rng = np.random.default_rng(42)
    rng.shuffle(all_colors)
    return all_colors[:n]


def _greedy_set_cover(
    P: np.ndarray, models_name: list[str]
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Greedily select the minimal set of models that covers all queries within TOLERANCE."""
    best_per_query = P.max(axis=1, keepdims=True)
    covered = P >= best_per_query - TOLERANCE
    coverage_frac = covered.mean(axis=0)

    uncovered: set[int] = set(range(len(P)))
    selected: list[int] = []
    while uncovered:
        best_idx = max(
            range(len(models_name)),
            key=lambda i: (sum(covered[q, i] for q in uncovered), coverage_frac[i]),
        )
        newly = {q for q in uncovered if covered[q, best_idx]}
        if not newly:
            break
        selected.append(best_idx)
        uncovered -= newly

    query_cover = np.full(len(P), -1, dtype=int)
    already: set[int] = set()
    for rank, idx in enumerate(selected):
        for q in range(len(P)):
            if q not in already and covered[q, idx]:
                query_cover[q] = rank
                already.add(q)

    return selected, covered, query_cover


BENCHMARKS = ["EmbedLLM", "RouterBench", "Sprout", "FusionBench"]
OUT_DIR = Path("plots_benchmarks") / "paper"

setup_paper_rcparams()

N_BENCH = len(BENCHMARKS)


def load_benchmark(benchmark: str) -> dict[str, Any]:
    """Load dataset, compute KL, run t-SNE and greedy set cover."""
    split_name = "EmbedLLM" if benchmark == "PartEmbedLLM" else benchmark
    logger.info(f"  [{benchmark}] Loading (split={split_name}) …")
    ds = load_dataset("Wikit/RoutingCompendium-perf", split=split_name)

    E = np.vstack(ds["embeddings"])  # type: ignore[index]
    P = np.vstack(ds["models_performance"])  # type: ignore[index]
    KL = kl_batch(P)
    models_name: list[str] = ds["models_name"][0]  # type: ignore[index]
    global_best = int(P.mean(axis=0).argmax())
    diff = P.max(axis=1) - P[:, global_best]

    logger.info(
        f"  [{benchmark}] n={len(E)}, m={P.shape[1]}, "
        f"E_dim={E.shape[1]}, mean_score={P.max(axis=1).mean():.3f}"
    )
    logger.info(f"  [{benchmark}] Running t-SNE …")
    E_2d = TSNE(n_components=2, random_state=0).fit_transform(E)

    logger.info(f"  [{benchmark}] Computing greedy set cover …")
    selected, _, query_cover = _greedy_set_cover(P, models_name)
    logger.info(f"  [{benchmark}] Cover size: {len(selected)} models")

    return dict(
        benchmark=benchmark,
        E_2d=E_2d,
        P=P,
        KL=KL,
        diff=diff,
        global_best=global_best,
        models_name=models_name,  # type: ignore
        selected=selected,
        query_cover=query_cover,
    )


def _cover_plot_data(data: dict[str, Any]) -> dict[str, Any]:
    """Precompute sorted color assignments and pie slices for one benchmark."""
    P, selected = data["P"], data["selected"]
    models_name, query_cover = data["models_name"], data["query_cover"]  # type: ignore
    n_sel = len(selected)  # type: ignore

    cover_counts = np.bincount(query_cover[query_cover >= 0], minlength=n_sel)  # type: ignore
    rank_order = np.argsort(cover_counts)[::-1]
    old_to_new = {old: new for new, old in enumerate(rank_order)}
    qcm_sorted = np.array([old_to_new[r] if r >= 0 else -1 for r in query_cover])
    colors = _multi_palette_colors(n_sel)

    # Newly covered queries per model (greedy order -> sorted by coverage)
    covered_mask = P >= P.max(axis=1, keepdims=True) - TOLERANCE  # type: ignore
    already: set[int] = set()
    pie_sizes = []
    for idx in selected:  # type: ignore
        newly = {q for q in range(len(P)) if covered_mask[q, idx]} - already  # type: ignore
        pie_sizes.append(len(newly))  # type: ignore
        already |= newly

    order = np.argsort(cover_counts)[::-1]
    pie_sizes_sorted = [pie_sizes[i] for i in order]
    pie_labels_sorted = [models_name[selected[i]].split("/")[-1] for i in order]  # type: ignore

    if n_sel > TOP_N:
        pie_sizes_final = pie_sizes_sorted[:TOP_N] + [sum(pie_sizes_sorted[TOP_N:])]  # type: ignore
        pie_labels_final = pie_labels_sorted[:TOP_N] + [f"Others ({n_sel - TOP_N})"]  # type: ignore
        pie_colors = colors[:TOP_N] + [(0.8, 0.8, 0.8, 1.0)]  # type: ignore
    else:
        pie_sizes_final, pie_labels_final = pie_sizes_sorted, pie_labels_sorted  # type: ignore
        pie_colors = colors[:n_sel]  # type: ignore

    return dict(  # type: ignore
        n_sel=n_sel,
        qcm_sorted=qcm_sorted,
        colors=colors,  # type: ignore
        cover_counts=cover_counts,
        rank_order=rank_order,
        pie_sizes_final=pie_sizes_final,  # type: ignore
        pie_labels_final=pie_labels_final,  # type: ignore
        pie_colors=pie_colors,  # type: ignore
        pie_sizes_sorted=pie_sizes_sorted,  # type: ignore
        pie_labels_sorted=pie_labels_sorted,  # type: ignore
    )


def _style_tsne_ax(ax: Axes, title: str, title_kw: dict[str, Any] | None = None) -> None:
    """Apply the shared t-SNE panel cosmetics: title, dim labels, no ticks, no spines."""
    ax.set_title(title, **(title_kw or {}))
    ax.set_xlabel("Dim. 1")
    ax.set_ylabel("Dim. 2")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_tsne_cover_ax(
    ax: Axes,
    E_2d: np.ndarray,
    qcm_sorted: np.ndarray,
    colors: list[Any],
    title: str,
    s: float = 3,
    title_kw: dict[str, Any] | None = None,
) -> None:
    """Scatter t-SNE points colored by their assigned covering model."""
    point_colors = [colors[r] if r >= 0 else (0.7, 0.7, 0.7, 1.0) for r in qcm_sorted]
    ax.scatter(E_2d[:, 0], E_2d[:, 1], c=point_colors, alpha=0.5, s=s, rasterized=True)  # type: ignore
    _style_tsne_ax(ax, title, title_kw)


def _draw_tsne_scalar_ax(
    ax: Axes,
    E_2d: np.ndarray,
    values: np.ndarray,
    title: str,
    cbar_label: str,
    s: float = 3,
    title_kw: dict[str, Any] | None = None,
) -> None:
    """Scatter t-SNE points colored by a scalar per query, with a colorbar."""
    sc = ax.scatter(  # type: ignore
        E_2d[:, 0],
        E_2d[:, 1],
        c=values,
        cmap=CMAP_DIVERGENCE,
        alpha=0.5,
        s=s,
        rasterized=True,
    )
    plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046, label=cbar_label)
    _style_tsne_ax(ax, title, title_kw)


def _draw_donut_ax(
    ax: Axes,
    pie_sizes_final: list[int],
    pie_colors: list[Any],
    n_sel: int,
    title: str,
    pct_fontsizes: tuple[float, float] = (9, 7),
    center_fontsize: float = 10,
    title_kw: dict[str, Any] | None = None,
) -> None:
    """Draw a donut pie chart showing cover-set composition with a center label."""
    total = sum(pie_sizes_final)

    def _autopct(pct: float) -> str:
        return f"{pct:.0f}%" if pct >= MIN_PIE_PCT else ""

    _, _, autotexts = ax.pie(
        pie_sizes_final,
        labels=None,
        autopct=_autopct,
        startangle=90,
        colors=pie_colors,
        pctdistance=0.78,
        wedgeprops=dict(linewidth=1.0, edgecolor="white"),
    )
    big, small = pct_fontsizes
    for at, size in zip(autotexts, pie_sizes_final):
        pct = size / total * 100
        at.set_fontsize(big if pct >= 5 else small)
        at.set_color("white" if pct >= 5 else "#333333")
    ax.add_artist(plt.Circle((0, 0), 0.50, fc="white"))
    ax.text(
        0,
        0,
        f"n={n_sel}\nmodels",
        ha="center",
        va="center",
        fontsize=center_fontsize,
        color="#333333",
    )
    ax.set_title(title, **(title_kw or {}))


def _draw_kl_hist_ax(ax: plt.Axes, KL: np.ndarray, title: str) -> None:
    """Ψ(q) histogram with mean."""
    ax.hist(KL, bins=40, color="steelblue", alpha=0.6, density=True)
    ax.axvline(
        KL.mean(), color="tomato", ls="--", lw=1.0, label=f"Mean = {KL.mean():.3f}"
    )
    ax.set_xlabel("Ψ(q)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    ax.set_title(title)


def _draw_gap_dist_ax(ax: plt.Axes, diff: np.ndarray, title: str) -> None:
    """Distribution of the routing gain (gap = best-per-query − global-best model)."""
    frac_positive = float((diff > 0).mean())
    mask_pos = diff > 0

    bins = np.histogram_bin_edges(diff, bins=40)
    ax.hist(
        diff[~mask_pos],
        bins=bins,
        color="#5DACBD",
        alpha=0.75,
        label=f"gap = 0  ({1 - frac_positive:.0%})",
    )
    ax.hist(
        diff[mask_pos],
        bins=bins,
        color="#E24B4A",
        alpha=0.75,
        label=f"gap > 0  ({frac_positive:.0%})",
    )
    ax.set_xlabel("Best-vs-Default Gap")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    ax.set_title(title)


def _draw_violin_ax(ax: plt.Axes, KL: np.ndarray, diff: np.ndarray, title: str) -> None:
    """Mann-Whitney violin: Ψ(q) ≤ P75 vs Ψ(q) > P75."""
    kl_p75 = np.percentile(KL, 75)
    high_mask = KL > kl_p75
    low_diff, high_diff = diff[~high_mask], diff[high_mask]

    _mwu = mannwhitneyu(high_diff, low_diff, alternative="greater")
    mwu_p = float(_mwu.pvalue)
    n_lo, n_hi = int((~high_mask).sum()), int(high_mask.sum())
    n_total = n_lo + n_hi
    mean_u = n_lo * n_hi / 2.0
    std_u = np.sqrt(n_lo * n_hi * (n_total + 1) / 12.0)
    mwu_z = (float(_mwu.statistic) - mean_u) / std_u
    effect_r = mwu_z / np.sqrt(n_total)
    sig = (
        "***"
        if mwu_p < 0.001
        else ("**" if mwu_p < 0.01 else ("*" if mwu_p < 0.05 else "n.s."))
    )

    vp = ax.violinplot(
        [low_diff, high_diff], positions=[0, 1], showmedians=True, showextrema=True
    )
    for body, col in zip(vp["bodies"], ["#5DACBD", "#E24B4A"]):
        body.set_facecolor(col)
        body.set_alpha(0.7)
    for part in ("cmedians", "cmins", "cmaxes", "cbars"):
        vp[part].set_color("black")
        vp[part].set_linewidth(1.2)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [f"Ψ(q) ≤ P75\n(n={n_lo:,})", f"Ψ(q) > P75\n(n={n_hi:,})"],
    )
    ax.set_ylabel("Best-vs-Default Gap")
    ax.set_title(f"{title}\np={mwu_p:.3f} {sig},  r={effect_r:.2f}")


# ── figure 1: t-SNE covering — all benchmarks ─────────────────────────────────


def plot_tsne_covering_all(all_data: list[dict], out_path: Path) -> None:
    """
    2 rows × N columns.
    Row 0: t-SNE coloured by covering model.
    Row 1: Donut pie of cover-set composition.
    """
    n = len(all_data)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.5))

    for col, data in enumerate(all_data):
        cd = _cover_plot_data(data)
        _draw_tsne_cover_ax(
            axes[0, col],
            data["E_2d"],
            cd["qcm_sorted"],
            cd["colors"],
            data["benchmark"],
        )
        _draw_donut_ax(
            axes[1, col],
            cd["pie_sizes_final"],
            cd["pie_colors"],
            cd["n_sel"],
            "Cover set",
        )

    plt.suptitle(
        "Embedding Clusters and Cover-Set Composition",
        fontsize=10,
        fontweight="bold",
        y=1.005,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")


# ── figure 2: KL distribution + gap distribution + violin — all benchmarks ────


def plot_kl_distribution_all(all_data: list[dict], out_path: Path) -> None:
    """
    3 rows × N columns.
    Row 0: KL histogram + KDE.
    Row 1: Routing-gain (gap) distribution — replaces bar chart from single-benchmark version.
    Row 2: Mann-Whitney violin (KL > Q75 -> higher routing gain?).
    """
    n = len(all_data)
    fig, axes = plt.subplots(3, n, figsize=(3.2 * n, 9.5))

    for col, data in enumerate(all_data):
        b = data["benchmark"]
        _draw_kl_hist_ax(axes[0, col], data["KL"], b)
        _draw_gap_dist_ax(axes[1, col], data["diff"], b)
        _draw_violin_ax(
            axes[2, col], data["KL"], data["diff"], f"{b}\nΨ(q) > P75 -> higher gap?"
        )

    plt.suptitle(
        "Routing Sparsity and Gain Analysis",
        fontsize=10,
        fontweight="bold",
        y=1.005,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")


# ── figure 3: t-SNE routing gain vs KL — all benchmarks ───────────────────────


def plot_tsne_kl_vs_diff_all(all_data: list[dict], out_path: Path) -> None:
    """
    2 rows × N columns.
    Row 0: t-SNE coloured by routing gain (best-per-query − global-best).
    Row 1: t-SNE coloured by KL divergence I(q).
    """
    n = len(all_data)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.5))

    for col, data in enumerate(all_data):
        b = data["benchmark"]
        E_2d = data["E_2d"]

        _draw_tsne_scalar_ax(
            axes[0, col],
            E_2d,
            data["diff"],
            f"{b}\nBest-vs-Default Gap",
            "Best-vs-Default Gap",
        )
        _draw_tsne_scalar_ax(axes[1, col], E_2d, data["KL"], f"{b}\nΨ(q)", "Ψ(q)")

    plt.suptitle(
        "Routing Gain and Sparsity in Embedding Space",
        fontsize=10,
        fontweight="bold",
        y=1.005,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    """Load all benchmarks and render the three multi-benchmark figures."""
    logger.info("Loading all benchmarks …")
    all_data = [load_benchmark(b) for b in BENCHMARKS]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("\nRendering figures …")

    plot_tsne_covering_all(all_data, OUT_DIR / "tsne_covering_all.png")
    plot_kl_distribution_all(all_data, OUT_DIR / "kl_distribution_all.png")
    plot_tsne_kl_vs_diff_all(all_data, OUT_DIR / "tsne_kl_vs_diff_all.png")

    logger.info(f"\nDone. All figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
