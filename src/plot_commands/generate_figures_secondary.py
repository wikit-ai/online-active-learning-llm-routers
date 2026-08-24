"""
Usage (run from src/):

    python -m plot_commands.generate_figures_secondary `
    --secondary-path results/secondary_experiments/aggregated_RouterBench_{timestamp}.json `
    --output-dir     plots_secondary `
    --k              40

"""

from typing import Any
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from logging_config import logger

plt.style.use("ggplot")

ACL_FIGSIZE = (3.33, 2.6)
ACL_DPI = 300
LABEL_FONTSIZE = 7
TICK_FONTSIZE = 6
TITLE_FONTSIZE = 7.5
ANNOT_FONTSIZE = 5.5


COLD_START_ORDER: list[str] = [
    "cold_start_25",
    "cold_start_50",
    "cold_start_75",
    "cold_start_115",
    "cold_start_150",
    "cold_start_200",
]
COLD_START_NAMES: dict[str, str] = {
    "cold_start_25": "n=25",
    "cold_start_50": "n=50",
    "cold_start_75": "n=75",
    "cold_start_115": "n=115",
    "cold_start_150": "n=150",
    "cold_start_200": "n=200",
}

GUARDRAIL_ORDER: list[str] = [
    "guardrail_off",
    "guardrail_0.10",
    "guardrail_0.20",
    "guardrail_0.30",
    "guardrail_0.50",
]
GUARDRAIL_NAMES: dict[str, str] = {
    "guardrail_off": "Off",
    "guardrail_0.10": "τ=0.10",
    "guardrail_0.20": "τ=0.20",
    "guardrail_0.30": "τ=0.30",
    "guardrail_0.50": "τ=0.50",
}

RETRAIN_ORDER: list[str] = [
    "retrain_every_1",
    "retrain_every_5",
    "retrain_every_10",
    "retrain_every_20",
    "retrain_every_50",
]
RETRAIN_NAMES: dict[str, str] = {
    "retrain_every_1": "Every 1",
    "retrain_every_5": "Every 5",
    "retrain_every_10": "Every 10",
    "retrain_every_20": "Every 20",
    "retrain_every_50": "Every 50",
}

_ABLATION_CMAP = plt.cm.viridis  # type: ignore


def load_secondary_experiment(path: Path, exp_type: str) -> dict[str, dict[str, Any]]:
    """Load a single experiment type from the aggregated secondary-experiments JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["aggregated"][exp_type]


def _ablation_colors(n: int) -> list[tuple[float, float, float, float]]:
    """Return n evenly-spaced colors from the ablation color map."""
    return [_ABLATION_CMAP(i / max(n - 1, 1)) for i in range(n)]


def plot_ablation_scatter(
    data_per_k: dict[str, dict[str, Any]],
    variant_order: list[str],
    variant_names: dict[str, str],
    k_value: int,
    title: str,
    output_path: Path,
) -> None:
    """Scatter: x = corpus size, y = AUC performance, one point per variant."""
    k_str = str(k_value)
    variants_data = data_per_k.get(k_str, {})

    variants = [v for v in variant_order if v in variants_data]
    if not variants:
        logger.info(
            f"[WARN] No variants for k={k_value} in {output_path.name}; skipping."
        )
        return

    x_vals: list[float | None] = []
    y_vals: list[float | None] = []
    x_errs: list[float] = []
    y_errs: list[float] = []
    labels: list[str] = []
    for v in variants:
        d = variants_data[v]
        cs = d.get("final_support_set_size", {})
        ap = d.get("auc_performance", {})
        x_vals.append(cs.get("mean"))
        x_errs.append(cs.get("se") or 0.0)
        y_vals.append(ap.get("mean"))
        y_errs.append(ap.get("se") or 0.0)
        labels.append(variant_names.get(v, v))

    colors = _ablation_colors(len(variants))

    fig, ax = plt.subplots(figsize=ACL_FIGSIZE)  # type: ignore

    for i in range(len(variants)):
        x, y = x_vals[i], y_vals[i]
        if x is None or y is None:
            continue
        ax.errorbar(  # type: ignore
            x,
            y,
            xerr=x_errs[i],
            yerr=y_errs[i],
            fmt="o",
            color=colors[i],
            markersize=5,
            capsize=3,
            capthick=1,
            linewidth=1,
            alpha=0.85,
        )
        ax.annotate(  # type: ignore
            labels[i],
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=ANNOT_FONTSIZE,
            alpha=0.9,
        )

    ax.set_xlabel("Corpus Size", fontsize=LABEL_FONTSIZE)  # type: ignore
    ax.set_ylabel("AUC Performance", fontsize=LABEL_FONTSIZE)  # type: ignore
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=4)  # type: ignore
    ax.set_ylim(0.60, 0.70)
    ax.tick_params(labelsize=TICK_FONTSIZE)  # type: ignore
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)  # type: ignore

    fig.tight_layout(pad=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=ACL_DPI, bbox_inches="tight")  # type: ignore
    plt.close(fig)
    logger.info(f"Written: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate scatter-plot figures for secondary analyses."
    )
    parser.add_argument(
        "--secondary-path",
        type=Path,
        required=True,
        help="Path to the combined secondary-experiments aggregated JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots_secondary"),
        help="Output directory for PNG files (default: plots_secondary/)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=40,
        help="k value to extract from results (default: 40)",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir

    ablations = [
        (
            "cold_start",
            COLD_START_ORDER,
            COLD_START_NAMES,
            f"Cold-start ablation (k={args.k})",
            "cold_start_scatter.png",
        ),
        (
            "guardrail",
            GUARDRAIL_ORDER,
            GUARDRAIL_NAMES,
            f"Guardrail ablation (k={args.k})",
            "guardrail_scatter.png",
        ),
        (
            "retrain_interval",
            RETRAIN_ORDER,
            RETRAIN_NAMES,
            f"Retrain interval ablation (k={args.k})",
            "retrain_interval_scatter.png",
        ),
    ]

    for exp_type, order, names, title, filename in ablations:
        try:
            data = load_secondary_experiment(args.secondary_path, exp_type)
        except KeyError:
            logger.info(
                f"[WARN] '{exp_type}' not found in {args.secondary_path}; skipping."
            )
            continue
        plot_ablation_scatter(
            data_per_k=data,
            variant_order=order,
            variant_names=names,
            k_value=args.k,
            title=title,
            output_path=output_dir / filename,
        )


if __name__ == "__main__":
    main()
