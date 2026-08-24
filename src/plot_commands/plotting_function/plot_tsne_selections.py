import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from plot_commands.plotting_function.utils import get_colors_per_strategy

plt.style.use("ggplot")


def plot_tsne_selections_per_seed(
    embeddings: np.ndarray,
    selections: dict[str, list[int]],
    seed: int,
    output_path: str,
    figsize: tuple[int, int] = (18, 15),
    perplexity: int = 30,
    random_state: int = 42,
):
    """Plot a t-SNE grid highlighting the points selected by each strategy.
    One subplot per strategy (sorted by number of selections).

    Args:
        embeddings: Corpus embeddings, one row per item
        selections: {strategy: selected item indices into `embeddings`}
        seed: Seed shown in the figure title
        output_path: File path to save the figure
        figsize: Figure size (width, height)
        perplexity: t-SNE perplexity
        random_state: t-SNE random seed
    """
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    coords = tsne.fit_transform(embeddings)

    pad = 0.05
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_range, y_range = x_max - x_min, y_max - y_min
    xlim = (x_min - pad * x_range, x_max + pad * x_range)
    ylim = (y_min - pad * y_range, y_max + pad * y_range)

    strategies = sorted(
        selections.keys(), key=lambda s: len(selections[s]), reverse=True
    )
    n = len(strategies)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)  # type: ignore
    axes = np.array(axes).flatten()

    all_strategies = list(selections.keys())
    strategy_colors = get_colors_per_strategy(all_strategies)
    color_map = dict(zip(all_strategies, strategy_colors))

    for i, strategy in enumerate(strategies):
        ax = axes[i]
        selected = set(selections[strategy])
        n_selected = len(selections[strategy])
        mask = np.array([j in selected for j in range(len(coords))])

        ax.scatter(
            coords[~mask, 0],
            coords[~mask, 1],
            c="#949494",
            s=8,
            alpha=0.5,
            linewidths=0,
            rasterized=True,
        )
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color_map[strategy],
            s=12,
            alpha=0.9,
            linewidths=0,
            zorder=3,
        )

        ax.set_title(f"{strategy} (n={n_selected})", fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    for j in range(i + 1, len(axes)):  # type: ignore
        axes[j].set_visible(False)

    plt.suptitle(  # type: ignore
        f"t-SNE — Selected Points per Strategy (Seed {seed})",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(h_pad=1.5, w_pad=0.5)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")  # type: ignore
    plt.close(fig)
