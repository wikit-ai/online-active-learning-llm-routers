# Online Active Learning for Retrieval-Based LLM Routers

Code for the paper [EMNLP 2026 Findings] *Targeting the Exceptions: Online Active Learning for Retrieval-Based LLM Routers* - Clovis Varangot-Reille, Antoine Gourru, Christophe Bouvard, Baptiste Jeudy

This repository is about a stream-based active learning method that builds the support corpus of a retrieval-based LLM router under a finite annotation budget. It annotates queries whose predicted **performance sparsity**  (the KL divergence of the per-query performance vector from uniform over the candidate pool) exceeds an adaptive threshold, targeting the *exception regions* informative regions.

Default parameters: embedding model is `snowflake-arctic-embed-m-v2.0` (pre-computed, no GPU required), cost penalty is `0.25` and K is `40`.

## Installation

```bash
# 1. Install PyTorch (only needed for re-embedding)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 2. Install chroma_cache (embedding cache backend)
pip install "chroma_cache[st] @ git+https://github.com/Lyon-NLP/chroma_cache.git"

pip install -r requirements.txt
```
---

## Strategies

| Strategy | Key in `run.py` | Variants (param -> keys) | Implementation |
|---|---|---|---|
| **SparseNN** (ours) | `inferred_sparse_nn_opt` | percentile: 75 / 25 (`_p25`) / 50 (`_p50`) / 90 (`_p90`) | `sampling_strategies/sparsity_learning/inferred_sparse_performance.py` |
| OracleSp | `oracle_sparse` | percentile: 0.75 / 0.90 (`_q90`) / 0.50 (`_q50`) / 0.25 (`_q25`) | `sampling_strategies/oracle_sparse_performance.py` |
| OracleCov | `oracle_ds_200` | n_corpus: 200 / 100 (`_100`) / 500 (`_500`) | `sampling_strategies/oracle_coverage.py` |
| Passive | `passive` | - | `sampling_strategies/passive_strategy.py` |
| R(alpha) | `rand_0_25` | alpha: 0.05 / 0.10 / 0.15 / 0.25 | `sampling_strategies/random_strategy.py` |
| a_rank | `unc_mc_t3` | percentile: 0.95 / 0.85 (`_q85`) / 0.75 (`_q75`) | `active_learning/heuristics/uncertainty.py` |
| a_var | `unc_t3` | percentile: 0.95 / 0.85 (`_q85`) / 0.75 (`_q75`) | `active_learning/heuristics/uncertainty.py` |
| a_min | `act_repr_min` | percentile: 0.05 / 0.15 (`_q15`) / 0.25 (`_q25`) | `active_learning/heuristics/diversity.py`|
| a_vMF | `repr_vmf_kap1` | percentile: 0.05 / 0.15 (`_q15`) / 0.25 (`_q25`) | `active_learning/heuristics/diversity.py` |

Strategies are registered by adding entries to the `strategies` dict in `run.py`.

---
## Running experiments

### `run_batch_experiments.ps1` 

Runs every experiment in the list, 10 seeds each, with `snowflake-arctic-embed-m-v2.0`:

- **stationary** - FusionBench, Sprout, RouterBench, EmbedLLM, plus the held-out R2Bench
- **domain shift** - RouterBench, EmbedLLM, FusionBench, Sprout
- **model-switch regret** - FusionBench, RouterBench, Sprout, EmbedLLM
- **partial annotation** - FusionBench, RouterBench, Sprout, EmbedLLM

Edit the `$experiments` list in the file to change what runs.

```powershell
.\run_batch_experiments.ps1
```

## Generating plots

### Per-benchmark plots (`plot_experiment.ps1`)

```powershell
# Stationary (default)
.\plot_experiment.ps1 -Penalty 0.25 -Benchmark RouterBench

# Domain shift
.\plot_experiment.ps1 -Penalty 0.25 -Benchmark Sprout -ExperimentType domain_shift
```

### All-benchmark summary figures

```powershell
# Stationary
.\plot_benchmarks_figure_tables.ps1 -Penalty 0.25

# Domain shift
.\plot_benchmarks_domain_shift.ps1 -Penalty 0.25
```

Both rebuild `benchmark_table.tex` from the stats cache under `plots_benchmarks/<experiment_type>/stats_cache/`. If that cache is empty the table step warns and is skipped - populate it with `plot_tables.ps1` below.

### LaTeX tables from scratch (`plot_tables.ps1`)

```powershell
.\plot_tables.ps1 -Penalty 0.25
.\plot_tables.ps1 -Penalty 0.25 -ExperimentType domain_shift
```

### Sparsity exploration - all benchmarks (`plot_sparsity_exploration.ps1`)

```powershell
.\plot_sparsity_exploration.ps1
```

### R2Bench combined figure (`plot_r2bench.ps1`)

```powershell
# Full figure (line plot + sparsity analysis) + LaTeX table
.\plot_r2bench.ps1 -Penalty 0.25
```

### Partial annotation comparison (`plot_partial_annotation.ps1`)

```powershell
.\plot_partial_annotation.ps1 -Penalty 0.25
```

### Model-switch regret (`plot_model_switch_regret.ps1`)

```powershell
.\plot_model_switch_regret.ps1
```
---

## Hyperparameter search

`run_hp_search.ps1` runs a warmup HP search over learning rate, weight decay, and hidden dimension on a held-out validation split (never touching the test set).

```powershell
.\run_hp_search.ps1
```

## Ablation / secondary experiments

`run_secondary_experiments.ps1` sweeps over cold-start size, retrain interval, and guardrail threshold.

```powershell
.\run_secondary_experiments.ps1 -Experiments all -Benchmark RouterBench

python -m plot_commands.generate_figures_secondary `
    --secondary-path results/secondary_experiments/aggregated_RouterBench_{timestamp}.json `
    --output-dir plots_secondary `
    --k 40
```
---

## Benchmarks

Benchmarks are loaded automatically from HuggingFace at runtime, from [`Wikit/RoutingCompendium-perf`](https://huggingface.co/datasets/Wikit/RoutingCompendium-perf) (per-query performance of every candidate) and [`Wikit/RoutingCompendium-cost`](https://huggingface.co/datasets/Wikit/RoutingCompendium-cost) (candidate prices).
