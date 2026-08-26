<#
.SYNOPSIS
    Run the full experiment batch (stationary, domain-shift, model-switch regret and
    partial annotation) over all benchmarks.

.EXAMPLE
    .\run_batch_experiments.ps1
#>

$embeddingModel = "snowflake-arctic-embed-m-v2.0"

$experiments = @(
    # # Stationary experiments
    @{Simulations=10; Benchmark="FusionBench"; Seed=1; Type="main_stationary"; CostPenalty=0.25; Budget=500; K_values=@(40)}
    @{Simulations=10; Benchmark="Sprout"; Seed=1; Type="main_stationary"; CostPenalty=0.25; Budget=500; K_values=@(40)}
    @{Simulations=10; Benchmark="RouterBench"; Seed=1; Type="main_stationary"; CostPenalty=0.25; Budget=500; K_values=@(40)}
    @{Simulations=10; Benchmark="EmbedLLM"; Seed=1; Type="main_stationary"; CostPenalty=0.25; Budget=500; K_values=@(40)}

    # Held-out stationary benchmark
    @{Simulations=10; Benchmark="R2Bench"; Seed=1; Type="main_stationary"; CostPenalty=0.25; Budget=500; K_values=@(40)}

    # # Non-stationary experiments (with domain shifts every 500)
    @{Simulations=10; Benchmark="RouterBench"; Seed=1; Type="main_domain_non_stationary"; CostPenalty=0.25; Budget=500; K_values=@(40)}
    @{Simulations=10; Benchmark="EmbedLLM"; Seed=1; Type="main_domain_non_stationary"; CostPenalty=0.25; Budget=500; K_values=@(40)}
    @{Simulations=10; Benchmark="FusionBench"; Seed=1; Type="main_domain_non_stationary"; CostPenalty=0.25; Budget=500; K_values=@(40)}
    @{Simulations=10; Benchmark="Sprout"; Seed=1; Type="main_domain_non_stationary"; CostPenalty=0.25; Budget=500; K_values=@(40)}

    # Model-switch regret experiments (continual vs hindsight)
    @{Simulations=10; Benchmark="FusionBench"; Seed=1; Type="model_switch_regret"; CostPenalty=0.25; Budget=500; K_values=@(40)}
    @{Simulations=10; Benchmark="RouterBench"; Seed=1; Type="model_switch_regret"; CostPenalty=0.25; Budget=500; K_values=@(40)}
    @{Simulations=10; Benchmark="Sprout"; Seed=1; Type="model_switch_regret"; CostPenalty=0.25; Budget=500; K_values=@(40)}
    @{Simulations=10; Benchmark="EmbedLLM"; Seed=1; Type="model_switch_regret"; CostPenalty=0.25; Budget=500; K_values=@(40)}

    # Partial annotation experiments (full vs Wilcoxon)
    @{Simulations=10; Benchmark="FusionBench"; Seed=1; Type="partial_annotation"; CostPenalty=0.25; Budget=500; K_values=@(40)}
    @{Simulations=10; Benchmark="RouterBench"; Seed=1; Type="partial_annotation"; CostPenalty=0.25; Budget=500; K_values=@(40)}
    @{Simulations=10; Benchmark="Sprout"; Seed=1; Type="partial_annotation"; CostPenalty=0.25; Budget=500; K_values=@(40)}
    @{Simulations=10; Benchmark="EmbedLLM"; Seed=1; Type="partial_annotation"; CostPenalty=0.25; Budget=500; K_values=@(40)}

)


foreach ($exp in $experiments) {
    $args = @{
        n_simulations  = $exp.Simulations
        embedding_model = $embeddingModel
        benchmark      = $exp.Benchmark
        type           = $exp.Type
        base_seed      = $exp.Seed
        cost_penalty   = $exp.CostPenalty
        budget         = $exp.Budget
        k_values       = $exp.K_values
    }
    .\run_experiment.ps1 @args
}
