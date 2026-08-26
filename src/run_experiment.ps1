<#
.SYNOPSIS
    Run one experiment type over n_simulations seeds for a single benchmark.

.EXAMPLE
    .\run_experiment.ps1 -n_simulations 10 -embedding_model "snowflake-arctic-embed-m-v2.0" -benchmark "RouterBench" -type "main_stationary" -k_values 3,5,7

.EXAMPLE
    .\run_experiment.ps1 -n_simulations 10 -embedding_model "snowflake-arctic-embed-m-v2.0" -benchmark "Sprout" -type "main_domain_non_stationary" -cost_penalty 0.25 -budget 500
#>

param(
    [Parameter(Mandatory=$true)]
    [int]$n_simulations,

    [Parameter(Mandatory=$true)]
    [ValidateSet("bge-m3", "snowflake-arctic-embed-l-v2.0", "snowflake-arctic-embed-m-v2.0", "potion-multilingual-128M", "text-embedding-3-large")]
    [string]$embedding_model,

    [Parameter(Mandatory=$true)]
    [ValidateSet("EmbedLLM", "RouterBench", "Sprout", "FusionBench", "R2Bench")]
    [string]$benchmark,

    [Parameter(Mandatory=$true)]
    [ValidateSet("main_stationary", "main_domain_non_stationary", "model_switch_regret", "partial_annotation")]
    [string]$type,

    [Parameter(Mandatory=$false)]
    [int]$base_seed = 1,

    [Parameter(Mandatory=$false)]
    [int]$budget = 500,

    [Parameter(Mandatory=$false)]
    [double]$cost_penalty = 0.0,

    [Parameter(Mandatory=$false)]
    [int[]]$k_values = @(40),

    [Parameter(Mandatory=$false)]
    [ValidateSet("true", "false", "")]
    [string]$stationary = ""
)

$seeds = @()
for ($i = 0; $i -lt $n_simulations; $i++) {
    $seeds += $base_seed + $i
}

Write-Host "Running $n_simulations simulations with seeds: $($seeds -join ', ')" -ForegroundColor Yellow
Write-Host "Experiment Type: $type" -ForegroundColor Yellow
Write-Host "Embedding Model: $embedding_model" -ForegroundColor Yellow
Write-Host "Benchmark: $benchmark" -ForegroundColor Yellow
Write-Host "Budget: $budget" -ForegroundColor Yellow
Write-Host "Cost Penalty: $cost_penalty" -ForegroundColor Yellow
Write-Host "K Values: $($k_values -join ', ')" -ForegroundColor Yellow
if ($stationary -ne "") {
    Write-Host "Stationary: $stationary" -ForegroundColor Yellow
}
Write-Host ""

$success_count = 0
$failed_count = 0

foreach ($seed in $seeds) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Running simulation $($success_count + $failed_count + 1)/$n_simulations with seed: $seed" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    try {
        $cmd_args = @("run.py", "--type", $type, "--seed", $seed, "--embedding_model", $embedding_model, "--benchmark", $benchmark, "--budget", $budget, "--cost_penalty", $cost_penalty, "--k_values") + $k_values

        if ($stationary -ne "") {
            $cmd_args += @("--stationary", $stationary)
        }
        python @cmd_args
        if ($LASTEXITCODE -eq 0) {
            $success_count++
            Write-Host "Simulation with seed $seed completed successfully" -ForegroundColor Green
        } else {
            $failed_count++
            Write-Host "Simulation with seed $seed failed with exit code $LASTEXITCODE" -ForegroundColor Red
        }
    }
    catch {
        $failed_count++
        Write-Host "Simulation with seed $seed failed with error: $_" -ForegroundColor Red
    }

    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Yellow
Write-Host "Experiment Summary" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "Total simulations: $n_simulations" -ForegroundColor Yellow
Write-Host "Successful: $success_count" -ForegroundColor Green
Write-Host "Failed: $failed_count" -ForegroundColor Red
Write-Host "Seeds used: $($seeds -join ', ')" -ForegroundColor Yellow
