<#
.SYNOPSIS
    Run the secondary (ablation) experiments: cold start, guardrail, retrain interval.

.EXAMPLE
    .\run_secondary_experiments.ps1 -Experiments all -Benchmark RouterBench

.EXAMPLE
    .\run_secondary_experiments.ps1 -Experiments cold_start,guardrail -Benchmark Sprout -Seeds 1,2,3
#>

param(
    [Parameter(Mandatory=$true)]
    [string[]]$Experiments,

    [Parameter(Mandatory=$true)]
    [ValidateSet("Sprout", "RouterBench", "EmbedLLM", "FusionBench")]
    [string]$Benchmark,

    [Parameter(Mandatory=$false)]
    [int[]]$Seeds = @(1, 2, 3, 4, 5, 6, 7, 8 ,9, 10),

    [Parameter(Mandatory=$false)]
    [string]$EmbeddingModel = "snowflake-arctic-embed-m-v2.0",

    [Parameter(Mandatory=$false)]
    [int]$Budget = 500,

    [Parameter(Mandatory=$false)]
    [int[]]$KValues = @(40),

    [Parameter(Mandatory=$false)]
    [float]$CostPenalty = 0.25,

    [Parameter(Mandatory=$false)]
    [string]$OutputDir = "results/secondary_experiments",

    [Parameter(Mandatory=$false)]
    [int[]]$ColdStartValues = $null,

    [Parameter(Mandatory=$false)]
    [int[]]$RetrainIntervalValues = $null,

    [Parameter(Mandatory=$false)]
    [float[]]$GuardrailThresholds = $null
)

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Running Secondary (Ablation) Experiments" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Experiments:     $($Experiments -join ', ')" -ForegroundColor Yellow
Write-Host "Benchmark:       $Benchmark" -ForegroundColor Yellow
Write-Host "Embedding model: $EmbeddingModel" -ForegroundColor Yellow
Write-Host "Seeds:           $($Seeds -join ', ')" -ForegroundColor Yellow
Write-Host "Budget:          $Budget" -ForegroundColor Yellow
Write-Host "K values:        $($KValues -join ', ')" -ForegroundColor Yellow
Write-Host "Output dir:      $OutputDir" -ForegroundColor Yellow
Write-Host ""

$cmd = "python -m experiment.secondary_experiments"
$cmd += " --experiments $($Experiments -join ' ')"
$cmd += " --seeds $($Seeds -join ' ')"
$cmd += " --benchmark $Benchmark"
$cmd += " --embedding_model $EmbeddingModel"
$cmd += " --budget $Budget"
$cmd += " --k_values $($KValues -join ' ')"
$cmd += " --cost_penalty $CostPenalty"
$cmd += " --output_dir $OutputDir"

if ($ColdStartValues) {
    $cmd += " --cold_start_values $($ColdStartValues -join ' ')"
}
if ($RetrainIntervalValues) {
    $cmd += " --retrain_interval_values $($RetrainIntervalValues -join ' ')"
}
if ($GuardrailThresholds) {
    $cmd += " --guardrail_thresholds $($GuardrailThresholds -join ' ')"
}

Write-Host "Running: $cmd" -ForegroundColor Gray
Invoke-Expression $cmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: secondary experiments failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Secondary experiments completed successfully!" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Output directory: $OutputDir" -ForegroundColor Yellow
