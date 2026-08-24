<#
.SYNOPSIS
    Grid-search the MLP hyperparameters of the inferred-sparse strategy.

.EXAMPLE
    .\run_hp_search.ps1

.EXAMPLE
    .\run_hp_search.ps1 -Benchmarks RouterBench,EmbedLLM -Seeds 0,1,2

.EXAMPLE
    .\run_hp_search.ps1 -LearningRates 1e-4,1e-3 -WeightDecays 1e-3,1e-2 -HiddenDims 16,32,64,128
#>

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("EmbedLLM", "RouterBench", "Sprout", "FusionBench")]
    [string[]]$Benchmarks = @("EmbedLLM", "RouterBench", "Sprout", "FusionBench"),

    [Parameter(Mandatory=$false)]
    [int[]]$Seeds = @(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),

    [Parameter(Mandatory=$false)]
    [string]$EmbeddingModel = "snowflake-arctic-embed-m-v2.0",

    [Parameter(Mandatory=$false)]
    [int]$Budget = 500,

    [Parameter(Mandatory=$false)]
    [int[]]$KValues = @(40),

    [Parameter(Mandatory=$false)]
    [double]$CostPenalty = 0.25,

    [Parameter(Mandatory=$false)]
    [double]$ValFraction = 0.2,

    [Parameter(Mandatory=$false)]
    [int]$ColdStartN = 115,

    [Parameter(Mandatory=$false)]
    [string]$OutputDir = "results/warmup",

    [Parameter(Mandatory=$false)]
    [double[]]$LearningRates = $null,

    [Parameter(Mandatory=$false)]
    [double[]]$WeightDecays = $null,

    [Parameter(Mandatory=$false)]
    [int[]]$HiddenDims = $null
)

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Warmup HP Search" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Benchmarks:      $($Benchmarks -join ', ')" -ForegroundColor Yellow
Write-Host "Embedding model: $EmbeddingModel" -ForegroundColor Yellow
Write-Host "Seeds:           $($Seeds -join ', ')" -ForegroundColor Yellow
Write-Host "Budget:          $Budget" -ForegroundColor Yellow
Write-Host "K values:        $($KValues -join ', ')" -ForegroundColor Yellow
Write-Host "Val fraction:    $ValFraction" -ForegroundColor Yellow
Write-Host "Cold start n:    $ColdStartN" -ForegroundColor Yellow
Write-Host "Output dir:      $OutputDir" -ForegroundColor Yellow
Write-Host ""

$cmd = "python -m experiment.warmup.hp_search"
$cmd += " --benchmarks $($Benchmarks -join ' ')"
$cmd += " --seeds $($Seeds -join ' ')"
$cmd += " --embedding_model $EmbeddingModel"
$cmd += " --budget $Budget"
$cmd += " --k_values $($KValues -join ' ')"
$cmd += " --cost_penalty $CostPenalty"
$cmd += " --val_fraction $ValFraction"
$cmd += " --cold_start_n $ColdStartN"
$cmd += " --output_dir $OutputDir"

if ($LearningRates) {
    $cmd += " --learning_rates $($LearningRates -join ' ')"
}
if ($WeightDecays) {
    $cmd += " --weight_decays $($WeightDecays -join ' ')"
}
if ($HiddenDims) {
    $cmd += " --hidden_dims $($HiddenDims -join ' ')"
}

Write-Host "Running: $cmd" -ForegroundColor Gray
Write-Host ""
Invoke-Expression $cmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "HP search failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "HP search completed!" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Results in: $OutputDir" -ForegroundColor Yellow
Write-Host "Check aggregated_*.json for best_per_benchmark summary." -ForegroundColor Yellow
