<#
.SYNOPSIS
    Generate every figure and the LaTeX table for one benchmark, optionally with t-SNE plots.

.EXAMPLE
    .\plot_experiment.ps1 -Penalty "0.25" -Benchmark "RouterBench"

.EXAMPLE
    .\plot_experiment.ps1 -Penalty "0.25" -Benchmark "Sprout" -ExperimentType "domain_shift"

.EXAMPLE
    .\plot_experiment.ps1 -Penalty "0.25" -Benchmark "EmbedLLM" -ExperimentType "domain_shift" -Tsne -Seeds 1,2,3
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$Penalty,

    [Parameter(Mandatory=$true)]
    [ValidateSet("Sprout", "RouterBench", "EmbedLLM", "FusionBench")]
    [string]$Benchmark,

    [Parameter(Mandatory=$false)]
    [ValidateSet("stationary", "domain_shift")]
    [string]$ExperimentType = "stationary",

    [Parameter(Mandatory=$false)]
    [int[]]$Seeds = @(1, 2, 3),

    [Parameter(Mandatory=$false)]
    [string[]]$StrategyPrefixes = $null,

    [Parameter(Mandatory=$false)]
    [switch]$Tsne
)

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Generating Experiment Plots"            -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Experiment: $ExperimentType"  -ForegroundColor Yellow
Write-Host "Benchmark:  $Benchmark"       -ForegroundColor Yellow
Write-Host "Penalty:    $Penalty"         -ForegroundColor Yellow
Write-Host "Seeds:      $($Seeds -join ', ')" -ForegroundColor Yellow
Write-Host ""

Write-Host "Step 1: Generating plots..." -ForegroundColor Green
$plotCmd = "python -m plot_commands.plot_all --penalty $Penalty --benchmark $($Benchmark.ToLower()) --experiment-type $ExperimentType"

if ($StrategyPrefixes) {
    $plotCmd += " --strategy-prefixes $($StrategyPrefixes -join ' ')"
}

Write-Host "Running: $plotCmd" -ForegroundColor Gray
Invoke-Expression $plotCmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: plotting failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Plots completed successfully!" -ForegroundColor Green
Write-Host ""

if ($Tsne) {
    Write-Host "Step 2: Generating t-SNE visualizations..." -ForegroundColor Green
    $tsneCmd = "python -m plot_commands.plot_tsne --penalty $Penalty --benchmark $Benchmark --experiment-type $ExperimentType --seeds $($Seeds -join ' ')"

    if ($StrategyPrefixes) {
        $tsneCmd += " --strategy-prefixes $($StrategyPrefixes -join ' ')"
    }

    Write-Host "Running: $tsneCmd" -ForegroundColor Gray
    Invoke-Expression $tsneCmd

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: t-SNE plotting failed!" -ForegroundColor Red
        exit 1
    }

    Write-Host "t-SNE visualizations completed successfully!" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "Step 2: Skipping t-SNE visualizations (use -Tsne to enable)." -ForegroundColor Gray
    Write-Host ""
}

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "All plots generated successfully!"      -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Output: plots_benchmarks/$ExperimentType" -ForegroundColor Yellow
