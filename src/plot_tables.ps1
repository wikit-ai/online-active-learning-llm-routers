<#
.SYNOPSIS
    Build the benchmark LaTeX table from scratch: refresh the per-benchmark stats cache, then render the table.

.DESCRIPTION
    plot_latex_table reads plots_benchmarks/<ExperimentType>/stats_cache/<benchmark>.json, which is
    written as a side effect of plot_experiment.ps1. This script therefore runs plot_experiment.ps1
    for every benchmark first (Step 1), then renders benchmark_table.tex from the whole cache (Step 2).

    Use -SkipCache when the cache is already up to date and only the .tex needs regenerating.

.EXAMPLE
    .\plot_tables.ps1 -Penalty "0.25"

.EXAMPLE
    .\plot_tables.ps1 -Penalty "0.25" -ExperimentType domain_shift

.EXAMPLE
    .\plot_tables.ps1 -Penalty "0.25" -Benchmarks RouterBench,EmbedLLM -KnnK 40

.EXAMPLE
    .\plot_tables.ps1 -Penalty "0.25" -SkipCache -CorpusSize
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$Penalty,

    [Parameter(Mandatory=$false)]
    [ValidateSet("stationary", "domain_shift")]
    [string]$ExperimentType = "stationary",

    [Parameter(Mandatory=$false)]
    [ValidateSet("Sprout", "RouterBench", "EmbedLLM", "FusionBench")]
    [string[]]$Benchmarks = @("EmbedLLM", "RouterBench", "Sprout", "FusionBench"),

    [Parameter(Mandatory=$false)]
    [string]$OutputDir = $null,

    [Parameter(Mandatory=$false)]
    [string[]]$StrategyPrefixes = $null,

    [Parameter(Mandatory=$false)]
    [int]$KnnK = 0,

    [Parameter(Mandatory=$false)]
    [switch]$CorpusSize,

    [Parameter(Mandatory=$false)]
    [switch]$SkipCache
)

if (-not $OutputDir) {
    $OutputDir = "plots_benchmarks/$ExperimentType"
}

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Generating Benchmark LaTeX Table"                 -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Experiment: $ExperimentType"           -ForegroundColor Yellow
Write-Host "Penalty:    $Penalty"                  -ForegroundColor Yellow
Write-Host "Benchmarks: $($Benchmarks -join ', ')" -ForegroundColor Yellow
Write-Host "Output dir: $OutputDir"                -ForegroundColor Yellow
Write-Host ""

if ($SkipCache) {
    Write-Host "Step 1: Skipping cache refresh (use without -SkipCache to rebuild)." -ForegroundColor Gray
    Write-Host ""
} else {
    Write-Host "Step 1: Refreshing stats cache via plot_experiment.ps1..." -ForegroundColor Green

    foreach ($benchmark in $Benchmarks) {
        Write-Host "  -> $benchmark" -ForegroundColor Yellow

        $expArgs = @{
            Penalty        = $Penalty
            Benchmark      = $benchmark
            ExperimentType = $ExperimentType
        }
        if ($StrategyPrefixes) {
            $expArgs["StrategyPrefixes"] = $StrategyPrefixes
        }

        .\plot_experiment.ps1 @expArgs

        if ($LASTEXITCODE -ne 0) {
            Write-Host "Warning: $benchmark failed; it will be missing from the table." -ForegroundColor Yellow
        }
    }

    Write-Host "Cache refresh completed!" -ForegroundColor Green
    Write-Host ""
}

Write-Host "Step 2: Rendering benchmark_table.tex from cache..." -ForegroundColor Green
$latexCmd = "python -m plot_commands.plotting_function.plot_latex_table --output-dir `"$OutputDir`""

if ($KnnK -ne 0) {
    $latexCmd += " --knn-k $KnnK"
}
if ($CorpusSize) {
    $latexCmd += " --corpus-size"
}

Write-Host "Running: $latexCmd" -ForegroundColor Gray
Invoke-Expression $latexCmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: LaTeX table generation failed (no cached benchmarks in $OutputDir/stats_cache/)." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Done!" -ForegroundColor Cyan
Write-Host "Output: $OutputDir/benchmark_table.tex" -ForegroundColor Yellow
Write-Host "================================================" -ForegroundColor Cyan
