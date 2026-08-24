<#
.SYNOPSIS
    Generate the combined R2Bench figure (line plot + sparsity t-SNE + cover set donut) and its LaTeX table.

.EXAMPLE
    .\plot_r2bench.ps1 -Penalty "0.25"

.EXAMPLE
    .\plot_r2bench.ps1 -Penalty "0.25" -SkipSparsity

.EXAMPLE
    .\plot_r2bench.ps1 -Penalty "0.25" -SkipFigures

.EXAMPLE
    .\plot_r2bench.ps1 -Penalty "0.25" -StrategyPrefixes inferred_sparse,oracle_sparse
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$Penalty,

    [Parameter(Mandatory=$false)]
    [string]$OutputDir = "plots_benchmarks/r2bench",

    [Parameter(Mandatory=$false)]
    [string[]]$StrategyPrefixes = $null,

    [Parameter(Mandatory=$false)]
    [switch]$SkipLine,

    [Parameter(Mandatory=$false)]
    [switch]$SkipSparsity,

    [Parameter(Mandatory=$false)]
    [switch]$SkipFigures
)

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Generating R2Bench Figure and LaTeX Table"        -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Penalty:    $Penalty"   -ForegroundColor Yellow
Write-Host "Output dir: $OutputDir" -ForegroundColor Yellow
if ($StrategyPrefixes) {
    Write-Host "Strategy prefixes: $($StrategyPrefixes -join ', ')" -ForegroundColor Yellow
}
Write-Host ""

$cmd = "python -m plot_commands.plot_benchmarks_benchmark_r2bench --penalty $Penalty --output-dir `"$OutputDir`""

if ($StrategyPrefixes) {
    $cmd += " --strategy-prefixes $($StrategyPrefixes -join ' ')"
}
if ($SkipLine)     { $cmd += " --skip-line" }
if ($SkipSparsity) { $cmd += " --skip-sparsity" }
if ($SkipFigures)  { $cmd += " --skip-figures" }

Write-Host "Running: $cmd" -ForegroundColor Gray
Invoke-Expression $cmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: R2Bench plotting failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Done!" -ForegroundColor Cyan
Write-Host "Output: $OutputDir" -ForegroundColor Yellow
Write-Host "================================================" -ForegroundColor Cyan
