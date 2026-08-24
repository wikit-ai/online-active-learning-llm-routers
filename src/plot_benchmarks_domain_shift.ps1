<#
.SYNOPSIS
    Generate the 2x2 all-benchmarks domain-shift figure plus the domain-shift LaTeX table.

.EXAMPLE
    .\plot_benchmarks_domain_shift.ps1 -Penalty "0.25"

.EXAMPLE
    .\plot_benchmarks_domain_shift.ps1 -Penalty "0.25" -StrategyPrefixes inferred_sparse,oracle_sparse
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$Penalty,

    [Parameter(Mandatory=$false)]
    [string]$OutputDir = "plots_benchmarks/domain_shift",

    [Parameter(Mandatory=$false)]
    [string[]]$StrategyPrefixes = $null
)

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Generating All Benchmarks Domain-Shift Plots (2x2)" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Penalty:    $Penalty" -ForegroundColor Yellow
Write-Host "Output dir: $OutputDir" -ForegroundColor Yellow
if ($StrategyPrefixes) {
    Write-Host "Strategy prefixes: $($StrategyPrefixes -join ', ')" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "Step 1: Generating 2x2 benchmark plot..." -ForegroundColor Green
$cmd = "python -m plot_commands.plot_benchmarks_benchmark --penalty $Penalty --experiment-type domain_shift --output-dir `"$OutputDir`""

if ($StrategyPrefixes) {
    $cmd += " --strategy-prefixes $($StrategyPrefixes -join ' ')"
}

Write-Host "Running: $cmd" -ForegroundColor Gray
Invoke-Expression $cmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: benchmark plot failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Benchmark plot completed successfully!" -ForegroundColor Green
Write-Host ""

Write-Host "Step 2: Generating domain-shift LaTeX table..." -ForegroundColor Green
$latexCmd = "python -m plot_commands.plotting_function.plot_latex_table --output-dir `"$OutputDir`""
Write-Host "Running: $latexCmd" -ForegroundColor Gray
Invoke-Expression $latexCmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: domain-shift LaTeX table failed (cache may be empty - run plot_tables.ps1 -ExperimentType domain_shift first)." -ForegroundColor Yellow
} else {
    Write-Host "Domain-shift LaTeX table completed successfully!" -ForegroundColor Green
}
Write-Host ""

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Done!" -ForegroundColor Cyan
Write-Host "Output:" -ForegroundColor Yellow
Write-Host "  - 2x2 plot:           $OutputDir" -ForegroundColor Gray
Write-Host "  - Domain-shift table: $OutputDir/benchmark_table.tex" -ForegroundColor Gray
Write-Host "==================================================" -ForegroundColor Cyan
