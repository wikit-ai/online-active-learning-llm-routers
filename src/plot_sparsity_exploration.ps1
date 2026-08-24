<#
.SYNOPSIS
    Generate the sparsity-exploration figures across all benchmarks.

.EXAMPLE
    .\plot_sparsity_exploration.ps1
#>

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Sparsity Exploration Plots (all benchmarks)"    -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

$cmd = "python -m plot_commands.plot_benchmarks_sparsity_exploration"
Write-Host "Running: $cmd" -ForegroundColor Gray
Invoke-Expression $cmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: sparsity exploration plot failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Done! Figures saved to: plots_benchmarks/paper" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
