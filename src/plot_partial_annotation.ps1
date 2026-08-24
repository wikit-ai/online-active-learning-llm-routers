<#
.SYNOPSIS
    Generate the partial-annotation comparison plots (full vs Wilcoxon).

.EXAMPLE
    .\plot_partial_annotation.ps1 -Penalty "0.25"

.EXAMPLE
    .\plot_partial_annotation.ps1 -Penalty "0.25" -OutputDir "plots_benchmarks/partial_annotation"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$Penalty,

    [Parameter(Mandatory=$false)]
    [string]$OutputDir = "plots_benchmarks/partial_annotation"
)

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Generating Partial Annotation Comparison Plots"   -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Penalty:    $Penalty"   -ForegroundColor Yellow
Write-Host "Output dir: $OutputDir" -ForegroundColor Yellow
Write-Host ""

$cmd = "python -m plot_commands.plot_benchmarks_partial_annotation --penalty $Penalty --output-dir `"$OutputDir`""
Write-Host "Running: $cmd" -ForegroundColor Gray
Invoke-Expression $cmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: partial annotation plot failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Done!" -ForegroundColor Cyan
Write-Host "Output: $OutputDir" -ForegroundColor Yellow
Write-Host "================================================" -ForegroundColor Cyan
