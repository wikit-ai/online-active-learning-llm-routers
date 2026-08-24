<#
.SYNOPSIS
    Generate the model-switch regret line plot (continual vs hindsight).

.EXAMPLE
    .\plot_model_switch_regret.ps1

.EXAMPLE
    .\plot_model_switch_regret.ps1 -ResultsDir "results" -OutputDir "plots_benchmarks/model_switch_regret"
#>

param(
    [Parameter(Mandatory=$false)]
    [string]$ResultsDir = "results",

    [Parameter(Mandatory=$false)]
    [string]$EmbeddingModel = "snowflake-arctic-embed-m-v2.0",

    [Parameter(Mandatory=$false)]
    [string]$OutputDir = "plots_benchmarks/model_switch_regret"
)

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Generating Model-Switch Regret Line Plot" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Results dir   : $ResultsDir" -ForegroundColor Yellow
Write-Host "Embedding model: $EmbeddingModel" -ForegroundColor Yellow
Write-Host "Output dir    : $OutputDir" -ForegroundColor Yellow
Write-Host ""

$cmd = "python -m plot_commands.plot_model_switch_regret" `
     + " --results-dir `"$ResultsDir`"" `
     + " --embedding-model `"$EmbeddingModel`"" `
     + " --output-dir `"$OutputDir`""

Write-Host "Running: $cmd" -ForegroundColor Gray
Invoke-Expression $cmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: plotting failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Plot saved to: $OutputDir" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
