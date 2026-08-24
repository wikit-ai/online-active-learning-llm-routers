<#
.SYNOPSIS
    Generates the plots for the domain-shift (non-stationary) experiments (AUC vs Cost, DTO, domain periods, and optionally t-SNE).

.EXAMPLE
    .\plot_domain_non_stationary.ps1 -Penalty "0.25" -Benchmark "Sprout"

.EXAMPLE
    .\plot_domain_non_stationary.ps1 -Penalty "0.25" -Benchmark "RouterBench" -Seeds 1,2,3 -Tsne -StrategyPrefixes "rand_","act_unc","oracle"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$Penalty,

    [Parameter(Mandatory=$true)]
    [ValidateSet("Sprout", "RouterBench", "EmbedLLM", "FusionBench")]
    [string]$Benchmark,

    [Parameter(Mandatory=$false)]
    [int[]]$Seeds = @(1, 2, 3),

    [Parameter(Mandatory=$false)]
    [string[]]$StrategyPrefixes = $null,

    [Parameter(Mandatory=$false)]
    [switch]$Tsne
)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Generating Domain-Shift Experiment Plots" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Benchmark: $Benchmark" -ForegroundColor Yellow
Write-Host "Penalty: $Penalty" -ForegroundColor Yellow
Write-Host "Seeds: $($Seeds -join ', ')" -ForegroundColor Yellow
Write-Host ""

# Step 1: AUC vs Cost + domain-period comparison
Write-Host "Step 1: Generating AUC vs Cost and domain-period plots..." -ForegroundColor Green
$aucCommand = "python -m plot_commands.plot_all_domain_non_stationary --penalty $Penalty --benchmark $($Benchmark.ToLower())"

if ($StrategyPrefixes) {
    $aucCommand += " --strategy-prefixes $($StrategyPrefixes -join ' ')"
}

Write-Host "Running: $aucCommand" -ForegroundColor Gray
Invoke-Expression $aucCommand

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: AUC vs Cost plotting failed!" -ForegroundColor Red
    exit 1
}

Write-Host "AUC vs Cost plots completed successfully!" -ForegroundColor Green
Write-Host ""

# Step 2: t-SNE (optional)
if ($Tsne) {
    Write-Host "Step 2: Generating t-SNE visualizations..." -ForegroundColor Green
    $tsneCommand = "python -m plot_commands.plot_tsne_non_stationary --penalty $Penalty --benchmark $Benchmark --seeds $($Seeds -join ' ')"

    if ($StrategyPrefixes) {
        $tsneCommand += " --strategy-prefixes $($StrategyPrefixes -join ' ')"
    }

    Write-Host "Running: $tsneCommand" -ForegroundColor Gray
    Invoke-Expression $tsneCommand

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

# Summary
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "All plots generated successfully!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Output directories:" -ForegroundColor Yellow
Write-Host "  - AUC plots:             plots_benchmarks/domain_shift/perf_vs_cost/$($Benchmark.ToLower())" -ForegroundColor Gray
Write-Host "  - DTO plots:             plots_benchmarks/domain_shift/dto_vs_cost/$($Benchmark.ToLower())" -ForegroundColor Gray
Write-Host "  - Final perf plots:      plots_benchmarks/domain_shift/final_performance/$($Benchmark.ToLower())" -ForegroundColor Gray
Write-Host "  - Domain period plots:   plots_benchmarks/domain_shift/domain_period_comparison/$($Benchmark.ToLower())" -ForegroundColor Gray
Write-Host "  - Performance vs step:   plots_benchmarks/domain_shift/performance_vs_step/$Benchmark" -ForegroundColor Gray
