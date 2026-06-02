# SAHI + P2 Ablation Study (T20)
#
# Runs the full 2×2 ablation comparing P2 head and SAHI inference.
#
# Combinations:
#   1. Standard model + Standard inference  (baseline, reuse existing)
#   2. Standard model + SAHI inference       (SAHI only)
#   3. P2 model + Standard inference          (P2 only)
#   4. P2 model + SAHI inference              (P2 + SAHI)
#
# Usage:
#   .\scripts\run_sahi_p2_ablation.ps1                      # Full ablation
#   .\scripts\run_sahi_p2_ablation.ps1 -SkipTrain           # Skip P2 training
#   .\scripts\run_sahi_p2_ablation.ps1 -EvalOnly            # Evaluate only
#
# Prerequisites:
#   - Baseline model already trained (e.g., Run3 or T5)
#   - P2 model trained or will be trained by this script
#
# Output:
#   - results/sahi_p2_ablation.json  — all metrics
#   - Console summary table

param(
    [string]$Dataset = "nslsr-lwir",
    [int]$Epochs = 200,
    [int]$Batch = 16,
    [switch]$SkipTrain,
    [switch]$EvalOnly,
    [switch]$Help
)

if ($Help) {
    Get-Help $MyInvocation.MyCommand.Path
    exit
}

$ProjectDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $ProjectDir

$ResultsDir = "$ProjectDir\results"
New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null

# ── Step 1: Train P2 model (if not skipped) ──────────────────────────
if (-not $SkipTrain) {
    Write-Host "=" * 60
    Write-Host "STEP 1: Train YOLO11m with P2 head"
    Write-Host "=" * 60

    & $PSScriptRoot\train_p2.ps1 -Dataset $Dataset -Epochs $Epochs -Batch $Batch
    if ($LASTEXITCODE -ne 0) {
        Write-Error "P2 training failed."
        exit 1
    }
    Write-Host "P2 training complete."
}

if ($EvalOnly) {
    Write-Host "Evaluation-only mode. Skipping training."
}

# ── Step 2: Evaluate all combinations ────────────────────────────────
Write-Host ""
Write-Host "=" * 60
Write-Host "STEP 2: Evaluate all ablation combinations"
Write-Host "=" * 60

# 2a: Standard inference on existing experiments
Write-Host ""
Write-Host "--- Standard inference (baseline) ---"
uv run python scripts/evaluate_test.py --all --output $ResultsDir\standard_eval.json

# 2b: SAHI inference on all experiments
Write-Host ""
Write-Host "--- SAHI inference ---"
uv run python scripts/evaluate_test.py --all --sahi --slice-size 640 --overlap 0.2 --output $ResultsDir\sahi_eval.json

# ── Step 3: Print combined ablation table ────────────────────────────
Write-Host ""
Write-Host "=" * 60
Write-Host "STEP 3: Combined Ablation Results"
Write-Host "=" * 60

# Load and compare results
$stdResults = Get-Content $ResultsDir\standard_eval.json | ConvertFrom-Json
$sahiResults = Get-Content $ResultsDir\sahi_eval.json | ConvertFrom-Json

Write-Host ""
Write-Host "Standard Inference:"
Write-Host ("{0,-20} {1,8} {2,10} {3,10} {4,8}" -f "Experiment", "mAP50", "mAP50-95", "Precision", "Recall")
Write-Host ("-" * 60)

$stdResults.PSObject.Properties | ForEach-Object {
    $name = $_.Name
    $data = $_.Value
    $m = $data.metrics
    if ($m) {
        Write-Host ("{0,-20} {1,8:F2} {2,10:F2} {3,10:F2} {4,8:F2}" -f $name, $m.mAP50, $m."mAP50-95", $m.Precision, $m.Recall)
    }
}

Write-Host ""
Write-Host "SAHI Inference (slice=640, overlap=0.2):"
Write-Host ("{0,-20} {1,8} {2,10} {3,10} {4,8}" -f "Experiment", "mAP50", "mAP50-95", "Precision", "Recall")
Write-Host ("-" * 60)

$sahiResults.PSObject.Properties | ForEach-Object {
    $name = $_.Name
    $data = $_.Value
    $m = $data.metrics
    if ($m) {
        Write-Host ("{0,-20} {1,8:F2} {2,10:F2} {3,10:F2} {4,8:F2}" -f $name, $m.mAP50, $m."mAP50-95", $m.Precision, $m.Recall)
    }
}

Write-Host ""
Write-Host "P2 SAHI+P2 Ablation complete. Results saved to $ResultsDir"
