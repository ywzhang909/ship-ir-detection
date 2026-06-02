# T3: Smoke test Shape-IoU + fusion (yolo11n, 2 epochs, batch=2)
# Quick validation: model loads, loss runs, training loop completes
#
# Run from yolo-training/:
#   .\scripts\train_t3_smoke_shape_iou.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$run_name = "T3_smoke_shape_iou_$timestamp"

Write-Host "=== T3 Smoke Test: Shape-IoU + Fusion ===" -ForegroundColor Cyan
Write-Host "Run name: $run_name"
Write-Host ""

uv run python train.py `
    --model yolo11n.pt `
    --dataset ship `
    --epochs 2 `
    --batch 2 `
    --imgsz 640 `
    --device 0 `
    --use-fusion `
    --fusion-hidden 128 `
    --fusion-layers 1 `
    --fusion-attention 2 `
    --fusion-lr-scale 1e-3 `
    --fusion-residual-target 0.5 `
    --fusion-residual-warmup 5 `
    --iou-loss shape-iou `
    --name $run_name `
    --project ship-detection `
    --run-name $run_name

if ($LASTEXITCODE -eq 0) {
    Write-Host "T3 smoke test PASSED!" -ForegroundColor Green
} else {
    Write-Host "T3 smoke test FAILED (exit code: $LASTEXITCODE)" -ForegroundColor Red
}
