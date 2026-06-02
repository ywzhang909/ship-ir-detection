# T4: Shape-IoU training — full run
# Compares Shape-IoU vs standard CIoU for small IR ship detection.
# Shape-IoU adds shape-aware distance (ω_w, ω_h) and aspect ratio (Δθ)
# penalty to the standard IoU loss.
#
# Run from yolo-training/ directory:
#   .\scripts\train_t4_shape_iou.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$run_name = "T4_shape_iou_$timestamp"

Write-Host "=== T4 Shape-IoU Training ===" -ForegroundColor Cyan
Write-Host "Run name: $run_name"
Write-Host ""

uv run python train.py `
    --model yolo11m.pt `
    --dataset ship `
    --epochs 200 `
    --batch 16 `
    --imgsz 1280 `
    --device 0 `
    --use-fusion `
    --fusion-hidden 256 `
    --fusion-layers 2 `
    --fusion-attention 4 `
    --fusion-lr-scale 1e-3 `
    --fusion-residual-target 0.5 `
    --fusion-residual-warmup 10 `
    --iou-loss shape-iou `
    --name $run_name `
    --project ship-detection `
    --wandb `
    --run-name $run_name

if ($LASTEXITCODE -eq 0) {
    Write-Host "T4 completed successfully!" -ForegroundColor Green
} else {
    Write-Host "T4 FAILED (exit code: $LASTEXITCODE)" -ForegroundColor Red
}
