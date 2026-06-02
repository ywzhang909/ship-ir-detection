# Train YOLO11m with P2 Small-Object Detection Head
#
# Usage:
#   .\scripts\train_p2.ps1                       # NSLSR-LWIR
#   .\scripts\train_p2.ps1 -Dataset nslsr-swir   # NSLSR-SWIR
#   .\scripts\train_p2.ps1 -Epochs 100 -Batch 8  # Quick test
#
# The P2 head adds a stride-4 detection level for improved small-object
# recall. This script loads the P2 YAML config and transfers pretrained
# weights from yolo11m.pt for all shared layers.
#
# Output directory:
#   runs/detect/ship-detection/{dataset}-yolo11m_p2/
#
# Compare with baseline (no P2):
#   uv run python train.py --dataset nslsr-lwir --wandb
#
# Evaluation:
#   uv run python scripts/evaluate_test.py --all

param(
    [string]$Dataset = "nslsr-lwir",
    [int]$Epochs = 200,
    [int]$Batch = 16,
    [int]$Imgsz = 640,
    [string]$Device = "0",
    [switch]$Wandb = $true,
    [switch]$Help
)

if ($Help) {
    Get-Help $MyInvocation.MyCommand.Path
    exit
}

$ProjectDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $ProjectDir

Write-Host "=" * 60
Write-Host "P2 SMALL-OBJECT HEAD TRAINING"
Write-Host "Dataset: $Dataset"
Write-Host "Epochs:  $Epochs"
Write-Host "Batch:   $Batch"
Write-Host "Imgsz:   $Imgsz"
Write-Host "=" * 60

$wandbArg = if ($Wandb) { "--wandb" } else { "" }

uv run python train.py `
    --model configs/yolo11m_p2.yaml `
    --dataset $Dataset `
    --epochs $Epochs `
    --batch $Batch `
    --imgsz $Imgsz `
    --device $Device `
    $wandbArg

Write-Host "P2 training complete."
