@echo off
REM T3: P2 小目标检测头 + 融合
REM Uses yolo11m_p2.yaml (4 detection heads: P2/P3/P4/P5)
REM NOTE: P2 head uses more memory, batch may need reduction

set COMMON_ARGS=--dataset nslsr-lwir --preprocess --preprocess-method awb_dual_fusion --use-fusion --fusion-hidden 16 --fusion-layers 2 --fusion-attention se --fusion-lr-scale 10.0 --fusion-residual-target 0.5 --epochs 150 --imgsz 640

echo ========================================
echo T3/1: P2 + fusion (batch=12 for P2 head)
echo ========================================
uv run python train.py %COMMON_ARGS% --model configs/yolo11m_p2.yaml --batch 12 --project runs/tune_fusion --name T3_P2_fusion --wandb
if %ERRORLEVEL% neq 0 (
    echo T3/1 FAILED
    exit /b %ERRORLEVEL%
)

echo ========================================
echo ALL T3 CONFIGS COMPLETED
echo ========================================
