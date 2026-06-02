@echo off
REM T2: 不同的 residual_alpha target (0.3/0.7/1.0)
REM Uses best T1 arch (use best from T1 results)
REM residual_warmup=10, lr_scale=10x

set COMMON_ARGS=--dataset nslsr-lwir --preprocess --preprocess-method awb_dual_fusion --use-fusion --fusion-lr-scale 10.0 --fusion-hidden 32 --fusion-layers 3 --fusion-attention cbam --epochs 150 --batch 16 --imgsz 640

echo ========================================
echo T2/3: residual_target=0.3
echo ========================================
uv run python train.py %COMMON_ARGS% --fusion-residual-target 0.3 --project runs/tune_fusion --name T2_rt03 --wandb
if %ERRORLEVEL% neq 0 (
    echo T2/3 FAILED at rt03
    exit /b %ERRORLEVEL%
)

echo ========================================
echo T2/3: residual_target=0.7
echo ========================================
uv run python train.py %COMMON_ARGS% --fusion-residual-target 0.7 --project runs/tune_fusion --name T2_rt07 --wandb
if %ERRORLEVEL% neq 0 (
    echo T2/3 FAILED at rt07
    exit /b %ERRORLEVEL%
)

echo ========================================
echo T2/3: residual_target=1.0
echo ========================================
uv run python train.py %COMMON_ARGS% --fusion-residual-target 1.0 --project runs/tune_fusion --name T2_rt10 --wandb
if %ERRORLEVEL% neq 0 (
    echo T2/3 FAILED at rt10
    exit /b %ERRORLEVEL%
)

echo ========================================
echo ALL T2 CONFIGS COMPLETED
echo ========================================
