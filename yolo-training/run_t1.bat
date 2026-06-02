@echo off
REM T1: 融合模块架构搜索 - 4 configs sequential
REM hidden: 32/64, layers: 3/4, attention: cbam
REM residual_target: 0.5 (same as Run2 baseline for fair comparison)

set COMMON_ARGS=--dataset nslsr-lwir --preprocess --preprocess-method awb_dual_fusion --use-fusion --fusion-lr-scale 10.0 --fusion-residual-target 0.5 --epochs 150 --batch 16 --imgsz 640

echo ========================================
echo T1/4: h=32, layers=3, attn=cbam
echo ========================================
uv run python train.py %COMMON_ARGS% --fusion-hidden 32 --fusion-layers 3 --fusion-attention cbam --project runs/tune_fusion --name T1_h32_l3_cbam --wandb
if %ERRORLEVEL% neq 0 (
    echo T1/4 FAILED at h32_l3_cbam
    exit /b %ERRORLEVEL%
)

echo ========================================
echo T2/4: h=32, layers=4, attn=cbam
echo ========================================
uv run python train.py %COMMON_ARGS% --fusion-hidden 32 --fusion-layers 4 --fusion-attention cbam --project runs/tune_fusion --name T1_h32_l4_cbam --wandb
if %ERRORLEVEL% neq 0 (
    echo T2/4 FAILED at h32_l4_cbam
    exit /b %ERRORLEVEL%
)

echo ========================================
echo T3/4: h=64, layers=3, attn=cbam
echo ========================================
uv run python train.py %COMMON_ARGS% --fusion-hidden 64 --fusion-layers 3 --fusion-attention cbam --project runs/tune_fusion --name T1_h64_l3_cbam --wandb
if %ERRORLEVEL% neq 0 (
    echo T3/4 FAILED at h64_l3_cbam
    exit /b %ERRORLEVEL%
)

echo ========================================
echo T4/4: h=64, layers=4, attn=cbam
echo ========================================
uv run python train.py %COMMON_ARGS% --fusion-hidden 64 --fusion-layers 4 --fusion-attention cbam --project runs/tune_fusion --name T1_h64_l4_cbam --wandb
if %ERRORLEVEL% neq 0 (
    echo T4/4 FAILED at h64_l4_cbam
    exit /b %ERRORLEVEL%
)

echo ========================================
echo ALL T1 CONFIGS COMPLETED
echo ========================================
