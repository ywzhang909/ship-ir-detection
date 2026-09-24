$logFile = "D:\Projects\AUVDect\data\ship\yolo-training\T22_large_training.log"
$startTime = Get-Date
"=== T22_large started at $startTime ===" | Out-File -FilePath $logFile -Encoding utf8
Set-Location "D:\Projects\AUVDect\data\ship\yolo-training"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PATH = "D:\Projects\AUVDect\data\ship\.venv\Scripts;$env:PATH"
$cmd = "uv run python train.py --dataset nslsr-lwir --preprocess --preprocess-method awb_rpca_dual_fusion --model yolo11l.pt --use-fusion --fusion-hidden 16 --fusion-layers 2 --fusion-attention se --fusion-lr-scale 10.0 --epochs 150 --batch 4 --imgsz 640 --device 0 --name T22_large --project ship-detection --wandb"
$startTime = Get-Date
Invoke-Expression $cmd 2>&1 | Out-File -FilePath $logFile -Encoding utf8 -Append
$endTime = Get-Date
$duration = $endTime - $startTime
"=== T22_large finished at $endTime (duration: $($duration.TotalMinutes) minutes) ===" | Out-File -FilePath $logFile -Encoding utf8 -Append
