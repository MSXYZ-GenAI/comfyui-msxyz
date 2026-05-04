@echo off
chcp 65001 > nul

set SCRIPT_PATH=Trainer_DLAA.py
set DATASET_PATH=dataset
set OUT_DIR=runs
set RESUME_PATH=runs\checkpoints\last.pth

echo Resuming DLAA training...
echo Script: %SCRIPT_PATH%
echo Dataset: %DATASET_PATH%
echo Checkpoint: %RESUME_PATH%

python %SCRIPT_PATH% --dataset %DATASET_PATH% --out-dir %OUT_DIR% --epochs 120 --batch-size 8 --patch-size 192 --lr 1e-4 --workers 4 --amp --resume %RESUME_PATH%

pause