@echo off
chcp 65001 > nul
title Python Trainer_DLAA.py

REM Ayarlar
SET PYTHON_EXECUTABLE=python
SET SCRIPT_PATH=Trainer_DLAA.py
SET DATASET_NAME=dataset
SET OUTPUT_DIR=runs
SET EPOCHS=120
SET BATCH_SIZE=8
SET PATCH_SIZE=192
SET LEARNING_RATE=1e-4
SET WORKERS=4
SET USE_AMP=true

REM Log Dosyası
SET LOG_FILE=trainer_dlaa_log.txt

REM Log dosyasını oluştur
echo %date% %time% - Script başlatıldı: %SCRIPT_PATH% >> "%LOG_FILE%"
echo %date% %time% - Dataset: %DATASET_NAME% >> "%LOG_FILE%"
echo %date% %time% - Output Dir: %OUTPUT_DIR% >> "%LOG_FILE%"
echo %date% %time% - Epochs: %EPOCHS% >> "%LOG_FILE%"
echo %date% %time% - Batch Size: %BATCH_SIZE% >> "%LOG_FILE%"
echo %date% %time% - Patch Size: %PATCH_SIZE% >> "%LOG_FILE%"
echo %date% %time% - Learning Rate: %LEARNING_RATE% >> "%LOG_FILE%"
echo %date% %time% - Workers: %WORKERS% >> "%LOG_FILE%"
echo %date% %time% - AMP Enabled: %USE_AMP% >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

REM Python Komutu
echo Python scripti çalıştırılıyor: %SCRIPT_PATH%
echo Parametreler: --dataset %DATASET_NAME% --out-dir %OUTPUT_DIR% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --patch-size %PATCH_SIZE% --lr %LEARNING_RATE% --workers %WORKERS% --amp

REM Automatic Mixed Precision
IF "%USE_AMP%"=="true" (
    SET AMP_ARG=--amp
) ELSE (
    SET AMP_ARG=
)

REM Komutu çalıştır
%PYTHON_EXECUTABLE% %SCRIPT_PATH% ^
    --dataset %DATASET_NAME% ^
    --out-dir %OUTPUT_DIR% ^
    --epochs %EPOCHS% ^
    --batch-size %BATCH_SIZE% ^
    --patch-size %PATCH_SIZE% ^
    --lr %LEARNING_RATE% ^
    --workers %WORKERS% ^
    %AMP_ARG%

REM Hata durumunu kontrol et
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo ==================================================
    echo HATA: Python scripti bir hata ile tamamlandı.
    echo ERRORLEVEL: %ERRORLEVEL%
    echo Detaylar için konsol çıktısına ve log dosyasına bakınız: %LOG_FILE%
    echo ==================================================
    echo %date% %time% - HATA ile tamamlandı. ERRORLEVEL: %ERRORLEVEL% >> "%LOG_FILE%"
    goto :end
) ELSE (
    echo.
    echo ==================================================
    echo Python scripti başarıyla tamamlandı.
    echo Detaylar için konsol çıktısına ve log dosyasına bakınız: %LOG_FILE%
    echo ==================================================
    echo %date% %time% - Başarıyla tamamlandı. >> "%LOG_FILE%"
)

:end
EXIT /B %ERRORLEVEL%