@echo off
chcp 65001 > nul
title Clean Dataset Script


SET PYTHON_EXECUTABLE=python
SET SCRIPT_PATH=Clean_Dataset.py

REM Kullanılacak argümanlar
SET DATASET_DIR=dataset
SET MIN_WIDTH=384
SET MIN_HEIGHT=384
SET MIN_DETAIL=2.0
SET MIN_EDGE_DENSITY=0.018
SET MIN_THIN_DETAIL=0.8
SET MIN_CONTRAST=8
SET DUPLICATE_HAMMING=1

REM Log dosyası
SET LOG_FILE=clean_dataset_script.log

REM Log Dosyasına Bilgi Yazma
echo %date% %time% - ============================================================== >> "%LOG_FILE%"
echo %date% %time% - Script başlatıldı: %SCRIPT_PATH% >> "%LOG_FILE%"
echo %date% %time% - Dataset Dir: %DATASET_DIR% >> "%LOG_FILE%"
echo %date% %time% - Min Width: %MIN_WIDTH% >> "%LOG_FILE%"
echo %date% %time% - Min Height: %MIN_HEIGHT% >> "%LOG_FILE%"
echo %date% %time% - Min Detail: %MIN_DETAIL% >> "%LOG_FILE%"
echo %date% %time% - Min Edge Density: %MIN_EDGE_DENSITY% >> "%LOG_FILE%"
echo %date% %time% - Min Thin Detail: %MIN_THIN_DETAIL% >> "%LOG_FILE%"
echo %date% %time% - Min Contrast: %MIN_CONTRAST% >> "%LOG_FILE%"
echo %date% %time% - Duplicate Hamming: %DUPLICATE_HAMMING% >> "%LOG_FILE%"
echo %date% %time% - ============================================================== >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

REM Python Komutu
echo Python scripti çalıştırılıyor: %SCRIPT_PATH%
echo Argumanlar:
echo   --dataset %DATASET_DIR%
echo   --min-width %MIN_WIDTH%
echo   --min-height %MIN_HEIGHT%
echo   --min-detail %MIN_DETAIL%
echo   --min-edge-density %MIN_EDGE_DENSITY%
echo   --min-thin-detail %MIN_THIN_DETAIL%
echo   --min-contrast %MIN_CONTRAST%
echo   --duplicate-hamming %DUPLICATE_HAMMING%
echo.

REM Komutu çalıştır
%PYTHON_EXECUTABLE% %SCRIPT_PATH% ^
    --dataset %DATASET_DIR% ^
    --min-width %MIN_WIDTH% ^
    --min-height %MIN_HEIGHT% ^
    --min-detail %MIN_DETAIL% ^
    --min-edge-density %MIN_EDGE_DENSITY% ^
    --min-thin-detail %MIN_THIN_DETAIL% ^
    --min-contrast %MIN_CONTRAST% ^
    --duplicate-hamming %DUPLICATE_HAMMING%

REM Hata durumunu kontrol et
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo ==============================================================
    echo HATA: Python scripti bir hata ile tamamlandı.
    echo Hata Kodu (ERRORLEVEL): %ERRORLEVEL%
    echo Lütfen detaylar için log dosyasına bakınız: %LOG_FILE%
    echo ==============================================================
    echo %date% %time% - HATA ile tamamlandı. ERRORLEVEL: %ERRORLEVEL% >> "%LOG_FILE%"
    goto :end
) ELSE (
    echo.
    echo ==============================================================
    echo Python scripti başarıyla tamamlandı.
    echo Lütfen detaylar için log dosyasına bakınız: %LOG_FILE%
    echo ==============================================================
    echo %date% %time% - Başarıyla tamamlandı. >> "%LOG_FILE%"
)

:end
EXIT /B %ERRORLEVEL%