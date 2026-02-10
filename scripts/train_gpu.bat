@echo off
echo ============================================
echo TeethIdentifier GPU Training
echo ============================================
echo.

REM Add CUDA 11.2 to PATH for this session
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin;%PATH%
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp;%PATH%

echo Checking GPU...
venv\Scripts\python.exe -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('GPU Count:', len(gpus)); [print(f'  GPU {i}: {gpu.name}') for i, gpu in enumerate(gpus)]"

if errorlevel 1 (
    echo.
    echo [ERROR] GPU check failed!
    pause
    exit /b 1
)

echo.
echo Starting training...
echo.

venv\Scripts\python.exe src\train.py

echo.
echo ============================================
echo Training Complete!
echo ============================================
pause
