@echo off
REM Daily Model Training Script for Courant AI Service
REM This script runs both Isolation Forest and LSTM model training

echo ========================================
echo Courant AI Service - Daily Model Training
echo Started at: %date% %time%
echo ========================================

REM Change to the script directory
cd /d "c:\Users\User\Documents\Work\courant_ai_service"

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found, using system Python
)

echo.
echo ----------------------------------------
echo Training Isolation Forest Models...
echo ----------------------------------------
python train_models_by_profile.py
if %ERRORLEVEL% neq 0 (
    echo ERROR: Isolation Forest training failed with exit code %ERRORLEVEL%
    goto :error
)

echo.
echo ----------------------------------------
echo Training LSTM Models...
echo ----------------------------------------
python train_lstm_models_by_profile.py
if %ERRORLEVEL% neq 0 (
    echo ERROR: LSTM training failed with exit code %ERRORLEVEL%
    goto :error
)

echo.
echo ========================================
echo Training completed successfully!
echo Finished at: %date% %time%
echo ========================================

REM Backup old models (optional)
if exist "trained_models.joblib" (
    echo Creating backup of Isolation Forest models...
    copy "trained_models.joblib" "backups\trained_models_%date:~-4,4%%date:~-10,2%%date:~-7,2%.joblib" >nul 2>&1
)

if exist "trained_lstm_models.joblib" (
    echo Creating backup of LSTM models...
    copy "trained_lstm_models.joblib" "backups\trained_lstm_models_%date:~-4,4%%date:~-10,2%%date:~-7,2%.joblib" >nul 2>&1
)

goto :end

:error
echo.
echo ========================================
echo Training failed at: %date% %time%
echo Check the logs above for details
echo ========================================
exit /b 1

:end
exit /b 0
