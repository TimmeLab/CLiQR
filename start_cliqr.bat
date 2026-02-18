@echo off
REM CLiQR Recording System Launcher
REM This script starts the Solara GUI for data recording

echo ==========================================
echo    CLiQR Recording System
echo    Timme Lab - University of Cincinnati
echo ==========================================
echo.

REM Activate the conda environment
echo Activating environment...
set "MF=%USERPROFILE%\AppData\Local\miniforge3"
call "%MF%\Scripts\activate.bat" "%MF%"
call conda activate cliqr
if errorlevel 1 (
    echo ERROR: Could not activate 'cliqr' environment
    echo Please run: conda env create --file environment.yml
    pause
    exit /b 1
)

echo Starting application...
echo.
echo The application will open in your default browser.
echo When finished, close the browser and press Ctrl+C here.
echo.

REM Start the Solara app
solara run recording_gui.py

pause