@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Building Screen Recorder PC desktop app...
echo.

if not exist "venv\Scripts\activate.bat" (
    echo Creating venv...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing runtime and build dependencies...
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124 -q
pip install -r requirements.txt -q
pip install pyinstaller -q

echo.
echo Running PyInstaller...
pyinstaller --noconfirm screen_recorder_pc.spec

if %ERRORLEVEL% neq 0 (
    echo Build failed.
    exit /b 1
)

echo.
echo Build done. Output folder: dist\ScreenRecorderPC
echo Run: dist\ScreenRecorderPC\ScreenRecorderPC.exe
echo.
pause
