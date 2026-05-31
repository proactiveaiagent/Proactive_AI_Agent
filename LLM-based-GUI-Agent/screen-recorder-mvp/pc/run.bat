@echo off
cd /d "%~dp0"
if not exist "venv" (
    echo Creating venv...
    python -m venv venv
)
call venv\Scripts\activate.bat
pip install -r requirements.txt -q
python main.py
