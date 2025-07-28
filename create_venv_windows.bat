@echo off
echo Creating Python virtual environment for F1 project...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

REM Remove existing virtual environment if it exists
if exist ".venv" (
    echo Removing existing virtual environment...
    rmdir /s /q .venv
)

REM Create new virtual environment
echo Creating new virtual environment...
python -m venv .venv

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing requirements from requirements.txt...
pip install -r requirements.txt

REM Install Jupyter kernel
echo Installing Jupyter kernel...
python -m ipykernel install --user --name=f1-venv --display-name="F1 Project (venv)"

REM Create helper scripts in the virtual environment
echo.
echo Creating helper scripts...

REM Create activate_and_cd.bat
(
echo @echo off
echo REM Activate venv and navigate to notebooks/advanced
echo call "%%~dp0activate.bat"
echo cd /d "%%~dp0..\..\notebooks\advanced" 2^>nul
echo echo Virtual environment activated and moved to notebooks/advanced
) > .venv\Scripts\activate_and_cd.bat

REM Create run_weather_fetch.bat
(
echo @echo off
echo REM Run weather fetch script with venv Python
echo "%%~dp0python.exe" "%%~dp0..\..\notebooks\advanced\fetch_weather_data.py" %%*
) > .venv\Scripts\run_weather_fetch.bat

REM Create run_pipeline.bat
(
echo @echo off
echo REM Run F1 pipeline with venv Python
echo "%%~dp0python.exe" "%%~dp0..\..\notebooks\advanced\run_f1_pipeline.py" %%*
) > .venv\Scripts\run_pipeline.bat

echo.
echo ========================================
echo Virtual environment created successfully!
echo ========================================
echo.
echo Helper scripts created in .venv\Scripts\:
echo   activate_and_cd.bat     - Activates venv and moves to notebooks/advanced
echo   run_weather_fetch.bat   - Runs weather fetch script directly
echo   run_pipeline.bat        - Runs F1 pipeline directly
echo.
echo To activate the environment manually, run:
echo   .venv\Scripts\activate.bat
echo   OR
echo   .venv\Scripts\activate_and_cd.bat
echo.
echo To run scripts directly without activation:
echo   .venv\Scripts\run_weather_fetch.bat --status
echo   .venv\Scripts\run_pipeline.bat
echo.
echo In VS Code:
echo 1. Press Ctrl+Shift+P to open command palette
echo 2. Type "Python: Select Interpreter"
echo 3. Choose the interpreter at: .venv\Scripts\python.exe
echo.
echo For Jupyter notebooks:
echo 1. Select the kernel "F1 Project (venv)" from the kernel dropdown
echo.
pause