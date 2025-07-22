@echo off
REM F1 Pipeline Runner for Windows

echo F1 Prize Picks Pipeline
echo ======================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "F1_Pipeline_Integration.ipynb" (
    echo ERROR: Cannot find F1_Pipeline_Integration.ipynb
    echo Please run this script from the notebooks/advanced directory
    pause
    exit /b 1
)

REM Run the test script first
echo Running setup test...
python test_windows_setup.py
echo.

REM Ask user to continue
set /p continue=Do you want to run the pipeline? (y/n): 
if /i not "%continue%"=="y" exit /b 0

REM Convert and run the notebook
echo.
echo Converting notebook to script...
python -m jupyter nbconvert --to script F1_Pipeline_Integration.ipynb --output temp_pipeline

echo Running pipeline...
python temp_pipeline.py

REM Cleanup
if exist "temp_pipeline.py" del temp_pipeline.py

echo.
echo Pipeline execution complete!
pause