# PowerShell script to create virtual environment for F1 project

Write-Host "Creating Python virtual environment for F1 project..." -ForegroundColor Green

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Cyan
}
catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8 or higher from python.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Remove existing virtual environment if it exists
if (Test-Path ".venv") {
    Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
    Remove-Item -Path ".venv" -Recurse -Force
}

# Create new virtual environment
Write-Host "Creating new virtual environment..." -ForegroundColor Green
python -m venv .venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing requirements from requirements.txt..." -ForegroundColor Green
pip install -r requirements.txt

# Install Jupyter kernel
Write-Host "Installing Jupyter kernel..." -ForegroundColor Green
python -m ipykernel install --user --name=f1-venv --display-name="F1 Project (venv)"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Virtual environment created successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the environment manually, run:" -ForegroundColor Yellow
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "In VS Code:" -ForegroundColor Yellow
Write-Host "1. Press Ctrl+Shift+P to open command palette" -ForegroundColor White
Write-Host "2. Type 'Python: Select Interpreter'" -ForegroundColor White
Write-Host "3. Choose the interpreter at: .venv\Scripts\python.exe" -ForegroundColor White
Write-Host ""
Write-Host "For Jupyter notebooks:" -ForegroundColor Yellow
Write-Host "1. Select the kernel 'F1 Project (venv)' from the kernel dropdown" -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to exit"