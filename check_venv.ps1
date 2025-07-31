# Check virtual environment
Write-Host "Checking .venv directory contents..."

# Check if .venv exists
if (Test-Path ".\.venv") {
    Write-Host "✓ .venv directory exists"
    
    # Check Scripts directory
    if (Test-Path ".\.venv\Scripts") {
        Write-Host "✓ Scripts directory exists"
        Write-Host "`nContents of .venv\Scripts:"
        Get-ChildItem ".\.venv\Scripts" | Select-Object Name
    } else {
        Write-Host "✗ Scripts directory not found"
    }
    
    # Check for python.exe
    if (Test-Path ".\.venv\Scripts\python.exe") {
        Write-Host "`n✓ python.exe found"
        # Get python version
        & ".\.venv\Scripts\python.exe" --version
    } else {
        Write-Host "`n✗ python.exe not found"
    }
} else {
    Write-Host "✗ .venv directory not found"
}

Write-Host "`nCurrent Python being used:"
python --version
Get-Command python | Select-Object Source