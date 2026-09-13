$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Checking Python..." -ForegroundColor Cyan
python --version

Write-Host "Installing required Python packages..." -ForegroundColor Cyan
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt

Write-Host "Running installation validation..." -ForegroundColor Cyan
python .\validate_installation.py

Write-Host "Running isolated complete-workflow smoke test..." -ForegroundColor Cyan
python .\SYSTEM_SMOKE_TEST.py

Write-Host "Dependencies installed and system tests passed." -ForegroundColor Green
Write-Host "For scanned AOQ images/PDFs, install Tesseract OCR separately when needed." -ForegroundColor Yellow
