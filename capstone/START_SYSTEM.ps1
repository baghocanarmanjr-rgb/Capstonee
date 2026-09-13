$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
Write-Host "Starting Municipality of Asuncion Supply and Property Management System..." -ForegroundColor Green
Write-Host "Open http://127.0.0.1:5000 in the browser." -ForegroundColor Cyan
python .\app.py
