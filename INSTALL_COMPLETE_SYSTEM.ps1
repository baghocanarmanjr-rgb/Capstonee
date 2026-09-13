param(
    [string]$ExistingSystem = "C:\Users\User\Downloads\KAPSTON\capstone",
    [string]$Target = "C:\Users\User\Downloads\KAPSTON_COMPLETE\capstone"
)

$ErrorActionPreference = "Stop"
$PackageSystem = Join-Path $PSScriptRoot "capstone"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

if (-not (Test-Path $PackageSystem)) {
    throw "Package capstone folder was not found beside this installer."
}

$existingFull = [System.IO.Path]::GetFullPath($ExistingSystem)
$targetFull = [System.IO.Path]::GetFullPath($Target)
$migrationSource = $existingFull

Write-Host "Package:  $PackageSystem" -ForegroundColor Cyan
Write-Host "Existing: $existingFull" -ForegroundColor Cyan
Write-Host "Target:   $targetFull" -ForegroundColor Cyan

# If the user installs over the current system, move it safely first and migrate from the backup.
if ((Test-Path $targetFull) -and ($targetFull.TrimEnd('\') -ieq $existingFull.TrimEnd('\'))) {
    $backup = "${targetFull}_BACKUP_$stamp"
    Move-Item -LiteralPath $targetFull -Destination $backup
    $migrationSource = $backup
    Write-Host "Existing system moved to: $backup" -ForegroundColor Yellow
}
elseif (Test-Path $targetFull) {
    $backup = "${targetFull}_BACKUP_$stamp"
    Move-Item -LiteralPath $targetFull -Destination $backup
    Write-Host "Previous target moved to: $backup" -ForegroundColor Yellow
}

New-Item -ItemType Directory -Force -Path $targetFull | Out-Null
Copy-Item -Path (Join-Path $PackageSystem "*") -Destination $targetFull -Recurse -Force

# Preserve the user's current database, active trained model, uploads, and working data files.
if (Test-Path $migrationSource) {
    $oldDb = Join-Path $migrationSource "instance\psms.db"
    if (Test-Path $oldDb) {
        New-Item -ItemType Directory -Force -Path (Join-Path $targetFull "instance") | Out-Null
        Copy-Item -LiteralPath $oldDb -Destination (Join-Path $targetFull "instance\psms.db") -Force
        Write-Host "Existing database migrated." -ForegroundColor Green
    }

    $oldModel = Join-Path $migrationSource "model\fasttext_nlp"
    $requiredModelFiles = @(
        "fasttext_model.model",
        "status_classifier.joblib",
        "issue_classifier.joblib",
        "labels.json",
        "training_manifest.json",
        "fasttext_feature_schema_v2.py"
    )
    $modelFilesReady = $true
    foreach ($requiredFile in $requiredModelFiles) {
        if (-not (Test-Path (Join-Path $oldModel $requiredFile))) {
            $modelFilesReady = $false
        }
    }
    $modelManifestReady = $false
    if ($modelFilesReady) {
        try {
            $manifest = Get-Content (Join-Path $oldModel "training_manifest.json") -Raw | ConvertFrom-Json
            $modelManifestReady = (
                $manifest.decision_mode -eq "AI_ONLY_FASTTEXT_DUAL_CLASSIFIER" -and
                $manifest.deployed_model -eq "FastText + Logistic Regression" -and
                $manifest.feature_schema -eq "FASTTEXT_NUMERIC_RELATION_V2" -and
                $manifest.quality_gate_passed -eq $true
            )
        } catch {
            $modelManifestReady = $false
        }
    }
    if ($modelFilesReady -and $modelManifestReady) {
        $newModel = Join-Path $targetFull "model\fasttext_nlp"
        Remove-Item -LiteralPath $newModel -Recurse -Force -ErrorAction SilentlyContinue
        New-Item -ItemType Directory -Force -Path (Split-Path $newModel) | Out-Null
        Copy-Item -LiteralPath $oldModel -Destination $newModel -Recurse -Force
        Write-Host "Compatible active V2 model folder migrated." -ForegroundColor Green
    } else {
        Write-Host "No compatible active V2 model folder was found in the existing system. The system will still run through PO." -ForegroundColor Yellow
        Write-Host "Install the trained model ZIP later using install_colab_model_V2.py." -ForegroundColor Yellow
    }

    $oldUploads = Join-Path $migrationSource "uploads"
    if (Test-Path $oldUploads) {
        New-Item -ItemType Directory -Force -Path (Join-Path $targetFull "uploads") | Out-Null
        Copy-Item -Path (Join-Path $oldUploads "*") -Destination (Join-Path $targetFull "uploads") -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "Existing uploads migrated." -ForegroundColor Green
    }

    foreach ($name in @("PPMP_Data.xlsx", "COA_Inventory_Dataset.xlsx", "for-coa.xlsx")) {
        $sourceFile = Join-Path $migrationSource "data\$name"
        if (Test-Path $sourceFile) {
            Copy-Item -LiteralPath $sourceFile -Destination (Join-Path $targetFull "data\$name") -Force
        }
    }
}

# Confirm that the package contains the verified V2 runtime.
$expectedHash = "67CD33E0F41574693908277B502209CBCC56F7A64F71441ADD3202F255F0D9AC"
$runtimePath = Join-Path $targetFull "fasttext_nlp_runtime.py"
$actualHash = (Get-FileHash -LiteralPath $runtimePath -Algorithm SHA256).Hash.ToUpper()
if ($actualHash -ne $expectedHash) {
    throw "Installed runtime checksum mismatch. Expected $expectedHash but found $actualHash"
}
Write-Host "V2 runtime checksum verified." -ForegroundColor Green

Set-Location $targetFull
python -m py_compile .\app.py .\aoq_flexible.py .\inventory_dataset.py .\fasttext_nlp_runtime.py

Write-Host "" 
Write-Host "COMPLETE SYSTEM INSTALLED SUCCESSFULLY" -ForegroundColor Green
Write-Host "Location: $targetFull"
Write-Host "Next commands:" -ForegroundColor Cyan
Write-Host "  cd `"$targetFull`""
Write-Host "  powershell -ExecutionPolicy Bypass -File .\INSTALL_DEPENDENCIES.ps1"
Write-Host "  powershell -ExecutionPolicy Bypass -File .\START_SYSTEM.ps1"
Write-Host "Open: http://127.0.0.1:5000"
Write-Host "Default login for a fresh database: admin / admin123"
