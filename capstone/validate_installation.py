"""Basic installation validation. Run after dependencies are installed."""
from __future__ import annotations
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED_FILES = [
    ROOT / "app.py",
    ROOT / "aoq_flexible.py",
    ROOT / "inventory_dataset.py",
    ROOT / "fasttext_nlp_runtime.py",
    ROOT / "requirements.txt",
    ROOT / "SYSTEM_SMOKE_TEST.py",
    ROOT / "templates" / "aoq.html",
    ROOT / "templates" / "po.html",
    ROOT / "templates" / "nlp.html",
]
MODULES = ["flask", "werkzeug", "waitress", "openpyxl", "PIL", "pytesseract", "fitz", "docx", "numpy", "pandas", "gensim", "sklearn", "joblib"]

missing_files = [str(path.relative_to(ROOT)) for path in REQUIRED_FILES if not path.exists()]
missing_modules = []
for name in MODULES:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing_modules.append(f"{name}: {exc}")

print("Required files:", "OK" if not missing_files else "MISSING")
for value in missing_files:
    print(" -", value)
print("Python packages:", "OK" if not missing_modules else "MISSING/FAILED")
for value in missing_modules:
    print(" -", value)

if missing_files or missing_modules:
    raise SystemExit("Installation is incomplete. Run INSTALL_DEPENDENCIES.ps1 and check the files above.")

sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import app as system
system.init_db()
print("Database initialization: OK")
print("NLP model status:")
print(json.dumps(system.trained_nlp_model_status(), indent=2, default=str))
print("System validation completed.")
