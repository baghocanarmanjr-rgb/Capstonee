MUNICIPALITY OF ASUNCION
COMPLETE SUPPLY AND PROPERTY MANAGEMENT SYSTEM

CONNECTED WORKFLOW
PPMP -> APP -> PR -> AOQ -> PO -> PR-PO NLP Verification -> IAR/AIR -> Inventory -> RIS/ICS

IMPORTANT WORKFLOW RULES
- PR is created only from an approved APP.
- AOQ upload remains layout-flexible for PDF, image, DOCX, XLSX/XLSM, CSV, and TXT.
- The selected PR and detected AOQ result are displayed together during review.
- Detected values remain editable because unusual layouts and scans can require correction.
- Exactly one winning supplier is selected before AOQ saving.
- PO is generated from the approved AOQ winning offer.
- You can complete and approve the PO without running NLP.
- NLP is required before IAR/AIR and the later inventory workflow.
- IAR approval adds accepted PO items to Inventory.
- RIS/ICS approval deducts quantities from live inventory only once.

DEFAULT LOGIN FOR A FRESH DATABASE
Username: admin
Password: admin123

The password is stored as a secure hash. Use User Management to create accounts or reset passwords.

RECOMMENDED INSTALLATION
Use INSTALL_COMPLETE_SYSTEM.ps1 from the package root. It installs the complete
system into a separate folder and migrates your current database, compatible V2
model, uploads, and working data files when found.

AFTER INSTALLATION
1. Open PowerShell in the installed capstone folder.
2. Install and validate everything:
   powershell -ExecutionPolicy Bypass -File .\INSTALL_DEPENDENCIES.ps1
3. Start the system:
   powershell -ExecutionPolicy Bypass -File .\START_SYSTEM.ps1
4. Open:
   http://127.0.0.1:5000

INSTALL_DEPENDENCIES.ps1 automatically runs:
- validate_installation.py
- SYSTEM_SMOKE_TEST.py

The smoke test uses a temporary database. It does not alter your live records.

TRAINED V2 NLP MODEL
The complete system uses the verified fasttext_nlp_runtime.py V2 runtime.
The uploaded KAPSTON(2).zip did not contain your newly trained V2 model assets.
The package installer automatically migrates your existing compatible active model when found.

Manual model installation command:
python .\install_colab_model_V2.py "C:\path\FASTTEXT_NLP_MODEL_AI_ONLY (2).zip"

Model status check:
python -c "import os,json; from fasttext_nlp_runtime import model_status; print(json.dumps(model_status(os.path.join(os.getcwd(),'model','fasttext_nlp')),indent=2))"

Expected after a successful model migration/installation:
"active": true

FILES PRESERVED BY THE PACKAGE INSTALLER WHEN FOUND
- instance\psms.db
- model\fasttext_nlp\
- uploads\
- data\PPMP_Data.xlsx
- data\COA_Inventory_Dataset.xlsx
- data\for-coa.xlsx

The system remains usable through approved PO even when the trained NLP model is not installed.
