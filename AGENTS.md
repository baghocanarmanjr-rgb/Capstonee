# Project instructions

## Overview

This project is the Municipality of Asuncion Supply and Property

Management System.

It uses Python, Flask, SQLite, HTML templates, Excel data,

OCR, and FastText with Logistic Regression classifiers.

## Project structure

- capstone/app.py: application routes and procurement workflow.

- capstone/templates/: HTML pages.

- capstone/static/: interface styling.

- capstone/instance/psms.db: application database.

- capstone/aoq_flexible.py: quotation document processing.

- capstone/inventory_dataset.py: inventory dataset handling.

- capstone/fasttext_nlp_runtime.py: NLP model loading and prediction.

- capstone/model/fasttext_nlp/: active NLP model assets.

Confirm these paths against the current checkout before making changes.

## Workflow

PPMP -> APP -> PR -> AOQ -> PO -> NLP verification

-> IAR/AIR -> Inventory -> RIS/ICS.

Preserve these rules:

- Create PRs only from approved APP records.

- Require exactly one winning supplier before saving an AOQ.

- Generate POs from the approved AOQ winning offer.

- Allow PO approval before NLP verification.

- Require an eligible approved NLP result before IAR processing.

- Add accepted items to inventory when IAR is approved.

- Deduct issued quantities only once when RIS/ICS is approved.

## Development

- Use the demo branch for this assignment.

- Preserve existing user changes.

- Keep changes focused on the requested task.

- Do not edit generated __pycache__ or .pyc files.

- Use a temporary database for tests.

- Do not overwrite the existing database or trained models

  without explicit authorization.

## Validation

- Inspect SYSTEM_SMOKE_TEST.py before running workflow tests.

- Run relevant checks after code changes.

- Report which checks passed and which were not run.

- Model files being present does not prove predictions work.
