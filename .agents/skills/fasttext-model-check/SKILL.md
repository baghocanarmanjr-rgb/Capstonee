---

name: fasttext-model-check

description: Diagnose the Asuncion system's FastText PR-to-PO verification model, including missing assets, incompatible metadata, loading errors, and unexpected predictions.

---

# FastText Model Check

## Inspect the current implementation

Read AGENTS.md, then inspect:

- capstone/fasttext_nlp_runtime.py

- capstone/model/fasttext_nlp/training_manifest.json

- capstone/model/fasttext_nlp/labels.json

- capstone/model/fasttext_nlp/fasttext_feature_schema_v2.py

Use the current runtime as the source of compatibility requirements.

## Check model assets

Check that the active model folder contains:

- fasttext_model.model

- Any separate arrays required by the saved FastText model

- status_classifier.joblib

- issue_classifier.joblib

- labels.json

- training_manifest.json

- fasttext_feature_schema_v2.py

Do not substitute files from previous, incorrect, or legacy

model folders without verifying their compatibility.

## Check compatibility

Compare the manifest and labels with the runtime's expected

decision mode, deployed model, feature schema, and status labels.

Check installed package versions when investigating loading errors.

Distinguish these levels of evidence:

1. Required files are present.

2. Metadata passes compatibility checks.

3. Models load successfully.

4. Predictions behave as expected.

Do not claim the model works based only on file presence

or metadata checks.

## Test predictions

For trusted project model files, use the runtime's actual

prediction interface after inspecting its inputs.

Include representative PR/PO pairs for:

- A matching requirement.

- An offer below a minimum requirement.

- Missing or ambiguous specifications.

- A different offered item.

Record expected and actual results.

A few example predictions do not establish overall accuracy.

## Preserve the existing model

Diagnose before proposing retraining or replacement.

Do not overwrite trained assets or modify the application

database during diagnostic checks.

## Report

Explain the failure, supporting evidence, and proposed fix.

Label bundled evaluation metrics as historical unless

the evaluation was actually rerun.
