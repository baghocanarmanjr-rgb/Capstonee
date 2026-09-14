PLACE THE VERIFIED V2 FASTTEXT MODEL FILES IN THIS FOLDER.

Required files:
- fasttext_model.model
- fasttext_model.model.wv.vectors_ngrams.npy (when exported separately)
- status_classifier.joblib
- issue_classifier.joblib
- labels.json
- training_manifest.json
- fasttext_feature_schema_v2.py

The system remains runnable through Purchase Order without these files.
PR-PO NLP Verification becomes active only when model_status reports active=true.
