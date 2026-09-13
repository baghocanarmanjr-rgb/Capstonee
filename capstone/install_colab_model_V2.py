"""Install a V2 FastText model ZIP exported by the new Google Colab notebook.

Usage:
    python install_colab_model_V2.py FASTTEXT_NLP_MODEL_AI_ONLY.zip
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TARGET = BASE_DIR / "model" / "fasttext_nlp"
EXPECTED_DECISION_MODE = "AI_ONLY_FASTTEXT_DUAL_CLASSIFIER"
EXPECTED_DEPLOYED_MODEL = "FastText + Logistic Regression"
EXPECTED_FEATURE_SCHEMA = "FASTTEXT_NUMERIC_RELATION_V2"
EXPECTED_STATUS_LABELS = ["Acceptable", "Needs Review", "Needs Recanvass", "Reject"]
REQUIRED = [
    Path("fasttext_model.model"),
    Path("status_classifier.joblib"),
    Path("issue_classifier.joblib"),
    Path("labels.json"),
    Path("training_manifest.json"),
    Path("fasttext_feature_schema_v2.py"),
]


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if root not in target.parents and target != root:
            raise RuntimeError(f"Unsafe path in ZIP: {member.filename}")
    archive.extractall(destination)


def locate_root(extracted: Path) -> Path:
    candidates = [extracted]
    candidates.extend(path.parent for path in extracted.rglob("training_manifest.json") if path.is_file())
    for root in candidates:
        if all((root / relative).exists() for relative in REQUIRED):
            return root
    raise RuntimeError("ZIP is missing required V2 files: " + ", ".join(str(path) for path in REQUIRED))


def validate(root: Path) -> None:
    labels = json.loads((root / "labels.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "training_manifest.json").read_text(encoding="utf-8"))
    if labels.get("status_labels") != EXPECTED_STATUS_LABELS:
        raise RuntimeError(f"Unexpected status labels: {labels.get('status_labels')!r}")
    if not isinstance(labels.get("issue_labels"), list) or not labels["issue_labels"]:
        raise RuntimeError("labels.json does not contain issue_labels.")
    if manifest.get("decision_mode") != EXPECTED_DECISION_MODE:
        raise RuntimeError(f"Wrong decision_mode: {manifest.get('decision_mode')!r}")
    if manifest.get("deployed_model") != EXPECTED_DEPLOYED_MODEL:
        raise RuntimeError(f"Wrong deployed_model: {manifest.get('deployed_model')!r}")
    if manifest.get("feature_schema") != EXPECTED_FEATURE_SCHEMA:
        raise RuntimeError(f"Wrong feature_schema: {manifest.get('feature_schema')!r}")
    if manifest.get("quality_gate_passed") is not True:
        raise RuntimeError("The export is not marked as passing the V2 quality gate.")
    if int(manifest.get("feature_dimension", 0)) <= 0:
        raise RuntimeError("training_manifest.json has no valid feature_dimension.")
    if int(manifest.get("status_feature_dimension", 0)) <= int(manifest.get("feature_dimension", 0)):
        raise RuntimeError("training_manifest.json has no valid status_feature_dimension.")
    if manifest.get("status_classifier_uses_issue_probabilities") is not True:
        raise RuntimeError("The export is not marked as using issue probabilities for status prediction.")


def install(zip_path: Path) -> None:
    zip_path = zip_path.expanduser().resolve()
    if not zip_path.exists():
        raise FileNotFoundError(f"Model ZIP was not found: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"Not a valid ZIP archive: {zip_path}")

    with tempfile.TemporaryDirectory(prefix="asuncion_fasttext_v2_") as temp:
        extracted = Path(temp) / "extracted"
        extracted.mkdir()
        with zipfile.ZipFile(zip_path) as archive:
            safe_extract(archive, extracted)
        root = locate_root(extracted)
        validate(root)

        backup = None
        if TARGET.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = TARGET.with_name(f"fasttext_nlp_backup_{stamp}")
            shutil.move(str(TARGET), str(backup))

        TARGET.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root, TARGET)

    print(f"Installed V2 FastText model to: {TARGET}")
    if backup:
        print(f"Previous active model moved to: {backup}")
    print("Next: replace fasttext_nlp_runtime.py with the V2 runtime, verify model_status, then test predictions.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python install_colab_model_V2.py FASTTEXT_NLP_MODEL_AI_ONLY.zip")
    install(Path(sys.argv[1]))
