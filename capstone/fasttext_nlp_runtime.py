"""Offline runtime for the V2 AI-only FastText PR-to-PO model.

The final status and issue are predicted by two trained Logistic Regression
classifiers. Numeric and measurement relationships are classifier inputs only;
this module contains no rule that directly assigns a status or issue, no weighted
hybrid score, no similarity threshold, and no fallback model.
"""
from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict

EXPECTED_DECISION_MODE = "AI_ONLY_FASTTEXT_DUAL_CLASSIFIER"
EXPECTED_DEPLOYED_MODEL = "FastText + Logistic Regression"
EXPECTED_FEATURE_SCHEMA = "FASTTEXT_NUMERIC_RELATION_V2"
EXPECTED_STATUS_LABELS = [
    "Acceptable",
    "Needs Review",
    "Needs Recanvass",
    "Reject",
]


@dataclass
class RuntimeAssets:
    fasttext_model: Any
    status_classifier: Any
    issue_classifier: Any
    status_labels: list[str]
    issue_labels: list[str]
    manifest: Dict[str, Any]
    feature_module: ModuleType


_CACHE: RuntimeAssets | None = None
_CACHE_ROOT: str | None = None


def required_paths(model_root: str) -> Dict[str, str]:
    return {
        "fasttext": os.path.join(model_root, "fasttext_model.model"),
        "status_classifier": os.path.join(model_root, "status_classifier.joblib"),
        "issue_classifier": os.path.join(model_root, "issue_classifier.joblib"),
        "labels": os.path.join(model_root, "labels.json"),
        "manifest": os.path.join(model_root, "training_manifest.json"),
        "feature_module": os.path.join(model_root, "fasttext_feature_schema_v2.py"),
    }


def _load_feature_module(path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("deployed_fasttext_feature_schema_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not create a loader for the deployed feature module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_status(model_root: str) -> Dict[str, Any]:
    paths = required_paths(model_root)
    missing = [
        os.path.relpath(path, model_root)
        for path in paths.values()
        if not os.path.exists(path)
    ]
    if missing:
        return {
            "active": False,
            "message": "V2 Colab-trained FastText model is not installed. Missing: "
            + ", ".join(missing),
            "missing": missing,
        }

    try:
        with open(paths["labels"], "r", encoding="utf-8") as handle:
            labels = json.load(handle)
        with open(paths["manifest"], "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        feature_module = _load_feature_module(paths["feature_module"])

        if labels.get("status_labels") != EXPECTED_STATUS_LABELS:
            return {
                "active": False,
                "message": "Installed status labels do not match the four-class workflow.",
            }
        if manifest.get("decision_mode") != EXPECTED_DECISION_MODE:
            return {
                "active": False,
                "message": "Installed model is not marked as the required AI-only FastText export.",
            }
        if manifest.get("deployed_model") != EXPECTED_DEPLOYED_MODEL:
            return {
                "active": False,
                "message": "Installed package is not the selected FastText deployment model.",
            }
        if manifest.get("feature_schema") != EXPECTED_FEATURE_SCHEMA:
            return {
                "active": False,
                "message": (
                    "Installed model uses an incompatible feature schema: "
                    f"{manifest.get('feature_schema')!r}. Expected {EXPECTED_FEATURE_SCHEMA!r}."
                ),
            }
        if getattr(feature_module, "FEATURE_SCHEMA", None) != EXPECTED_FEATURE_SCHEMA:
            return {
                "active": False,
                "message": "The deployed feature module does not match the V2 feature schema.",
            }
    except Exception as error:
        return {
            "active": False,
            "message": f"Model metadata or feature module cannot be read: {error}",
        }

    return {
        "active": True,
        "message": (
            "V2 AI-only FastText PR-PO model is installed. The trained status "
            "and issue classifiers directly control the outputs."
        ),
        "manifest": manifest,
    }


def clear_cache() -> None:
    global _CACHE, _CACHE_ROOT
    _CACHE = None
    _CACHE_ROOT = None


def load_assets(model_root: str) -> RuntimeAssets:
    global _CACHE, _CACHE_ROOT
    normalized_root = os.path.abspath(model_root)
    if _CACHE is not None and _CACHE_ROOT == normalized_root:
        return _CACHE

    status = model_status(normalized_root)
    if not status.get("active"):
        raise RuntimeError(status.get("message") or "FastText NLP model is unavailable")

    try:
        import joblib
        from gensim.models import FastText
    except ImportError as error:
        raise RuntimeError(
            "FastText runtime dependencies are missing. Install requirements.txt before running NLP verification."
        ) from error

    paths = required_paths(normalized_root)
    with open(paths["labels"], "r", encoding="utf-8") as handle:
        labels = json.load(handle)
    with open(paths["manifest"], "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    feature_module = _load_feature_module(paths["feature_module"])
    fasttext_model = FastText.load(paths["fasttext"])
    status_classifier = joblib.load(paths["status_classifier"])
    issue_classifier = joblib.load(paths["issue_classifier"])

    base_dimension = int(
        feature_module.expected_feature_dimension(int(fasttext_model.vector_size))
    )
    issue_dimension = int(getattr(issue_classifier, "n_features_in_", -1))
    status_dimension = int(getattr(status_classifier, "n_features_in_", -1))
    manifest_base = int(manifest.get("issue_feature_dimension", manifest.get("feature_dimension", -1)))
    manifest_status = int(manifest.get("status_feature_dimension", -1))
    expected_status_dimension = base_dimension + len(labels["issue_labels"])
    dimensions = {
        "runtime_base_expected": base_dimension,
        "manifest_issue": manifest_base,
        "issue_classifier": issue_dimension,
        "runtime_status_expected": expected_status_dimension,
        "manifest_status": manifest_status,
        "status_classifier": status_dimension,
    }
    if not (
        base_dimension == manifest_base == issue_dimension
        and expected_status_dimension == manifest_status == status_dimension
    ):
        raise RuntimeError(
            "Training/runtime feature dimensions do not match: " + json.dumps(dimensions)
        )
    if manifest.get("status_classifier_uses_issue_probabilities") is not True:
        raise RuntimeError("The V2 status classifier is not marked as using issue probabilities.")

    _CACHE = RuntimeAssets(
        fasttext_model=fasttext_model,
        status_classifier=status_classifier,
        issue_classifier=issue_classifier,
        status_labels=[str(value) for value in labels["status_labels"]],
        issue_labels=[str(value) for value in labels["issue_labels"]],
        manifest=manifest,
        feature_module=feature_module,
    )
    _CACHE_ROOT = normalized_root
    return _CACHE


def _probability_map(classifier: Any, probabilities: Any, labels: list[str]) -> Dict[str, float]:
    result = {label: 0.0 for label in labels}
    for class_id, probability in zip(classifier.classes_, probabilities):
        class_index = int(class_id)
        if 0 <= class_index < len(labels):
            result[labels[class_index]] = round(float(probability) * 100.0, 2)
    return result


def predict_pair(model_root: str, pr_text: str, po_text: str) -> Dict[str, Any]:
    """Predict status and issue directly from the trained V2 classifiers."""
    assets = load_assets(model_root)
    base_features = assets.feature_module.build_pair_features(
        assets.fasttext_model,
        pr_text,
        po_text,
    ).reshape(1, -1)

    issue_probabilities_matrix = assets.issue_classifier.predict_proba(base_features)
    issue_id = int(assets.issue_classifier.predict(base_features)[0])
    status_features = __import__("numpy").hstack([base_features, issue_probabilities_matrix])
    status_id = int(assets.status_classifier.predict(status_features)[0])
    status_probabilities_raw = assets.status_classifier.predict_proba(status_features)[0]
    issue_probabilities_raw = issue_probabilities_matrix[0]

    status_probabilities = _probability_map(
        assets.status_classifier,
        status_probabilities_raw,
        assets.status_labels,
    )
    issue_probabilities = _probability_map(
        assets.issue_classifier,
        issue_probabilities_raw,
        assets.issue_labels,
    )
    status_label = assets.status_labels[status_id]
    issue_label = assets.issue_labels[issue_id]

    return {
        "status": status_label,
        "status_confidence": status_probabilities.get(status_label, 0.0),
        "status_probabilities": status_probabilities,
        "issue": issue_label,
        "issue_confidence": issue_probabilities.get(issue_label, 0.0),
        "issue_probabilities": issue_probabilities,
        "model_manifest": assets.manifest,
        "pr_token_count": len(assets.feature_module.tokenize(pr_text)),
        "po_token_count": len(assets.feature_module.tokenize(po_text)),
        "feature_dimension": int(base_features.shape[1]),
        "status_feature_dimension": int(status_features.shape[1]),
        "feature_schema": assets.feature_module.FEATURE_SCHEMA,
    }
