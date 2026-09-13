# ================================================================
# GOOGLE COLAB V2 TRAINING CODE
# Three approaches: TF-IDF, enhanced FastText, frozen DistilBERT
# Deployment: FastText + Logistic Regression with learned numeric
# relationship features and an export quality gate.
# Copy each CELL section into a separate Colab cell.
# ================================================================

# ========================= CELL 1 ================================
# CELL 1 — Install compatible dependencies
# Run this cell once. After it finishes, use Runtime > Restart session,
# then continue from Cell 2. Do not rerun Cell 1 after the restart.
%pip -q install --upgrade \
    "numpy==2.0.2" \
    "pandas==2.2.2" \
    "scipy>=1.13,<2" \
    "gensim==4.4.0" \
    "scikit-learn==1.8.0" \
    "joblib==1.5.3" \
    "openpyxl==3.1.5" \
    "transformers>=4.50,<5" \
    "accelerate>=1,<2"

print("Installation finished.")
print("NOW choose Runtime > Restart session, then run Cell 2.")

# ========================= CELL 2 ================================
# CELL 2 — Imports, configuration, and upload
import os
import re
import json
import time
import math
import random
import hashlib
import shutil
import sys
import importlib
import importlib.metadata
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import torch

from google.colab import files
from IPython.display import display
from gensim.models import FastText
from scipy.sparse import csr_matrix, hstack
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoTokenizer, AutoModel

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("A GPU is recommended for DistilBERT comparison.")

print("Upload the COA workbook and one validated PR-PO CSV.")
uploaded = files.upload()
uploaded_names = list(uploaded.keys())
EXCEL_FILES = [name for name in uploaded_names if name.lower().endswith((".xlsx", ".xlsm"))]
CSV_FILES = [name for name in uploaded_names if name.lower().endswith(".csv")]
EXCEL_PATH = EXCEL_FILES[0] if EXCEL_FILES else None
CSV_PATH = CSV_FILES[0] if CSV_FILES else None
print("COA workbook:", EXCEL_PATH)
print("Validated CSV:", CSV_PATH)

STATUS_LABELS = ["Acceptable", "Needs Review", "Needs Recanvass", "Reject"]
ISSUE_LABELS = [
    "Requirement satisfied",
    "Missing or ambiguous information",
    "Below minimum",
    "Above maximum",
    "Exact or range mismatch",
    "Quantity or unit mismatch",
    "Specification mismatch",
    "Different item",
    "Missing item",
]
STATUS2ID = {label: index for index, label in enumerate(STATUS_LABELS)}
ISSUE2ID = {label: index for index, label in enumerate(ISSUE_LABELS)}

DISTILBERT_MODEL = "distilbert-base-uncased"
FASTTEXT_VECTOR_SIZE = 120
FASTTEXT_BUCKET = 100_000
MAX_LENGTH = 256

OUTPUT_ROOT = Path("/content/fasttext_v2_output")
DEPLOY_DIR = OUTPUT_ROOT / "FASTTEXT_NLP_MODEL_AI_ONLY"
RESULTS_DIR = OUTPUT_ROOT / "THREE_NLP_COMPARISON_RESULTS"
for directory in (DEPLOY_DIR, RESULTS_DIR):
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)

print("Python:", sys.version)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)

# ========================= CELL 3 ================================
# CELL 3 — Create the shared V2 feature module
# This exact module is used during training and is exported with the model.
# The Windows runtime loads the exported copy, preventing training/runtime
# feature mismatches.
FEATURE_MODULE_CODE = '"""Shared feature extractor for FastText PR-to-PO classification.\n\nThe module computes text-vector interaction features and structured numeric/\nmeasurement relationship features. It never outputs a procurement decision.\nThe trained classifiers remain solely responsible for predicting status and issue.\n"""\nfrom __future__ import annotations\n\nimport math\nimport re\nfrom typing import Any, Dict, List, Tuple\n\nimport numpy as np\n\nFEATURE_SCHEMA = "FASTTEXT_NUMERIC_RELATION_V2"\nTOKEN_PATTERN = re.compile(r"\\d+(?:\\.\\d+)?|[a-zA-Z]+(?:[-\'][a-zA-Z]+)*")\nNUMBER_PATTERN = re.compile(r"\\d+(?:\\.\\d+)?")\n\n# number + unit. Longer alternatives come first to avoid partial matches.\nMEASUREMENT_PATTERN = re.compile(\n    r"(?P<value>\\d+(?:\\.\\d+)?)\\s*"\n    r"(?P<unit>"\n    r"terabytes?|tb|gigabytes?|gb|megabytes?|mb|"\n    r"kilograms?|kgs?|kg|grams?|g|"\n    r"kilowatts?|kw|watts?|w|"\n    r"years?|yrs?|yr|months?|mos?|mo|"\n    r"inches?|inch|centimeters?|centimetres?|cm|millimeters?|millimetres?|mm|"\n    r"liters?|litres?|l|milliliters?|millilitres?|ml|"\n    r"ports?|ppm|dpi|gsm|"\n    r"reams?|pieces?|pcs?|units?|bottles?|btls?|cartridges?|sets?"\n    r")\\b",\n    re.IGNORECASE,\n)\n\nRANGE_MEASUREMENT_PATTERN = re.compile(\n    r"(?P<low>\\d+(?:\\.\\d+)?)\\s*(?:to|and|through|-)\\s*"\n    r"(?P<high>\\d+(?:\\.\\d+)?)\\s*"\n    r"(?P<unit>"\n    r"terabytes?|tb|gigabytes?|gb|megabytes?|mb|"\n    r"kilograms?|kgs?|kg|grams?|g|"\n    r"kilowatts?|kw|watts?|w|"\n    r"years?|yrs?|yr|months?|mos?|mo|"\n    r"inches?|inch|centimeters?|centimetres?|cm|millimeters?|millimetres?|mm|"\n    r"liters?|litres?|l|milliliters?|millilitres?|ml|"\n    r"ports?|ppm|dpi|gsm|"\n    r"reams?|pieces?|pcs?|units?|bottles?|btls?|cartridges?|sets?"\n    r")\\b",\n    re.IGNORECASE,\n)\n\nMINIMUM_CUES = (\n    "minimum", "at least", "not less than", "no lower than", "or higher",\n    "or greater", "must be at least", "minimum of",\n)\nMAXIMUM_CUES = (\n    "maximum", "at most", "not more than", "must not exceed", "not exceed",\n    "up to", "no more than", "maximum of",\n)\nEXACT_CUES = (\n    "exactly", "shall be exactly", "required value", "exact quantity",\n    "must be exactly",\n)\nRANGE_CUES = (\n    "between", "from", "within the range", "range of", "not lower than",\n    "not exceeding",\n)\nQUANTITY_CUES = (\n    "quantity", "qty", "number of units", "number of pieces", "ordered quantity",\n)\n\nKEYWORD_GROUPS = {\n    "ssd": ("ssd", "solid-state drive", "solid state drive", "solid-state storage"),\n    "hdd": ("hdd", "hard disk drive", "hard-drive", "hard drive"),\n    "memory": ("ram", "system memory", "installed memory", "memory capacity"),\n    "weight": ("weight", "mass", "unit weight", "equipment weight"),\n    "power": ("power", "watt", "watts", "power draw", "power consumption"),\n    "ports": ("port", "ports", "lan port", "ethernet port", "network port"),\n    "warranty": ("warranty", "service warranty", "warranty period"),\n    "storage": ("storage", "disk capacity", "drive capacity", "ssd", "hdd"),\n}\n\n# canonical_group, multiplier to canonical units\nUNIT_MAP: Dict[str, Tuple[str, float]] = {\n    "terabyte": ("storage_gb", 1024.0), "terabytes": ("storage_gb", 1024.0), "tb": ("storage_gb", 1024.0),\n    "gigabyte": ("storage_gb", 1.0), "gigabytes": ("storage_gb", 1.0), "gb": ("storage_gb", 1.0),\n    "megabyte": ("storage_gb", 1.0 / 1024.0), "megabytes": ("storage_gb", 1.0 / 1024.0), "mb": ("storage_gb", 1.0 / 1024.0),\n    "kilogram": ("mass_kg", 1.0), "kilograms": ("mass_kg", 1.0), "kg": ("mass_kg", 1.0), "kgs": ("mass_kg", 1.0),\n    "gram": ("mass_kg", 0.001), "grams": ("mass_kg", 0.001), "g": ("mass_kg", 0.001),\n    "kilowatt": ("power_w", 1000.0), "kilowatts": ("power_w", 1000.0), "kw": ("power_w", 1000.0),\n    "watt": ("power_w", 1.0), "watts": ("power_w", 1.0), "w": ("power_w", 1.0),\n    "year": ("time_month", 12.0), "years": ("time_month", 12.0), "yr": ("time_month", 12.0), "yrs": ("time_month", 12.0),\n    "month": ("time_month", 1.0), "months": ("time_month", 1.0), "mo": ("time_month", 1.0), "mos": ("time_month", 1.0),\n    "inch": ("length_in", 1.0), "inches": ("length_in", 1.0),\n    "centimeter": ("length_in", 1.0 / 2.54), "centimeters": ("length_in", 1.0 / 2.54),\n    "centimetre": ("length_in", 1.0 / 2.54), "centimetres": ("length_in", 1.0 / 2.54), "cm": ("length_in", 1.0 / 2.54),\n    "millimeter": ("length_in", 1.0 / 25.4), "millimeters": ("length_in", 1.0 / 25.4),\n    "millimetre": ("length_in", 1.0 / 25.4), "millimetres": ("length_in", 1.0 / 25.4), "mm": ("length_in", 1.0 / 25.4),\n    "liter": ("volume_l", 1.0), "liters": ("volume_l", 1.0), "litre": ("volume_l", 1.0), "litres": ("volume_l", 1.0), "l": ("volume_l", 1.0),\n    "milliliter": ("volume_l", 0.001), "milliliters": ("volume_l", 0.001),\n    "millilitre": ("volume_l", 0.001), "millilitres": ("volume_l", 0.001), "ml": ("volume_l", 0.001),\n    "port": ("ports", 1.0), "ports": ("ports", 1.0),\n    "ppm": ("print_speed_ppm", 1.0), "dpi": ("resolution_dpi", 1.0), "gsm": ("paper_gsm", 1.0),\n    "ream": ("qty_ream", 1.0), "reams": ("qty_ream", 1.0),\n    "piece": ("qty_piece", 1.0), "pieces": ("qty_piece", 1.0), "pc": ("qty_piece", 1.0), "pcs": ("qty_piece", 1.0),\n    "unit": ("qty_piece", 1.0), "units": ("qty_piece", 1.0),\n    "bottle": ("qty_bottle", 1.0), "bottles": ("qty_bottle", 1.0), "btl": ("qty_bottle", 1.0), "btls": ("qty_bottle", 1.0),\n    "cartridge": ("qty_cartridge", 1.0), "cartridges": ("qty_cartridge", 1.0),\n    "set": ("qty_set", 1.0), "sets": ("qty_set", 1.0),\n}\n\nNUMERIC_FEATURE_NAMES = [\n    "pr_number_count_scaled", "po_number_count_scaled", "comparable_measurement_found",\n    "pr_numeric_missing", "po_numeric_missing", "same_measurement_group", "unit_conversion_used",\n    "cue_minimum", "cue_maximum", "cue_exact", "cue_range", "cue_unspecified",\n    "log_pr_low", "log_pr_high", "log_po_value",\n    "signed_log_po_minus_low", "signed_log_po_minus_high",\n    "abs_log_po_minus_low", "abs_log_po_minus_high",\n    "ratio_po_to_low_clipped", "ratio_po_to_high_clipped",\n    "po_ge_low", "po_le_low", "po_eq_low", "po_between", "po_below_low", "po_above_high",\n    "pr_quantity_cue", "po_quantity_cue",\n    "pr_keyword_ssd", "po_keyword_ssd", "pr_keyword_hdd", "po_keyword_hdd",\n    "pr_keyword_memory", "po_keyword_memory", "pr_keyword_weight", "po_keyword_weight",\n    "pr_keyword_power", "po_keyword_power", "pr_keyword_ports", "po_keyword_ports",\n    "pr_keyword_warranty", "po_keyword_warranty", "pr_keyword_storage", "po_keyword_storage",\n    "minimum_satisfied_interaction", "minimum_violated_interaction",\n    "maximum_satisfied_interaction", "maximum_violated_interaction",\n    "exact_satisfied_interaction", "exact_violated_interaction",\n    "range_satisfied_interaction", "range_violated_interaction",\n    "required_numeric_missing_interaction",\n    "ssd_confirmed_interaction", "ssd_ambiguous_interaction", "ssd_hdd_mismatch_interaction",\n    "quantity_satisfied_interaction", "quantity_mismatch_interaction",\n]\n\n\ndef normalize_text(text: Any) -> str:\n    return re.sub(r"\\s+", " ", str(text or "")).strip()\n\n\ndef tokenize(text: Any) -> List[str]:\n    return TOKEN_PATTERN.findall(normalize_text(text).lower())\n\n\ndef sentence_vector(model: Any, text: Any) -> np.ndarray:\n    tokens = tokenize(text)\n    if not tokens:\n        return np.zeros(int(model.vector_size), dtype=np.float32)\n    return np.mean([model.wv[token] for token in tokens], axis=0).astype(np.float32)\n\n\ndef _has_cue(text: Any, cues: Tuple[str, ...]) -> float:\n    lowered = normalize_text(text).lower()\n    return float(any(cue in lowered for cue in cues))\n\n\ndef _has_keyword(text: Any, group: str) -> float:\n    lowered = normalize_text(text).lower()\n    return float(any(term in lowered for term in KEYWORD_GROUPS[group]))\n\n\ndef _signed_log(value: float) -> float:\n    return float(math.copysign(math.log1p(abs(value)), value)) if value else 0.0\n\n\ndef _safe_ratio(numerator: float, denominator: float) -> float:\n    if abs(denominator) < 1e-12:\n        return 0.0\n    return float(np.clip(numerator / denominator, 0.0, 10.0) / 10.0)\n\n\ndef extract_measurements(text: Any) -> List[Dict[str, Any]]:\n    result: List[Dict[str, Any]] = []\n    normalized = normalize_text(text)\n    occupied: List[Tuple[int, int]] = []\n\n    # Parse compact ranges such as "14 to 16 inches" where the unit appears once.\n    for match in RANGE_MEASUREMENT_PATTERN.finditer(normalized):\n        raw_unit = match.group("unit").lower()\n        group, multiplier = UNIT_MAP[raw_unit]\n        for key in ("low", "high"):\n            raw_value = float(match.group(key))\n            result.append(\n                {\n                    "raw_value": raw_value,\n                    "raw_unit": raw_unit,\n                    "group": group,\n                    "value": raw_value * multiplier,\n                    "converted": float(abs(multiplier - 1.0) > 1e-12),\n                    "start": match.start(key),\n                    "end": match.end(key),\n                }\n            )\n        occupied.append((match.start(), match.end()))\n\n    for match in MEASUREMENT_PATTERN.finditer(normalized):\n        if any(not (match.end() <= start or match.start() >= end) for start, end in occupied):\n            continue\n        raw_value = float(match.group("value"))\n        raw_unit = match.group("unit").lower()\n        group, multiplier = UNIT_MAP[raw_unit]\n        result.append(\n            {\n                "raw_value": raw_value,\n                "raw_unit": raw_unit,\n                "group": group,\n                "value": raw_value * multiplier,\n                "converted": float(abs(multiplier - 1.0) > 1e-12),\n                "start": match.start(),\n                "end": match.end(),\n            }\n        )\n        occupied.append((match.start(), match.end()))\n\n    # Include untyped numbers only when they were not part of a typed measurement.\n    for match in NUMBER_PATTERN.finditer(normalized):\n        if any(start <= match.start() < end for start, end in occupied):\n            continue\n        result.append(\n            {\n                "raw_value": float(match.group(0)),\n                "raw_unit": "",\n                "group": "raw_number",\n                "value": float(match.group(0)),\n                "converted": 0.0,\n                "start": match.start(),\n                "end": match.end(),\n            }\n        )\n\n    result.sort(key=lambda item: item["start"])\n    return result\n\n\ndef _select_relation(pr_text: Any, po_text: Any) -> Dict[str, float]:\n    pr_values = extract_measurements(pr_text)\n    po_values = extract_measurements(po_text)\n\n    relation = {\n        "found": 0.0,\n        "same_group": 0.0,\n        "converted": 0.0,\n        "pr_low": 0.0,\n        "pr_high": 0.0,\n        "po_value": 0.0,\n    }\n    if not pr_values or not po_values:\n        return relation\n\n    # Prefer a PR measurement group that also occurs in PO.\n    selected_pr = None\n    selected_po = None\n    for pr_item in pr_values:\n        match = next((po_item for po_item in po_values if po_item["group"] == pr_item["group"]), None)\n        if match is not None:\n            selected_pr = pr_item\n            selected_po = match\n            break\n\n    if selected_pr is None:\n        selected_pr = pr_values[0]\n        selected_po = po_values[0]\n    else:\n        relation["same_group"] = 1.0\n\n    same_group_pr = [item for item in pr_values if item["group"] == selected_pr["group"]]\n    pr_low = float(selected_pr["value"])\n    pr_high = pr_low\n    if len(same_group_pr) >= 2:\n        first_two = sorted(float(item["value"]) for item in same_group_pr[:2])\n        pr_low, pr_high = first_two[0], first_two[1]\n\n    relation.update(\n        {\n            "found": 1.0,\n            "converted": float(bool(selected_pr["converted"] or selected_po["converted"])),\n            "pr_low": pr_low,\n            "pr_high": pr_high,\n            "po_value": float(selected_po["value"]),\n        }\n    )\n    return relation\n\n\ndef numeric_relation_features(pr_text: Any, po_text: Any) -> np.ndarray:\n    pr_measurements = extract_measurements(pr_text)\n    po_measurements = extract_measurements(po_text)\n    relation = _select_relation(pr_text, po_text)\n\n    cue_min = _has_cue(pr_text, MINIMUM_CUES)\n    cue_max = _has_cue(pr_text, MAXIMUM_CUES)\n    cue_exact = _has_cue(pr_text, EXACT_CUES)\n    cue_range = _has_cue(pr_text, RANGE_CUES)\n    cue_unspecified = float(not any((cue_min, cue_max, cue_exact, cue_range)))\n\n    low = relation["pr_low"]\n    high = relation["pr_high"]\n    offered = relation["po_value"]\n    diff_low = offered - low if relation["found"] else 0.0\n    diff_high = offered - high if relation["found"] else 0.0\n    tolerance = max(1e-6, abs(low) * 1e-6)\n    found = bool(relation["found"])\n    ge_low = bool(found and offered >= low - tolerance)\n    le_low = bool(found and offered <= low + tolerance)\n    eq_low = bool(found and abs(offered - low) <= tolerance)\n    between = bool(found and low - tolerance <= offered <= high + tolerance)\n    below_low = bool(found and offered < low - tolerance)\n    above_high = bool(found and offered > high + tolerance)\n\n    pr_ssd = bool(_has_keyword(pr_text, "ssd"))\n    po_ssd = bool(_has_keyword(po_text, "ssd"))\n    po_hdd = bool(_has_keyword(po_text, "hdd"))\n    po_storage = bool(_has_keyword(po_text, "storage"))\n    pr_quantity = bool(_has_cue(pr_text, QUANTITY_CUES))\n\n    values = [\n        min(len(pr_measurements), 5) / 5.0,\n        min(len(po_measurements), 5) / 5.0,\n        relation["found"],\n        float(len(pr_measurements) == 0),\n        float(len(po_measurements) == 0),\n        relation["same_group"],\n        relation["converted"],\n        cue_min, cue_max, cue_exact, cue_range, cue_unspecified,\n        math.log1p(abs(low)), math.log1p(abs(high)), math.log1p(abs(offered)),\n        _signed_log(diff_low), _signed_log(diff_high),\n        math.log1p(abs(diff_low)), math.log1p(abs(diff_high)),\n        _safe_ratio(offered, low), _safe_ratio(offered, high),\n        float(ge_low), float(le_low), float(eq_low), float(between), float(below_low), float(above_high),\n        float(pr_quantity), _has_cue(po_text, QUANTITY_CUES),\n        _has_keyword(pr_text, "ssd"), _has_keyword(po_text, "ssd"),\n        _has_keyword(pr_text, "hdd"), _has_keyword(po_text, "hdd"),\n        _has_keyword(pr_text, "memory"), _has_keyword(po_text, "memory"),\n        _has_keyword(pr_text, "weight"), _has_keyword(po_text, "weight"),\n        _has_keyword(pr_text, "power"), _has_keyword(po_text, "power"),\n        _has_keyword(pr_text, "ports"), _has_keyword(po_text, "ports"),\n        _has_keyword(pr_text, "warranty"), _has_keyword(po_text, "warranty"),\n        _has_keyword(pr_text, "storage"), _has_keyword(po_text, "storage"),\n        float(bool(cue_min) and ge_low), float(bool(cue_min) and below_low),\n        float(bool(cue_max) and le_low), float(bool(cue_max) and above_high),\n        float(bool(cue_exact) and eq_low), float(bool(cue_exact) and found and not eq_low),\n        float(bool(cue_range) and between), float(bool(cue_range) and found and not between),\n        float(any((cue_min, cue_max, cue_exact, cue_range)) and len(po_measurements) == 0),\n        float(pr_ssd and po_ssd),\n        float(pr_ssd and po_storage and not po_ssd and not po_hdd),\n        float(pr_ssd and po_hdd),\n        float(pr_quantity and eq_low), float(pr_quantity and found and not eq_low),\n    ]\n    features = np.asarray(values, dtype=np.float32)\n    if features.shape[0] != len(NUMERIC_FEATURE_NAMES):\n        raise RuntimeError(\n            f"Numeric feature schema mismatch: {features.shape[0]} values for "\n            f"{len(NUMERIC_FEATURE_NAMES)} names"\n        )\n    return features\n\n\ndef build_pair_features(model: Any, pr_text: Any, po_text: Any) -> np.ndarray:\n    pr_vector = sentence_vector(model, pr_text)\n    po_vector = sentence_vector(model, po_text)\n    text_features = np.concatenate(\n        [\n            pr_vector,\n            po_vector,\n            po_vector - pr_vector,\n            np.abs(pr_vector - po_vector),\n            pr_vector * po_vector,\n        ]\n    ).astype(np.float32)\n    numeric_features = numeric_relation_features(pr_text, po_text)\n    return np.concatenate([text_features, numeric_features]).astype(np.float32)\n\n\ndef expected_feature_dimension(vector_size: int) -> int:\n    return int(vector_size) * 5 + len(NUMERIC_FEATURE_NAMES)\n'

feature_path = Path("/content/fasttext_feature_schema_v2.py")
feature_path.write_text(FEATURE_MODULE_CODE, encoding="utf-8")

if "/content" not in sys.path:
    sys.path.insert(0, "/content")
import fasttext_feature_schema_v2 as feature_v2
importlib.reload(feature_v2)

print("Feature schema:", feature_v2.FEATURE_SCHEMA)
print("Numeric features:", len(feature_v2.NUMERIC_FEATURE_NAMES))
print("Expected FastText feature dimension:", feature_v2.expected_feature_dimension(FASTTEXT_VECTOR_SIZE))

# Sanity-check the direction and unit-conversion inputs. These are features,
# not final decisions.
for pr, po in [
    ("Laptop with minimum 8 GB RAM", "Laptop with 16 GB system memory"),
    ("Laptop with minimum 8 GB RAM", "Laptop with 4 GB RAM"),
    ("Laptop with minimum 512 GB SSD", "Laptop with 1 TB solid-state drive"),
    ("Weight must not exceed 5 kg", "Weight is 7000 g"),
]:
    print("\nPR:", pr)
    print("PO:", po)
    print("Measurements:", feature_v2.extract_measurements(pr), feature_v2.extract_measurements(po))

# ========================= CELL 4 ================================
# CELL 4 — Read and validate the staff-reviewed CSV
def clean_text(value):
    if value is None:
        return ""
    text = str(value)
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def add_pair(rows, pr_text, po_text, status, issue, group_id, source):
    pr_text = clean_text(pr_text)
    po_text = clean_text(po_text)
    if not pr_text or not po_text:
        return
    if status not in STATUS2ID:
        raise ValueError(f"Unknown status: {status}")
    if issue not in ISSUE2ID:
        raise ValueError(f"Unknown issue: {issue}")
    rows.append({
        "pr_text": pr_text,
        "po_text": po_text,
        "status": status,
        "issue": issue,
        "group_id": str(group_id),
        "source": str(source),
    })

STATUS_ALIASES = {
    "acceptable": "Acceptable",
    "needs review": "Needs Review",
    "needs recanvass": "Needs Recanvass",
    "reject": "Reject",
    "rejected": "Reject",
}
ISSUE_ALIASES = {
    "no discrepancy": "Requirement satisfied",
    "requirement satisfied": "Requirement satisfied",
    "requirements satisfied": "Requirement satisfied",
    "match": "Requirement satisfied",
    "matched": "Requirement satisfied",
    "incomplete or ambiguous specification": "Missing or ambiguous information",
    "ambiguous specification": "Missing or ambiguous information",
    "missing information": "Missing or ambiguous information",
    "missing required specification": "Missing or ambiguous information",
    "equivalence requires technical validation": "Missing or ambiguous information",
    "unclear information": "Missing or ambiguous information",
    "specification below minimum": "Below minimum",
    "below minimum": "Below minimum",
    "specification above maximum": "Above maximum",
    "above maximum": "Above maximum",
    "exact specification mismatch": "Exact or range mismatch",
    "specification outside range": "Exact or range mismatch",
    "range mismatch": "Exact or range mismatch",
    "exact or range mismatch": "Exact or range mismatch",
    "quantity mismatch": "Quantity or unit mismatch",
    "unit mismatch": "Quantity or unit mismatch",
    "quantity or unit mismatch": "Quantity or unit mismatch",
    "storage type mismatch": "Specification mismatch",
    "specification mismatch": "Specification mismatch",
    "different item or model": "Different item",
    "different item": "Different item",
    "unrelated item": "Different item",
    "missing item": "Missing item",
    "no corresponding item": "Missing item",
}


def load_validated_csv(path):
    columns = ["pr_text", "po_text", "status", "issue", "group_id", "source"]
    if not path:
        print("WARNING: No validated CSV was uploaded. Results will be synthetic-only.")
        return pd.DataFrame(columns=columns)

    df = pd.read_csv(path)
    df.columns = [str(column).strip().lower() for column in df.columns]
    aliases = {
        "pr": "pr_text", "purchase_request": "pr_text", "pr_description": "pr_text", "pr_item": "pr_text",
        "po": "po_text", "purchase_order": "po_text", "po_description": "po_text", "po_item": "po_text",
        "result": "status", "classification": "status",
    }
    for old_name, new_name in aliases.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})

    missing = {"pr_text", "po_text", "status", "issue"} - set(df.columns)
    if missing:
        raise ValueError(f"Validated CSV is missing columns: {sorted(missing)}. Detected: {list(df.columns)}")

    for column in ["pr_text", "po_text", "status", "issue"]:
        df[column] = df[column].map(clean_text)
    df["status"] = df["status"].map(lambda value: STATUS_ALIASES.get(value.lower(), value))
    df["issue"] = df["issue"].map(lambda value: ISSUE_ALIASES.get(value.lower(), value))

    invalid_status = sorted(set(df["status"]) - set(STATUS_LABELS))
    invalid_issue = sorted(set(df["issue"]) - set(ISSUE_LABELS))
    if invalid_status or invalid_issue:
        raise ValueError(f"Invalid normalized labels. Status={invalid_status}; Issue={invalid_issue}")

    if "group_id" not in df.columns:
        df["group_id"] = ""
    df["group_id"] = df["group_id"].fillna("").astype(str)
    blanks = df["group_id"].str.strip().eq("")
    df.loc[blanks, "group_id"] = [
        "validated::" + hashlib.sha1((pr + "||" + po).encode("utf-8")).hexdigest()[:12]
        for pr, po in zip(df.loc[blanks, "pr_text"], df.loc[blanks, "po_text"])
    ]
    if "source" not in df.columns:
        df["source"] = "staff_validated"
    df["source"] = df["source"].fillna("staff_validated").astype(str)

    df = df[df["pr_text"].str.len().gt(0) & df["po_text"].str.len().gt(0)]
    return df[columns].drop_duplicates().reset_index(drop=True)

validated_df = load_validated_csv(CSV_PATH)
print("Validated rows:", len(validated_df))
if not validated_df.empty:
    display(validated_df.groupby(["status", "issue", "source"]).size().reset_index(name="rows"))
    staff_count = int((validated_df["source"] == "staff_validated").sum())
    print("Rows explicitly marked staff_validated:", staff_count)
    if staff_count < 50:
        print("WARNING: Fewer than 50 staff-validated rows. Add more real reviewed pairs before claiming real-world accuracy.")

# ========================= CELL 5 ================================
# CELL 5 — Generate a balanced semantic curriculum and workbook augmentation
MIN_TEMPLATES = [
    "{item}; at least {value} {term}",
    "{item}; minimum of {value} {term}",
    "{item}; {value} {term} or higher",
    "{item}; not less than {value} {term}",
    "{item}; {term} must be no lower than {value}",
]
MAX_TEMPLATES = [
    "{item}; at most {value} {term}",
    "{item}; maximum of {value} {term}",
    "{item}; not more than {value} {term}",
    "{item}; {term} must not exceed {value}",
    "{item}; up to {value} {term}",
]
EXACT_TEMPLATES = [
    "{item}; exactly {value} {term}",
    "{item}; {term} must be exactly {value}",
    "{item}; required value for {term}: {value}",
]
RANGE_TEMPLATES = [
    "{item}; {term} between {low} and {high}",
    "{item}; {term} from {low} to {high}",
    "{item}; {term} not lower than {low} and not exceeding {high}",
]
OFFER_TEMPLATES = [
    "{item}; offered {term}: {value}",
    "{item}; provided with {value} {term}",
    "{item}; supplied specification includes {value} {term}",
    "{item}; {term} is {value}",
]

CONCEPTS = [
    {"name": "memory", "item": "Laptop computer", "terms": ["RAM", "system memory", "installed memory"], "values": [4, 6, 8, 12, 16, 24, 32, 48, 64], "group": "storage_gb"},
    {"name": "storage", "item": "Laptop computer", "terms": ["SSD", "solid-state drive", "solid state storage"], "values": [128, 256, 512, 768, 1024, 1536, 2048], "group": "storage_gb"},
    {"name": "weight", "item": "Portable equipment", "terms": ["weight", "equipment weight", "unit weight"], "values": [1, 2, 3, 4, 5, 6, 8, 10], "group": "mass_kg"},
    {"name": "power", "item": "Electrical equipment", "terms": ["power consumption", "power draw", "rated power"], "values": [40, 60, 75, 90, 100, 120, 150, 200], "group": "power_w"},
    {"name": "warranty", "item": "Office equipment", "terms": ["warranty", "service warranty", "warranty period"], "values": [6, 12, 18, 24, 36, 48, 60], "group": "time_month"},
    {"name": "display", "item": "LED monitor", "terms": ["display size", "screen size", "monitor size"], "values": [12, 13, 14, 15.6, 17, 19, 22, 24], "group": "length_in"},
    {"name": "capacity", "item": "Water container", "terms": ["capacity", "holding capacity", "volume capacity"], "values": [1, 2, 5, 10, 20, 30, 50, 100], "group": "volume_l"},
    {"name": "ports", "item": "Network switch", "terms": ["LAN ports", "Ethernet ports", "network ports"], "values": [2, 4, 5, 8, 12, 16, 24, 48], "group": "ports"},
    {"name": "print_speed", "item": "Office printer", "terms": ["print speed", "printing speed", "output speed"], "values": [10, 15, 18, 20, 25, 30, 35, 40], "group": "print_speed_ppm"},
]


def render_value(group, value, rng):
    value = float(value)
    options = []
    if group == "storage_gb":
        options.append(f"{value:g} GB")
        if value >= 1024 and abs(value / 1024 - round(value / 1024, 2)) < 1e-9:
            options.append(f"{value / 1024:g} TB")
    elif group == "mass_kg":
        options.extend([f"{value:g} kg", f"{value * 1000:g} g"])
    elif group == "power_w":
        options.append(f"{value:g} watts")
        if value >= 1000:
            options.append(f"{value / 1000:g} kW")
    elif group == "time_month":
        options.append(f"{value:g} months")
        if value % 12 == 0:
            options.append(f"{value / 12:g} years")
    elif group == "length_in":
        options.extend([f"{value:g} inches", f"{value * 2.54:.1f} cm"])
    elif group == "volume_l":
        options.extend([f"{value:g} liters", f"{value * 1000:g} ml"])
    elif group == "ports":
        options.append(f"{value:g} ports")
    elif group == "print_speed_ppm":
        options.append(f"{value:g} ppm")
    else:
        options.append(f"{value:g}")
    return rng.choice(options)


def mild_noise(text, rng):
    choices = [text, text.upper(), text.lower(), re.sub(r"\s*;\s*", ", ", text)]
    return rng.choice(choices)


def generate_semantic_curriculum(seed=42):
    rng = random.Random(seed)
    rows = []
    for concept in CONCEPTS:
        values = sorted(concept["values"])
        for required in values[1:-1]:
            lower = [value for value in values if value < required]
            higher = [value for value in values if value > required]
            for bundle in range(3):
                term = rng.choice(concept["terms"])
                required_text = render_value(concept["group"], required, rng)
                group = f"semantic::{concept['name']}::min::{required}::{bundle}"
                pr = rng.choice(MIN_TEMPLATES).format(item=concept["item"], value=required_text, term=term)
                good = rng.choice([required] + higher)
                bad = rng.choice(lower)
                po_term = rng.choice(concept["terms"])
                add_pair(rows, pr, mild_noise(rng.choice(OFFER_TEMPLATES).format(item=concept["item"], term=po_term, value=render_value(concept["group"], good, rng)), rng), "Acceptable", "Requirement satisfied", group, "semantic_curriculum")
                add_pair(rows, pr, mild_noise(rng.choice(OFFER_TEMPLATES).format(item=concept["item"], term=po_term, value=render_value(concept["group"], bad, rng)), rng), "Needs Recanvass", "Below minimum", group, "semantic_curriculum")
                add_pair(rows, pr, f"{concept['item']}; {po_term} included but numerical value was not stated", "Needs Review", "Missing or ambiguous information", group, "semantic_curriculum")

            for bundle in range(5):
                term = rng.choice(concept["terms"])
                required_text = render_value(concept["group"], required, rng)
                group = f"semantic::{concept['name']}::max::{required}::{bundle}"
                pr = rng.choice(MAX_TEMPLATES).format(item=concept["item"], value=required_text, term=term)
                good = rng.choice(lower + [required])
                bad = rng.choice(higher)
                po_term = rng.choice(concept["terms"])
                add_pair(rows, pr, rng.choice(OFFER_TEMPLATES).format(item=concept["item"], term=po_term, value=render_value(concept["group"], good, rng)), "Acceptable", "Requirement satisfied", group, "semantic_curriculum")
                add_pair(rows, pr, rng.choice(OFFER_TEMPLATES).format(item=concept["item"], term=po_term, value=render_value(concept["group"], bad, rng)), "Needs Recanvass", "Above maximum", group, "semantic_curriculum")
                add_pair(rows, pr, f"{concept['item']}; {po_term} provided without the numerical limit", "Needs Review", "Missing or ambiguous information", group, "semantic_curriculum")

        for required in values:
            for bundle in range(3):
                group = f"semantic::{concept['name']}::exact::{required}::{bundle}"
                term = rng.choice(concept["terms"])
                pr = rng.choice(EXACT_TEMPLATES).format(item=concept["item"], value=render_value(concept["group"], required, rng), term=term)
                other = rng.choice([value for value in values if value != required])
                add_pair(rows, pr, rng.choice(OFFER_TEMPLATES).format(item=concept["item"], term=rng.choice(concept["terms"]), value=render_value(concept["group"], required, rng)), "Acceptable", "Requirement satisfied", group, "semantic_curriculum")
                add_pair(rows, pr, rng.choice(OFFER_TEMPLATES).format(item=concept["item"], term=rng.choice(concept["terms"]), value=render_value(concept["group"], other, rng)), "Needs Recanvass", "Exact or range mismatch", group, "semantic_curriculum")

        for index in range(len(values) - 2):
            low, middle, high = values[index:index + 3]
            outside = [value for value in values if value < low or value > high]
            for bundle in range(3):
                group = f"semantic::{concept['name']}::range::{low}::{high}::{bundle}"
                term = rng.choice(concept["terms"])
                pr = rng.choice(RANGE_TEMPLATES).format(item=concept["item"], term=term, low=render_value(concept["group"], low, rng), high=render_value(concept["group"], high, rng))
                add_pair(rows, pr, rng.choice(OFFER_TEMPLATES).format(item=concept["item"], term=rng.choice(concept["terms"]), value=render_value(concept["group"], middle, rng)), "Acceptable", "Requirement satisfied", group, "semantic_curriculum")
                add_pair(rows, pr, rng.choice(OFFER_TEMPLATES).format(item=concept["item"], term=rng.choice(concept["terms"]), value=render_value(concept["group"], rng.choice(outside), rng)), "Needs Recanvass", "Exact or range mismatch", group, "semantic_curriculum")

    # Ambiguous and specification-type cases.
    ambiguity_pairs = [
        ("Laptop computer; minimum 512 GB SSD", "Laptop computer; 512 GB storage"),
        ("Laptop computer; Intel Core i5 or technically equivalent", "Laptop computer; AMD Ryzen 5 processor"),
        ("Office chair; stainless-steel frame", "Office chair; durable metal frame"),
        ("Office printer; Ethernet interface", "Office printer; network connectivity"),
    ]
    mismatch_pairs = [
        ("Laptop computer; minimum 512 GB SSD", "Laptop computer; 1 TB hard disk drive"),
        ("Ink cartridge; color magenta", "Ink cartridge; color cyan"),
        ("Office printer; Ethernet interface", "Office printer; USB only"),
        ("Office chair; stainless-steel frame", "Office chair; plastic frame"),
        ("Bond paper; A4 70 gsm", "Bond paper; legal size 80 gsm"),
    ]
    for index, (pr, po) in enumerate(ambiguity_pairs):
        for bundle in range(10):
            add_pair(rows, mild_noise(pr, rng), mild_noise(po, rng), "Needs Review", "Missing or ambiguous information", f"ambiguous::{index}::{bundle}", "semantic_curriculum")
    for index, (pr, po) in enumerate(mismatch_pairs):
        for bundle in range(10):
            add_pair(rows, mild_noise(pr, rng), mild_noise(po, rng), "Needs Recanvass", "Specification mismatch", f"mismatch::{index}::{bundle}", "semantic_curriculum")

    # Systematic storage-type curriculum: equal or higher capacity is not enough
    # when the PR requires SSD but the PO only says generic storage.
    for value in [256, 384, 768, 1024, 1536]:
        for bundle in range(8):
            group = f"storage_type::{value}::{bundle}"
            required = render_value("storage_gb", value, rng)
            higher = render_value("storage_gb", value * 2, rng)
            pr = rng.choice([
                f"Laptop computer; minimum {required} SSD",
                f"Laptop computer; at least {required} solid-state drive",
                f"Laptop computer; {required} solid state storage or higher",
            ])
            add_pair(rows, pr, f"Laptop computer; {higher} SSD", "Acceptable", "Requirement satisfied", group, "semantic_curriculum")
            add_pair(rows, pr, f"Laptop computer; {required} storage", "Needs Review", "Missing or ambiguous information", group, "semantic_curriculum")
            add_pair(rows, pr, f"Laptop computer; {higher} hard disk drive", "Needs Recanvass", "Specification mismatch", group, "semantic_curriculum")

    item_units = [
        ("Laptop computer", ["unit", "units", "pc", "pcs"]),
        ("Ink bottle", ["bottle", "bottles", "btl", "btls"]),
        ("Bond paper", ["ream", "reams"]),
        ("Office chair", ["piece", "pieces", "pc", "pcs"]),
        ("First-aid kit", ["set", "sets"]),
        ("Printer toner", ["cartridge", "cartridges"]),
    ]
    for item_index, (item, units) in enumerate(item_units):
        for quantity in [1, 2, 3, 5, 10, 20, 50]:
            for bundle in range(3):
                group = f"quantity::{item_index}::{quantity}::{bundle}"
                pr_unit, po_unit = rng.choice(units), rng.choice(units)
                add_pair(rows, f"{item}; quantity: {quantity} {pr_unit}", f"{item}; offered quantity: {quantity} {po_unit}", "Acceptable", "Requirement satisfied", group, "semantic_curriculum")
                add_pair(rows, f"{item}; exact quantity: {quantity} {pr_unit}", f"{item}; offered quantity: {quantity + rng.choice([1, 2, 5])} {po_unit}", "Needs Recanvass", "Quantity or unit mismatch", group, "semantic_curriculum")
                add_pair(rows, f"{item}; quantity: {quantity} {pr_unit}", f"{item}; quantity not indicated", "Needs Review", "Missing or ambiguous information", group, "semantic_curriculum")

    item_names = [
        "Laptop computer", "Desktop computer", "Office chair", "Bond paper",
        "Printer toner cartridge", "Network switch", "LED monitor",
        "External hard drive", "Steel filing cabinet", "Laboratory beaker",
        "Water pump", "First-aid kit", "Projector", "Barcode scanner",
    ]
    for item_index, required_item in enumerate(item_names):
        alternatives = [item for item in item_names if item != required_item]
        for bundle in range(8):
            group = f"identity::{item_index}::{bundle}"
            add_pair(rows, f"Required item: {required_item}", f"Offered item: {rng.choice(alternatives)}", "Reject", "Different item", group, "semantic_curriculum")
            missing_templates = [
                "No corresponding offered item was provided",
                f"The Purchase Order omits the requested {required_item}",
                f"No offered line matches the requested {required_item}",
                f"The requested {required_item} has no corresponding PO item",
            ]
            add_pair(rows, f"Required item: {required_item}", rng.choice(missing_templates), "Reject", "Missing item", group, "semantic_curriculum")

    return pd.DataFrame(rows).drop_duplicates(subset=["pr_text", "po_text", "status", "issue"]).reset_index(drop=True)

DESCRIPTION_KEYWORDS = [
    "item description", "description", "property description", "particulars",
    "article", "item", "equipment", "supply", "specification", "specifications",
]

def normalize_header(value):
    return re.sub(r"[^a-z0-9]+", " ", clean_text(value).lower()).strip()

def detect_header_row(raw_df, maximum_rows=30):
    best_index, best_score = 0, -1
    for row_index in range(min(maximum_rows, len(raw_df))):
        values = [normalize_header(value) for value in raw_df.iloc[row_index].tolist()]
        score = sum(2 for value in values for keyword in DESCRIPTION_KEYWORDS if keyword in value)
        if score > best_score:
            best_index, best_score = row_index, score
    return best_index

def extract_workbook_items(excel_path):
    if not excel_path:
        return pd.DataFrame(columns=["item_text", "sheet"])
    records = []
    workbook = pd.ExcelFile(excel_path)
    print("Worksheets:", workbook.sheet_names)
    for sheet in workbook.sheet_names:
        raw = pd.read_excel(excel_path, sheet_name=sheet, header=None)
        if raw.empty:
            continue
        header_row = detect_header_row(raw)
        df = pd.read_excel(excel_path, sheet_name=sheet, header=header_row).dropna(how="all")
        text_columns = [column for column in df.columns if any(keyword in normalize_header(column) for keyword in DESCRIPTION_KEYWORDS)][:5]
        if not text_columns:
            continue
        for _, row in df.iterrows():
            parts = [clean_text(row.get(column)) for column in text_columns]
            text = "; ".join(dict.fromkeys(part for part in parts if part))
            if 4 <= len(text) <= 500:
                records.append({"item_text": text, "sheet": sheet})
    result = pd.DataFrame(records)
    if result.empty:
        return result
    result["normalized"] = result["item_text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    return result.drop_duplicates("normalized").reset_index(drop=True)

def workbook_augmentation(items_df, seed=42, maximum_items=150):
    columns = ["pr_text", "po_text", "status", "issue", "group_id", "source"]
    if items_df.empty:
        return pd.DataFrame(columns=columns)
    rng = random.Random(seed)
    rows = []
    sampled = items_df.sample(n=min(maximum_items, len(items_df)), random_state=seed)
    texts = sampled["item_text"].tolist()
    for item in texts:
        item_hash = hashlib.sha1(item.encode("utf-8")).hexdigest()[:12]
        group = f"coa::{item_hash}"
        for variant in [item, item.upper(), re.sub(r"\s*;\s*", ", ", item)]:
            add_pair(rows, f"Required item: {item}", f"Offered item: {variant}", "Acceptable", "Requirement satisfied", group, "coa_workbook")
        sections = [part.strip() for part in re.split(r"[,;:/\-]\s*", item) if part.strip()]
        incomplete = sections[0] if len(sections) > 1 else " ".join(item.split()[:max(2, len(item.split()) // 2)])
        if incomplete.lower() != item.lower():
            add_pair(rows, f"Required item: {item}", f"Offered item: {incomplete}", "Needs Review", "Missing or ambiguous information", group, "coa_workbook")
        candidates = [candidate for candidate in texts if candidate != item]
        for _ in range(min(2, len(candidates))):
            add_pair(rows, f"Required item: {item}", f"Offered item: {rng.choice(candidates)}", "Reject", "Different item", group, "coa_workbook")
    return pd.DataFrame(rows).drop_duplicates(subset=["pr_text", "po_text", "status", "issue"]).reset_index(drop=True)

semantic_df = generate_semantic_curriculum(SEED)
workbook_items_df = extract_workbook_items(EXCEL_PATH)
workbook_df = workbook_augmentation(workbook_items_df, SEED, maximum_items=150)
print("Semantic rows:", len(semantic_df))
print("Unique workbook items:", len(workbook_items_df))
print("Workbook augmentation rows:", len(workbook_df))

# ========================= CELL 6 ================================
# CELL 6 — Build leakage-resistant splits and sample weights
CHALLENGE_SET = [
    ("Laptop with at least 8 GB RAM", "Laptop with 16 GB system memory", "Acceptable", "Requirement satisfied"),
    ("Laptop with minimum 8 GB memory", "Laptop with 4 GB RAM", "Needs Recanvass", "Below minimum"),
    ("Laptop with minimum 12 GB installed memory", "Laptop with 24 GB RAM", "Acceptable", "Requirement satisfied"),
    ("Laptop with minimum 12 GB installed memory", "Laptop with 6 GB RAM", "Needs Recanvass", "Below minimum"),
    ("Laptop with minimum 512 GB SSD", "Laptop with 1 TB solid-state drive", "Acceptable", "Requirement satisfied"),
    ("Laptop with minimum 512 GB SSD", "Laptop with 512 GB storage", "Needs Review", "Missing or ambiguous information"),
    ("Laptop with minimum 512 GB SSD", "Laptop with 1 TB hard disk drive", "Needs Recanvass", "Specification mismatch"),
    ("Equipment weight must not exceed 5 kg", "Equipment weight is 3 kg", "Acceptable", "Requirement satisfied"),
    ("Equipment weight must not exceed 5 kg", "Equipment weight is 7000 g", "Needs Recanvass", "Above maximum"),
    ("Power consumption must not exceed 100 watts", "Power draw is 75 watts", "Acceptable", "Requirement satisfied"),
    ("Power consumption must not exceed 100 watts", "Power draw is 150 watts", "Needs Recanvass", "Above maximum"),
    ("Network switch with exactly 8 LAN ports", "Network switch with 8 Ethernet ports", "Acceptable", "Requirement satisfied"),
    ("Network switch with exactly 8 LAN ports", "Network switch with 16 Ethernet ports", "Needs Recanvass", "Exact or range mismatch"),
    ("Display size between 14 and 16 inches", "Display size is 15.6 inches", "Acceptable", "Requirement satisfied"),
    ("Display size between 14 and 16 inches", "Display size is 17 inches", "Needs Recanvass", "Exact or range mismatch"),
    ("Office equipment with at least 2 years warranty", "Warranty period is 36 months", "Acceptable", "Requirement satisfied"),
    ("Office equipment with at least 2 years warranty", "Warranty period is 12 months", "Needs Recanvass", "Below minimum"),
    ("Bond paper; exact quantity 10 reams", "Bond paper; offered quantity 10 reams", "Acceptable", "Requirement satisfied"),
    ("Bond paper; exact quantity 10 reams", "Bond paper; offered quantity 5 reams", "Needs Recanvass", "Quantity or unit mismatch"),
    ("Required item: printer toner cartridge", "Offered item: office chair", "Reject", "Different item"),
    ("Required item: water pump", "No corresponding offered item was provided", "Reject", "Missing item"),
]

def pair_key(pr, po):
    return clean_text(pr).lower() + " || " + clean_text(po).lower()

challenge_keys = {pair_key(pr, po) for pr, po, _, _ in CHALLENGE_SET}
full_df = pd.concat([semantic_df, workbook_df, validated_df], ignore_index=True)
full_df = full_df.drop_duplicates(subset=["pr_text", "po_text", "status", "issue"]).reset_index(drop=True)

# Detect exact contradictory labels before any split.
full_df["pair_key"] = [pair_key(pr, po) for pr, po in zip(full_df["pr_text"], full_df["po_text"])]
conflicts = (
    full_df.groupby("pair_key")
    .agg(status_count=("status", "nunique"), issue_count=("issue", "nunique"),
         statuses=("status", lambda values: sorted(set(values))),
         issues=("issue", lambda values: sorted(set(values))))
    .reset_index()
)
conflicts = conflicts[(conflicts["status_count"] > 1) | (conflicts["issue_count"] > 1)]
if not conflicts.empty:
    display(conflicts.head(50))
    raise RuntimeError("Contradictory labels detected. Correct them before training.")

# The challenge set must not be memorized as exact training pairs.
removed_challenge_duplicates = int(full_df["pair_key"].isin(challenge_keys).sum())
full_df = full_df[~full_df["pair_key"].isin(challenge_keys)].drop(columns=["pair_key"]).reset_index(drop=True)
print("Exact challenge duplicates removed from training data:", removed_challenge_duplicates)

full_df["status_id"] = full_df["status"].map(STATUS2ID).astype(int)
full_df["issue_id"] = full_df["issue"].map(ISSUE2ID).astype(int)
print("Total rows:", len(full_df))
print("Status distribution:\n", full_df["status"].value_counts())
print("Issue distribution:\n", full_df["issue"].value_counts())
print("Sources:\n", full_df["source"].value_counts())


def group_split_with_all_labels(df, test_size, seed, attempts=1000):
    expected_status = set(df["status_id"].unique())
    expected_issue = set(df["issue_id"].unique())
    for attempt in range(attempts):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + attempt)
        left_indices, right_indices = next(splitter.split(df, groups=df["group_id"]))
        left, right = df.iloc[left_indices], df.iloc[right_indices]
        if (set(left["status_id"].unique()) == expected_status and
            set(right["status_id"].unique()) == expected_status and
            set(left["issue_id"].unique()) == expected_issue and
            set(right["issue_id"].unique()) == expected_issue):
            return left.reset_index(drop=True), right.reset_index(drop=True)
    raise RuntimeError("Could not create a group split containing all classes. Add more independent rare-class examples.")

train_df, temporary_df = group_split_with_all_labels(full_df, 0.20, SEED)
validation_df, test_df = group_split_with_all_labels(temporary_df, 0.50, SEED + 1000)

for name, frame in [("Train", train_df), ("Validation", validation_df), ("Test", test_df)]:
    print(name, len(frame), frame["status"].value_counts().to_dict())

assert set(train_df["group_id"]).isdisjoint(set(validation_df["group_id"]))
assert set(train_df["group_id"]).isdisjoint(set(test_df["group_id"]))
assert set(validation_df["group_id"]).isdisjoint(set(test_df["group_id"]))

SOURCE_WEIGHTS = {
    "staff_validated": 8.0,
    "curated_seed_requires_staff_review": 2.5,
    "semantic_curriculum": 1.5,
    "coa_workbook": 0.5,
}
train_weights = train_df["source"].map(SOURCE_WEIGHTS).fillna(1.0).to_numpy(dtype=np.float64)
display(pd.DataFrame({"source": train_df["source"], "weight": train_weights}).groupby("source")["weight"].agg(["count", "mean"]))

for name, frame in [("train", train_df), ("validation", validation_df), ("test", test_df)]:
    frame.to_csv(RESULTS_DIR / f"{name}_pairs.csv", index=False)

# ========================= CELL 7 ================================
# CELL 7 — Train TF-IDF and enhanced FastText
def classification_metrics(true_ids, predicted_ids, labels):
    _, _, macro_f1, _ = precision_recall_fscore_support(true_ids, predicted_ids, average="macro", zero_division=0)
    _, _, weighted_f1, _ = precision_recall_fscore_support(true_ids, predicted_ids, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(true_ids, predicted_ids)),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "report": classification_report(true_ids, predicted_ids, labels=list(range(len(labels))), target_names=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(true_ids, predicted_ids, labels=list(range(len(labels))).tolist() if False else list(range(len(labels)))).tolist(),
    }


def tune_logistic_regression(train_x, train_y, validation_x, validation_y, sample_weight, solver="lbfgs"):
    best_model, best_score, best_c = None, -1.0, None
    for c_value in [0.25, 0.5, 1.0, 2.0, 4.0]:
        candidate = LogisticRegression(
            C=c_value,
            max_iter=5000,
            class_weight="balanced",
            solver=solver,
            tol=1e-3,
            random_state=SEED,
        )
        candidate.fit(train_x, train_y, sample_weight=sample_weight)
        predictions = candidate.predict(validation_x)
        _, _, score, _ = precision_recall_fscore_support(validation_y, predictions, average="macro", zero_division=0)
        if score > best_score:
            best_model, best_score, best_c = candidate, float(score), c_value
    return best_model, best_c, best_score


def pair_text(frame):
    return ("[PR] " + frame["pr_text"].astype(str) + " [PO] " + frame["po_text"].astype(str)).tolist()


def numeric_matrix(frame):
    return np.vstack([
        feature_v2.numeric_relation_features(pr, po)
        for pr, po in zip(frame["pr_text"], frame["po_text"])
    ]).astype(np.float32)

numeric_train = numeric_matrix(train_df)
numeric_validation = numeric_matrix(validation_df)
numeric_test = numeric_matrix(test_df)

print("Training TF-IDF...")
tfidf_vectorizer = FeatureUnion([
    ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=30000)),
    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=30000)),
])
tfidf_train_text = tfidf_vectorizer.fit_transform(pair_text(train_df))
tfidf_validation_text = tfidf_vectorizer.transform(pair_text(validation_df))
tfidf_test_text = tfidf_vectorizer.transform(pair_text(test_df))
tfidf_train_x = hstack([tfidf_train_text, csr_matrix(numeric_train)], format="csr")
tfidf_validation_x = hstack([tfidf_validation_text, csr_matrix(numeric_validation)], format="csr")
tfidf_test_x = hstack([tfidf_test_text, csr_matrix(numeric_test)], format="csr")

# Train issue first. The trained status classifier then receives the issue
# probability vector as additional learned input. This prevents contradictory
# outputs without hardcoding a status-to-issue rule.
tfidf_issue_model, tfidf_issue_c, _ = tune_logistic_regression(tfidf_train_x, train_df["issue_id"], tfidf_validation_x, validation_df["issue_id"], train_weights, solver="saga")
tfidf_train_issue_prob = tfidf_issue_model.predict_proba(tfidf_train_x)
tfidf_validation_issue_prob = tfidf_issue_model.predict_proba(tfidf_validation_x)
tfidf_test_issue_prob = tfidf_issue_model.predict_proba(tfidf_test_x)
tfidf_status_train_x = hstack([tfidf_train_x, csr_matrix(tfidf_train_issue_prob)], format="csr")
tfidf_status_validation_x = hstack([tfidf_validation_x, csr_matrix(tfidf_validation_issue_prob)], format="csr")
tfidf_status_test_x = hstack([tfidf_test_x, csr_matrix(tfidf_test_issue_prob)], format="csr")
tfidf_status_model, tfidf_status_c, _ = tune_logistic_regression(tfidf_status_train_x, train_df["status_id"], tfidf_status_validation_x, validation_df["status_id"], train_weights, solver="saga")
tfidf_issue_predictions = tfidf_issue_model.predict(tfidf_test_x)
tfidf_status_predictions = tfidf_status_model.predict(tfidf_status_test_x)

print("Training FastText with a reduced n-gram bucket to keep the deployment package manageable...")
fasttext_corpus = [
    feature_v2.tokenize(pr) + ["<PAIR>"] + feature_v2.tokenize(po)
    for pr, po in zip(train_df["pr_text"], train_df["po_text"])
]
fasttext_model = FastText(
    vector_size=FASTTEXT_VECTOR_SIZE,
    window=5,
    min_count=1,
    workers=2,
    sg=1,
    min_n=3,
    max_n=6,
    bucket=FASTTEXT_BUCKET,
    seed=SEED,
)
fasttext_model.build_vocab(corpus_iterable=fasttext_corpus)
fasttext_model.train(corpus_iterable=fasttext_corpus, total_examples=len(fasttext_corpus), epochs=35)


def fasttext_matrix(frame):
    return np.vstack([
        feature_v2.build_pair_features(fasttext_model, pr, po)
        for pr, po in zip(frame["pr_text"], frame["po_text"])
    ]).astype(np.float32)

fasttext_train_x = fasttext_matrix(train_df)
fasttext_validation_x = fasttext_matrix(validation_df)
fasttext_test_x = fasttext_matrix(test_df)
expected_dimension = feature_v2.expected_feature_dimension(FASTTEXT_VECTOR_SIZE)
assert fasttext_train_x.shape[1] == expected_dimension
print("FastText feature dimension:", expected_dimension)

fasttext_issue_model, fasttext_issue_c, _ = tune_logistic_regression(fasttext_train_x, train_df["issue_id"], fasttext_validation_x, validation_df["issue_id"], train_weights)
fasttext_train_issue_prob = fasttext_issue_model.predict_proba(fasttext_train_x)
fasttext_validation_issue_prob = fasttext_issue_model.predict_proba(fasttext_validation_x)
fasttext_test_issue_prob = fasttext_issue_model.predict_proba(fasttext_test_x)
fasttext_status_train_x = np.hstack([fasttext_train_x, fasttext_train_issue_prob])
fasttext_status_validation_x = np.hstack([fasttext_validation_x, fasttext_validation_issue_prob])
fasttext_status_test_x = np.hstack([fasttext_test_x, fasttext_test_issue_prob])
fasttext_status_model, fasttext_status_c, _ = tune_logistic_regression(fasttext_status_train_x, train_df["status_id"], fasttext_status_validation_x, validation_df["status_id"], train_weights)
fasttext_issue_predictions = fasttext_issue_model.predict(fasttext_test_x)
fasttext_status_predictions = fasttext_status_model.predict(fasttext_status_test_x)
print("TF-IDF and FastText training complete.")

# ========================= CELL 8 ================================
# CELL 8 — Train frozen DistilBERT and compare all three approaches
print("Extracting frozen DistilBERT embeddings...")
distilbert_tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL)
distilbert_model = AutoModel.from_pretrained(DISTILBERT_MODEL).to(DEVICE).eval()


def distilbert_embeddings(frame, batch_size=32):
    embeddings = []
    pr_values = frame["pr_text"].astype(str).tolist()
    po_values = frame["po_text"].astype(str).tolist()
    for start in range(0, len(frame), batch_size):
        encoded = distilbert_tokenizer(
            pr_values[start:start + batch_size],
            po_values[start:start + batch_size],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
        with torch.inference_mode():
            output = distilbert_model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        embeddings.append(pooled.cpu().numpy().astype(np.float32))
    return np.vstack(embeddings)

distilbert_train_x = np.hstack([distilbert_embeddings(train_df), numeric_train])
distilbert_validation_x = np.hstack([distilbert_embeddings(validation_df), numeric_validation])
distilbert_test_x = np.hstack([distilbert_embeddings(test_df), numeric_test])
distilbert_issue_model, distilbert_issue_c, _ = tune_logistic_regression(distilbert_train_x, train_df["issue_id"], distilbert_validation_x, validation_df["issue_id"], train_weights)
distilbert_train_issue_prob = distilbert_issue_model.predict_proba(distilbert_train_x)
distilbert_validation_issue_prob = distilbert_issue_model.predict_proba(distilbert_validation_x)
distilbert_test_issue_prob = distilbert_issue_model.predict_proba(distilbert_test_x)
distilbert_status_train_x = np.hstack([distilbert_train_x, distilbert_train_issue_prob])
distilbert_status_validation_x = np.hstack([distilbert_validation_x, distilbert_validation_issue_prob])
distilbert_status_test_x = np.hstack([distilbert_test_x, distilbert_test_issue_prob])
distilbert_status_model, distilbert_status_c, _ = tune_logistic_regression(distilbert_status_train_x, train_df["status_id"], distilbert_status_validation_x, validation_df["status_id"], train_weights)
distilbert_issue_predictions = distilbert_issue_model.predict(distilbert_test_x)
distilbert_status_predictions = distilbert_status_model.predict(distilbert_status_test_x)


def challenge_feature(model_name, pr, po):
    numeric = feature_v2.numeric_relation_features(pr, po).reshape(1, -1)
    if model_name == "TF-IDF + Logistic Regression":
        text = tfidf_vectorizer.transform([f"[PR] {pr} [PO] {po}"])
        return hstack([text, csr_matrix(numeric)], format="csr")
    if model_name == "FastText + Logistic Regression":
        return feature_v2.build_pair_features(fasttext_model, pr, po).reshape(1, -1)
    temporary = pd.DataFrame({"pr_text": [pr], "po_text": [po]})
    return np.hstack([distilbert_embeddings(temporary, batch_size=1), numeric])


def challenge_predictions(model_name, status_model, issue_model):
    details = []
    expected_status_ids, predicted_status_ids = [], []
    expected_issue_ids, predicted_issue_ids = [], []
    for index, (pr, po, expected_status, expected_issue) in enumerate(CHALLENGE_SET, start=1):
        feature = challenge_feature(model_name, pr, po)
        issue_probability = issue_model.predict_proba(feature)
        issue_id = int(issue_model.predict(feature)[0])
        if hasattr(feature, "tocsr"):
            status_feature = hstack([feature, csr_matrix(issue_probability)], format="csr")
        else:
            status_feature = np.hstack([feature, issue_probability])
        status_id = int(status_model.predict(status_feature)[0])
        predicted_status = STATUS_LABELS[status_id]
        predicted_issue = ISSUE_LABELS[issue_id]
        expected_status_ids.append(STATUS2ID[expected_status])
        predicted_status_ids.append(status_id)
        expected_issue_ids.append(ISSUE2ID[expected_issue])
        predicted_issue_ids.append(issue_id)
        details.append({
            "case": index, "pr_text": pr, "po_text": po,
            "expected_status": expected_status, "predicted_status": predicted_status,
            "status_correct": predicted_status == expected_status,
            "expected_issue": expected_issue, "predicted_issue": predicted_issue,
            "issue_correct": predicted_issue == expected_issue,
        })
    return {
        "status_accuracy": float(accuracy_score(expected_status_ids, predicted_status_ids)),
        "issue_accuracy": float(accuracy_score(expected_issue_ids, predicted_issue_ids)),
        "details": details,
    }


def false_acceptable_rate(true_ids, predicted_ids):
    true_ids = np.asarray(true_ids)
    predicted_ids = np.asarray(predicted_ids)
    non_acceptable = true_ids != STATUS2ID["Acceptable"]
    return 0.0 if not non_acceptable.any() else float(np.mean(predicted_ids[non_acceptable] == STATUS2ID["Acceptable"]))


def critical_recall(true_ids, predicted_ids):
    true_ids = np.asarray(true_ids)
    predicted_ids = np.asarray(predicted_ids)
    critical = np.isin(true_ids, [STATUS2ID["Needs Recanvass"], STATUS2ID["Reject"]])
    return 0.0 if not critical.any() else float(np.mean(np.isin(predicted_ids[critical], [STATUS2ID["Needs Recanvass"], STATUS2ID["Reject"]])))

model_outputs = {
    "TF-IDF + Logistic Regression": (tfidf_status_model, tfidf_issue_model, tfidf_status_predictions, tfidf_issue_predictions),
    "FastText + Logistic Regression": (fasttext_status_model, fasttext_issue_model, fasttext_status_predictions, fasttext_issue_predictions),
    "DistilBERT Embeddings + Logistic Regression": (distilbert_status_model, distilbert_issue_model, distilbert_status_predictions, distilbert_issue_predictions),
}
comparison_rows = []
reports = {}
for model_name, (status_model, issue_model, status_predictions, issue_predictions) in model_outputs.items():
    status_metrics = classification_metrics(test_df["status_id"], status_predictions, STATUS_LABELS)
    issue_metrics = classification_metrics(test_df["issue_id"], issue_predictions, ISSUE_LABELS)
    challenge = challenge_predictions(model_name, status_model, issue_model)
    comparison_rows.append({
        "model": model_name,
        "status_accuracy": status_metrics["accuracy"],
        "status_macro_f1": status_metrics["macro_f1"],
        "status_weighted_f1": status_metrics["weighted_f1"],
        "issue_accuracy": issue_metrics["accuracy"],
        "issue_macro_f1": issue_metrics["macro_f1"],
        "issue_weighted_f1": issue_metrics["weighted_f1"],
        "false_acceptable_rate": false_acceptable_rate(test_df["status_id"], status_predictions),
        "critical_recall": critical_recall(test_df["status_id"], status_predictions),
        "challenge_status_accuracy": challenge["status_accuracy"],
        "challenge_issue_accuracy": challenge["issue_accuracy"],
    })
    reports[model_name] = {"status": status_metrics, "issue": issue_metrics, "challenge": challenge}

comparison_df = pd.DataFrame(comparison_rows).sort_values(
    by=["status_macro_f1", "false_acceptable_rate", "critical_recall", "challenge_status_accuracy", "challenge_issue_accuracy"],
    ascending=[False, True, False, False, False],
).reset_index(drop=True)
comparison_df.insert(0, "predictive_rank", range(1, len(comparison_df) + 1))
display(comparison_df)

predictive_best = comparison_df.iloc[0]["model"]
print("Current-run predictive leader:", predictive_best)
print("Deployment retained from June selection: FastText + Logistic Regression")

fasttext_challenge_df = pd.DataFrame(reports["FastText + Logistic Regression"]["challenge"]["details"])
print("FastText challenge failures:")
display(fasttext_challenge_df[(~fasttext_challenge_df["status_correct"]) | (~fasttext_challenge_df["issue_correct"])])

# ========================= CELL 9 ================================
# CELL 9 — Quality gate and export
fasttext_row = comparison_df.loc[comparison_df["model"] == "FastText + Logistic Regression"].iloc[0]
QUALITY_GATE = {
    "status_macro_f1_min": 0.80,
    "issue_macro_f1_min": 0.75,
    "false_acceptable_rate_max": 0.08,
    "critical_recall_min": 0.90,
    "challenge_status_accuracy_min": 1.00,
    "challenge_issue_accuracy_min": 1.00,
}
failures = []
if fasttext_row["status_macro_f1"] < QUALITY_GATE["status_macro_f1_min"]:
    failures.append("status Macro F1 below minimum")
if fasttext_row["issue_macro_f1"] < QUALITY_GATE["issue_macro_f1_min"]:
    failures.append("issue Macro F1 below minimum")
if fasttext_row["false_acceptable_rate"] > QUALITY_GATE["false_acceptable_rate_max"]:
    failures.append("false-Acceptable rate above maximum")
if fasttext_row["critical_recall"] < QUALITY_GATE["critical_recall_min"]:
    failures.append("critical-case recall below minimum")
if fasttext_row["challenge_status_accuracy"] < QUALITY_GATE["challenge_status_accuracy_min"]:
    failures.append("challenge status accuracy below minimum")
if fasttext_row["challenge_issue_accuracy"] < QUALITY_GATE["challenge_issue_accuracy_min"]:
    failures.append("challenge issue accuracy below minimum")

print("FastText quality-gate metrics:")
for key in ["status_accuracy", "status_macro_f1", "issue_accuracy", "issue_macro_f1", "false_acceptable_rate", "critical_recall", "challenge_status_accuracy", "challenge_issue_accuracy"]:
    print(f"  {key}: {float(fasttext_row[key]):.2%}")

if failures:
    display(fasttext_challenge_df[(~fasttext_challenge_df["status_correct"]) | (~fasttext_challenge_df["issue_correct"])])
    raise RuntimeError("MODEL NOT EXPORTED. Quality gate failed: " + "; ".join(failures) + ". Add/correct validated examples and retrain.")

# Export only after all checks pass.
fasttext_model.save(str(DEPLOY_DIR / "fasttext_model.model"))
joblib.dump(fasttext_status_model, DEPLOY_DIR / "status_classifier.joblib")
joblib.dump(fasttext_issue_model, DEPLOY_DIR / "issue_classifier.joblib")
shutil.copy2("/content/fasttext_feature_schema_v2.py", DEPLOY_DIR / "fasttext_feature_schema_v2.py")

labels_payload = {
    "status_labels": STATUS_LABELS,
    "issue_labels": ISSUE_LABELS,
    "status2id": STATUS2ID,
    "issue2id": ISSUE2ID,
}
(DEPLOY_DIR / "labels.json").write_text(json.dumps(labels_payload, indent=2), encoding="utf-8")

comparison_records = comparison_df.drop(columns=["predictive_rank"]).to_dict(orient="records")
package_versions = {}
for package in ["numpy", "pandas", "scipy", "gensim", "scikit-learn", "joblib", "transformers"]:
    try:
        package_versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        package_versions[package] = "not installed"

manifest = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "decision_mode": "AI_ONLY_FASTTEXT_DUAL_CLASSIFIER",
    "deployed_model": "FastText + Logistic Regression",
    "feature_schema": feature_v2.FEATURE_SCHEMA,
    "feature_module": "fasttext_feature_schema_v2.py",
    "feature_dimension": int(fasttext_train_x.shape[1]),
    "issue_feature_dimension": int(fasttext_train_x.shape[1]),
    "status_feature_dimension": int(fasttext_status_train_x.shape[1]),
    "status_classifier_uses_issue_probabilities": True,
    "numeric_feature_names": feature_v2.NUMERIC_FEATURE_NAMES,
    "predictive_best_current_run": predictive_best,
    "selection_basis": (
        "FastText was selected in the June three-approach model-selection phase. "
        "This run compares TF-IDF, enhanced FastText, and frozen DistilBERT on the same group-held-out split."
    ),
    "architecture": (
        "FastText PR vector + PO vector + signed difference + absolute difference + elementwise interaction, "
        "plus structured numeric/measurement relationship features. The issue classifier predicts first; "
        "its probability vector is appended as learned input to the trained status classifier."
    ),
    "numeric_relation_features_are_classifier_inputs_only": True,
    "runtime_rules_used": False,
    "runtime_hybrid_scoring_used": False,
    "runtime_similarity_threshold_used": False,
    "runtime_fallback_model_used": False,
    "status_labels": STATUS_LABELS,
    "issue_labels": ISSUE_LABELS,
    "fasttext_vector_size": FASTTEXT_VECTOR_SIZE,
    "fasttext_bucket": FASTTEXT_BUCKET,
    "status_classifier_c": fasttext_status_c,
    "issue_classifier_c": fasttext_issue_c,
    "training_rows": int(len(train_df)),
    "validation_rows": int(len(validation_df)),
    "test_rows": int(len(test_df)),
    "data_sources": {key: int(value) for key, value in full_df["source"].value_counts().items()},
    "quality_gate": QUALITY_GATE,
    "quality_gate_passed": True,
    "model_comparison": comparison_records,
    "fasttext_status_evaluation": reports["FastText + Logistic Regression"]["status"],
    "fasttext_issue_evaluation": reports["FastText + Logistic Regression"]["issue"],
    "semantic_challenge_evaluation": reports["FastText + Logistic Regression"]["challenge"],
    "package_versions": package_versions,
}
(DEPLOY_DIR / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
comparison_df.to_csv(DEPLOY_DIR / "model_comparison.csv", index=False)
fasttext_challenge_df.to_csv(DEPLOY_DIR / "fasttext_challenge_results.csv", index=False)

# Manuscript/evaluation evidence is kept separate so the comparison ZIP does not duplicate the model.
comparison_df.to_csv(RESULTS_DIR / "three_model_comparison.csv", index=False)
fasttext_challenge_df.to_csv(RESULTS_DIR / "fasttext_challenge_results.csv", index=False)
(RESULTS_DIR / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
for model_name, report in reports.items():
    safe_name = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    (RESULTS_DIR / f"{safe_name}_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

fasttext_zip = shutil.make_archive("/content/FASTTEXT_NLP_MODEL_AI_ONLY", "zip", root_dir=DEPLOY_DIR)
comparison_zip = shutil.make_archive("/content/THREE_NLP_COMPARISON_RESULTS", "zip", root_dir=RESULTS_DIR)
print("Deployment ZIP:", fasttext_zip, f"({Path(fasttext_zip).stat().st_size / 1024 / 1024:.1f} MB)")
print("Comparison ZIP:", comparison_zip, f"({Path(comparison_zip).stat().st_size / 1024 / 1024:.1f} MB)")
files.download(fasttext_zip)
files.download(comparison_zip)
