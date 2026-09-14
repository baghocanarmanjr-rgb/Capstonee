"""Shared feature extractor for FastText PR-to-PO classification.

The module computes text-vector interaction features and structured numeric/
measurement relationship features. It never outputs a procurement decision.
The trained classifiers remain solely responsible for predicting status and issue.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Tuple

import numpy as np

FEATURE_SCHEMA = "FASTTEXT_NUMERIC_RELATION_V2"
TOKEN_PATTERN = re.compile(r"\d+(?:\.\d+)?|[a-zA-Z]+(?:[-'][a-zA-Z]+)*")
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")

# number + unit. Longer alternatives come first to avoid partial matches.
MEASUREMENT_PATTERN = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>"
    r"terabytes?|tb|gigabytes?|gb|megabytes?|mb|"
    r"kilograms?|kgs?|kg|grams?|g|"
    r"kilowatts?|kw|watts?|w|"
    r"years?|yrs?|yr|months?|mos?|mo|"
    r"inches?|inch|centimeters?|centimetres?|cm|millimeters?|millimetres?|mm|"
    r"liters?|litres?|l|milliliters?|millilitres?|ml|"
    r"ports?|ppm|dpi|gsm|"
    r"reams?|pieces?|pcs?|units?|bottles?|btls?|cartridges?|sets?"
    r")\b",
    re.IGNORECASE,
)

RANGE_MEASUREMENT_PATTERN = re.compile(
    r"(?P<low>\d+(?:\.\d+)?)\s*(?:to|and|through|-)\s*"
    r"(?P<high>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>"
    r"terabytes?|tb|gigabytes?|gb|megabytes?|mb|"
    r"kilograms?|kgs?|kg|grams?|g|"
    r"kilowatts?|kw|watts?|w|"
    r"years?|yrs?|yr|months?|mos?|mo|"
    r"inches?|inch|centimeters?|centimetres?|cm|millimeters?|millimetres?|mm|"
    r"liters?|litres?|l|milliliters?|millilitres?|ml|"
    r"ports?|ppm|dpi|gsm|"
    r"reams?|pieces?|pcs?|units?|bottles?|btls?|cartridges?|sets?"
    r")\b",
    re.IGNORECASE,
)

MINIMUM_CUES = (
    "minimum", "at least", "not less than", "no lower than", "or higher",
    "or greater", "must be at least", "minimum of",
)
MAXIMUM_CUES = (
    "maximum", "at most", "not more than", "must not exceed", "not exceed",
    "up to", "no more than", "maximum of",
)
EXACT_CUES = (
    "exactly", "shall be exactly", "required value", "exact quantity",
    "must be exactly",
)
RANGE_CUES = (
    "between", "from", "within the range", "range of", "not lower than",
    "not exceeding",
)
QUANTITY_CUES = (
    "quantity", "qty", "number of units", "number of pieces", "ordered quantity",
)

KEYWORD_GROUPS = {
    "ssd": ("ssd", "solid-state drive", "solid state drive", "solid-state storage"),
    "hdd": ("hdd", "hard disk drive", "hard-drive", "hard drive"),
    "memory": ("ram", "system memory", "installed memory", "memory capacity"),
    "weight": ("weight", "mass", "unit weight", "equipment weight"),
    "power": ("power", "watt", "watts", "power draw", "power consumption"),
    "ports": ("port", "ports", "lan port", "ethernet port", "network port"),
    "warranty": ("warranty", "service warranty", "warranty period"),
    "storage": ("storage", "disk capacity", "drive capacity", "ssd", "hdd"),
}

# canonical_group, multiplier to canonical units
UNIT_MAP: Dict[str, Tuple[str, float]] = {
    "terabyte": ("storage_gb", 1024.0), "terabytes": ("storage_gb", 1024.0), "tb": ("storage_gb", 1024.0),
    "gigabyte": ("storage_gb", 1.0), "gigabytes": ("storage_gb", 1.0), "gb": ("storage_gb", 1.0),
    "megabyte": ("storage_gb", 1.0 / 1024.0), "megabytes": ("storage_gb", 1.0 / 1024.0), "mb": ("storage_gb", 1.0 / 1024.0),
    "kilogram": ("mass_kg", 1.0), "kilograms": ("mass_kg", 1.0), "kg": ("mass_kg", 1.0), "kgs": ("mass_kg", 1.0),
    "gram": ("mass_kg", 0.001), "grams": ("mass_kg", 0.001), "g": ("mass_kg", 0.001),
    "kilowatt": ("power_w", 1000.0), "kilowatts": ("power_w", 1000.0), "kw": ("power_w", 1000.0),
    "watt": ("power_w", 1.0), "watts": ("power_w", 1.0), "w": ("power_w", 1.0),
    "year": ("time_month", 12.0), "years": ("time_month", 12.0), "yr": ("time_month", 12.0), "yrs": ("time_month", 12.0),
    "month": ("time_month", 1.0), "months": ("time_month", 1.0), "mo": ("time_month", 1.0), "mos": ("time_month", 1.0),
    "inch": ("length_in", 1.0), "inches": ("length_in", 1.0),
    "centimeter": ("length_in", 1.0 / 2.54), "centimeters": ("length_in", 1.0 / 2.54),
    "centimetre": ("length_in", 1.0 / 2.54), "centimetres": ("length_in", 1.0 / 2.54), "cm": ("length_in", 1.0 / 2.54),
    "millimeter": ("length_in", 1.0 / 25.4), "millimeters": ("length_in", 1.0 / 25.4),
    "millimetre": ("length_in", 1.0 / 25.4), "millimetres": ("length_in", 1.0 / 25.4), "mm": ("length_in", 1.0 / 25.4),
    "liter": ("volume_l", 1.0), "liters": ("volume_l", 1.0), "litre": ("volume_l", 1.0), "litres": ("volume_l", 1.0), "l": ("volume_l", 1.0),
    "milliliter": ("volume_l", 0.001), "milliliters": ("volume_l", 0.001),
    "millilitre": ("volume_l", 0.001), "millilitres": ("volume_l", 0.001), "ml": ("volume_l", 0.001),
    "port": ("ports", 1.0), "ports": ("ports", 1.0),
    "ppm": ("print_speed_ppm", 1.0), "dpi": ("resolution_dpi", 1.0), "gsm": ("paper_gsm", 1.0),
    "ream": ("qty_ream", 1.0), "reams": ("qty_ream", 1.0),
    "piece": ("qty_piece", 1.0), "pieces": ("qty_piece", 1.0), "pc": ("qty_piece", 1.0), "pcs": ("qty_piece", 1.0),
    "unit": ("qty_piece", 1.0), "units": ("qty_piece", 1.0),
    "bottle": ("qty_bottle", 1.0), "bottles": ("qty_bottle", 1.0), "btl": ("qty_bottle", 1.0), "btls": ("qty_bottle", 1.0),
    "cartridge": ("qty_cartridge", 1.0), "cartridges": ("qty_cartridge", 1.0),
    "set": ("qty_set", 1.0), "sets": ("qty_set", 1.0),
}

NUMERIC_FEATURE_NAMES = [
    "pr_number_count_scaled", "po_number_count_scaled", "comparable_measurement_found",
    "pr_numeric_missing", "po_numeric_missing", "same_measurement_group", "unit_conversion_used",
    "cue_minimum", "cue_maximum", "cue_exact", "cue_range", "cue_unspecified",
    "log_pr_low", "log_pr_high", "log_po_value",
    "signed_log_po_minus_low", "signed_log_po_minus_high",
    "abs_log_po_minus_low", "abs_log_po_minus_high",
    "ratio_po_to_low_clipped", "ratio_po_to_high_clipped",
    "po_ge_low", "po_le_low", "po_eq_low", "po_between", "po_below_low", "po_above_high",
    "pr_quantity_cue", "po_quantity_cue",
    "pr_keyword_ssd", "po_keyword_ssd", "pr_keyword_hdd", "po_keyword_hdd",
    "pr_keyword_memory", "po_keyword_memory", "pr_keyword_weight", "po_keyword_weight",
    "pr_keyword_power", "po_keyword_power", "pr_keyword_ports", "po_keyword_ports",
    "pr_keyword_warranty", "po_keyword_warranty", "pr_keyword_storage", "po_keyword_storage",
    "minimum_satisfied_interaction", "minimum_violated_interaction",
    "maximum_satisfied_interaction", "maximum_violated_interaction",
    "exact_satisfied_interaction", "exact_violated_interaction",
    "range_satisfied_interaction", "range_violated_interaction",
    "required_numeric_missing_interaction",
    "ssd_confirmed_interaction", "ssd_ambiguous_interaction", "ssd_hdd_mismatch_interaction",
    "quantity_satisfied_interaction", "quantity_mismatch_interaction",
]


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def tokenize(text: Any) -> List[str]:
    return TOKEN_PATTERN.findall(normalize_text(text).lower())


def sentence_vector(model: Any, text: Any) -> np.ndarray:
    tokens = tokenize(text)
    if not tokens:
        return np.zeros(int(model.vector_size), dtype=np.float32)
    return np.mean([model.wv[token] for token in tokens], axis=0).astype(np.float32)


def _has_cue(text: Any, cues: Tuple[str, ...]) -> float:
    lowered = normalize_text(text).lower()
    return float(any(cue in lowered for cue in cues))


def _has_keyword(text: Any, group: str) -> float:
    lowered = normalize_text(text).lower()
    return float(any(term in lowered for term in KEYWORD_GROUPS[group]))


def _signed_log(value: float) -> float:
    return float(math.copysign(math.log1p(abs(value)), value)) if value else 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 10.0) / 10.0)


def extract_measurements(text: Any) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    normalized = normalize_text(text)
    occupied: List[Tuple[int, int]] = []

    # Parse compact ranges such as "14 to 16 inches" where the unit appears once.
    for match in RANGE_MEASUREMENT_PATTERN.finditer(normalized):
        raw_unit = match.group("unit").lower()
        group, multiplier = UNIT_MAP[raw_unit]
        for key in ("low", "high"):
            raw_value = float(match.group(key))
            result.append(
                {
                    "raw_value": raw_value,
                    "raw_unit": raw_unit,
                    "group": group,
                    "value": raw_value * multiplier,
                    "converted": float(abs(multiplier - 1.0) > 1e-12),
                    "start": match.start(key),
                    "end": match.end(key),
                }
            )
        occupied.append((match.start(), match.end()))

    for match in MEASUREMENT_PATTERN.finditer(normalized):
        if any(not (match.end() <= start or match.start() >= end) for start, end in occupied):
            continue
        raw_value = float(match.group("value"))
        raw_unit = match.group("unit").lower()
        group, multiplier = UNIT_MAP[raw_unit]
        result.append(
            {
                "raw_value": raw_value,
                "raw_unit": raw_unit,
                "group": group,
                "value": raw_value * multiplier,
                "converted": float(abs(multiplier - 1.0) > 1e-12),
                "start": match.start(),
                "end": match.end(),
            }
        )
        occupied.append((match.start(), match.end()))

    # Include untyped numbers only when they were not part of a typed measurement.
    for match in NUMBER_PATTERN.finditer(normalized):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        result.append(
            {
                "raw_value": float(match.group(0)),
                "raw_unit": "",
                "group": "raw_number",
                "value": float(match.group(0)),
                "converted": 0.0,
                "start": match.start(),
                "end": match.end(),
            }
        )

    result.sort(key=lambda item: item["start"])
    return result


def _select_relation(pr_text: Any, po_text: Any) -> Dict[str, float]:
    pr_values = extract_measurements(pr_text)
    po_values = extract_measurements(po_text)

    relation = {
        "found": 0.0,
        "same_group": 0.0,
        "converted": 0.0,
        "pr_low": 0.0,
        "pr_high": 0.0,
        "po_value": 0.0,
    }
    if not pr_values or not po_values:
        return relation

    # Prefer a PR measurement group that also occurs in PO.
    selected_pr = None
    selected_po = None
    for pr_item in pr_values:
        match = next((po_item for po_item in po_values if po_item["group"] == pr_item["group"]), None)
        if match is not None:
            selected_pr = pr_item
            selected_po = match
            break

    if selected_pr is None:
        selected_pr = pr_values[0]
        selected_po = po_values[0]
    else:
        relation["same_group"] = 1.0

    same_group_pr = [item for item in pr_values if item["group"] == selected_pr["group"]]
    pr_low = float(selected_pr["value"])
    pr_high = pr_low
    if len(same_group_pr) >= 2:
        first_two = sorted(float(item["value"]) for item in same_group_pr[:2])
        pr_low, pr_high = first_two[0], first_two[1]

    relation.update(
        {
            "found": 1.0,
            "converted": float(bool(selected_pr["converted"] or selected_po["converted"])),
            "pr_low": pr_low,
            "pr_high": pr_high,
            "po_value": float(selected_po["value"]),
        }
    )
    return relation


def numeric_relation_features(pr_text: Any, po_text: Any) -> np.ndarray:
    pr_measurements = extract_measurements(pr_text)
    po_measurements = extract_measurements(po_text)
    relation = _select_relation(pr_text, po_text)

    cue_min = _has_cue(pr_text, MINIMUM_CUES)
    cue_max = _has_cue(pr_text, MAXIMUM_CUES)
    cue_exact = _has_cue(pr_text, EXACT_CUES)
    cue_range = _has_cue(pr_text, RANGE_CUES)
    cue_unspecified = float(not any((cue_min, cue_max, cue_exact, cue_range)))

    low = relation["pr_low"]
    high = relation["pr_high"]
    offered = relation["po_value"]
    diff_low = offered - low if relation["found"] else 0.0
    diff_high = offered - high if relation["found"] else 0.0
    tolerance = max(1e-6, abs(low) * 1e-6)
    found = bool(relation["found"])
    ge_low = bool(found and offered >= low - tolerance)
    le_low = bool(found and offered <= low + tolerance)
    eq_low = bool(found and abs(offered - low) <= tolerance)
    between = bool(found and low - tolerance <= offered <= high + tolerance)
    below_low = bool(found and offered < low - tolerance)
    above_high = bool(found and offered > high + tolerance)

    pr_ssd = bool(_has_keyword(pr_text, "ssd"))
    po_ssd = bool(_has_keyword(po_text, "ssd"))
    po_hdd = bool(_has_keyword(po_text, "hdd"))
    po_storage = bool(_has_keyword(po_text, "storage"))
    pr_quantity = bool(_has_cue(pr_text, QUANTITY_CUES))

    values = [
        min(len(pr_measurements), 5) / 5.0,
        min(len(po_measurements), 5) / 5.0,
        relation["found"],
        float(len(pr_measurements) == 0),
        float(len(po_measurements) == 0),
        relation["same_group"],
        relation["converted"],
        cue_min, cue_max, cue_exact, cue_range, cue_unspecified,
        math.log1p(abs(low)), math.log1p(abs(high)), math.log1p(abs(offered)),
        _signed_log(diff_low), _signed_log(diff_high),
        math.log1p(abs(diff_low)), math.log1p(abs(diff_high)),
        _safe_ratio(offered, low), _safe_ratio(offered, high),
        float(ge_low), float(le_low), float(eq_low), float(between), float(below_low), float(above_high),
        float(pr_quantity), _has_cue(po_text, QUANTITY_CUES),
        _has_keyword(pr_text, "ssd"), _has_keyword(po_text, "ssd"),
        _has_keyword(pr_text, "hdd"), _has_keyword(po_text, "hdd"),
        _has_keyword(pr_text, "memory"), _has_keyword(po_text, "memory"),
        _has_keyword(pr_text, "weight"), _has_keyword(po_text, "weight"),
        _has_keyword(pr_text, "power"), _has_keyword(po_text, "power"),
        _has_keyword(pr_text, "ports"), _has_keyword(po_text, "ports"),
        _has_keyword(pr_text, "warranty"), _has_keyword(po_text, "warranty"),
        _has_keyword(pr_text, "storage"), _has_keyword(po_text, "storage"),
        float(bool(cue_min) and ge_low), float(bool(cue_min) and below_low),
        float(bool(cue_max) and le_low), float(bool(cue_max) and above_high),
        float(bool(cue_exact) and eq_low), float(bool(cue_exact) and found and not eq_low),
        float(bool(cue_range) and between), float(bool(cue_range) and found and not between),
        float(any((cue_min, cue_max, cue_exact, cue_range)) and len(po_measurements) == 0),
        float(pr_ssd and po_ssd),
        float(pr_ssd and po_storage and not po_ssd and not po_hdd),
        float(pr_ssd and po_hdd),
        float(pr_quantity and eq_low), float(pr_quantity and found and not eq_low),
    ]
    features = np.asarray(values, dtype=np.float32)
    if features.shape[0] != len(NUMERIC_FEATURE_NAMES):
        raise RuntimeError(
            f"Numeric feature schema mismatch: {features.shape[0]} values for "
            f"{len(NUMERIC_FEATURE_NAMES)} names"
        )
    return features


def build_pair_features(model: Any, pr_text: Any, po_text: Any) -> np.ndarray:
    pr_vector = sentence_vector(model, pr_text)
    po_vector = sentence_vector(model, po_text)
    text_features = np.concatenate(
        [
            pr_vector,
            po_vector,
            po_vector - pr_vector,
            np.abs(pr_vector - po_vector),
            pr_vector * po_vector,
        ]
    ).astype(np.float32)
    numeric_features = numeric_relation_features(pr_text, po_text)
    return np.concatenate([text_features, numeric_features]).astype(np.float32)


def expected_feature_dimension(vector_size: int) -> int:
    return int(vector_size) * 5 + len(NUMERIC_FEATURE_NAMES)
