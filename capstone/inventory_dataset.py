"""Dynamic inventory workbook import for the PPMP item source.

This module detects any number of worksheets, discovers header rows and column
aliases, preserves additional fields, and synchronizes changed rows. It does
not perform anomaly detection. The only trained check is PR-to-PO NLP.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import re
import sqlite3
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter


CANONICAL_FIELDS = [
    "article",
    "description",
    "property_number",
    "unit",
    "unit_value",
    "balance_qty",
    "on_hand_qty",
    "shortage_qty",
    "shortage_value",
    "remarks",
    "accountable_person",
    "office",
]

CRITICAL_FIELDS = [
    "article",
    "description",
    "unit",
    "unit_value",
    "balance_qty",
    "on_hand_qty",
]

HEADER_ALIASES = {
    "article": ["article", "category", "asset category", "property type", "type of property"],
    "description": [
        "description", "item description", "particulars", "item name",
        "property description", "name of item", "specification",
    ],
    "property_number": [
        "semi expendable property no", "semi expendable property number",
        "property no", "property number", "property code", "asset number",
        "inventory tag", "inventory number",
    ],
    "unit": ["unit of measure", "unit", "uom", "measurement unit"],
    "unit_value": [
        "unit value", "unit cost", "acquisition cost", "recorded value",
        "amount", "cost", "price",
    ],
    "balance_qty": [
        "balance per card", "balance quantity", "book quantity", "card balance",
        "quantity per card", "balance",
    ],
    "on_hand_qty": [
        "on hand per card", "on hand quantity", "actual quantity", "physical count",
        "quantity on hand", "on hand",
    ],
    "shortage_qty": [
        "shortage overage quantity", "shortage quantity", "overage quantity",
        "quantity shortage overage", "variance quantity",
    ],
    "shortage_value": [
        "shortage overage value", "shortage value", "overage value",
        "value shortage overage", "variance value",
    ],
    "remarks": ["remarks", "remark", "notes", "note", "location", "assigned office"],
    "accountable_person": [
        "person accountable", "accountable person", "accountable officer",
        "property custodian", "assigned to", "end user",
    ],
    "office": ["office", "department", "division", "unit office", "assigned department"],
}

PROPERTY_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{3}-\d{4}-\d{2}$")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.I)


def _now() -> str:
    return _dt.datetime.now().strftime("%b %d, %Y %I:%M %p")


def _norm(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _to_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            if math.isnan(float(value)):
                return 0.0
        except Exception:
            pass
        return float(value)
    text = str(value).replace("₱", "").replace(",", "").replace("PHP", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else 0.0


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _match_header(value: Any) -> Optional[str]:
    text = _norm(value)
    if not text:
        return None

    # The merged COA heading "Shortage/Overage" is resolved later using the
    # Quantity/Value subheading in the following rows.
    if text in {"shortage overage", "shortage or overage", "variance"}:
        return "shortage_qty"

    best: Optional[Tuple[int, str]] = None
    for canonical, aliases in HEADER_ALIASES.items():
        for alias in aliases:
            a = _norm(alias)
            score = 0
            if text == a:
                score = 100 + len(a)
            elif len(a) >= 5 and a in text:
                score = 50 + len(a)
            elif len(text) >= 5 and text in a:
                score = 20 + len(text)
            if score and (best is None or score > best[0]):
                best = (score, canonical)
    return best[1] if best else None


def _header_candidate(rows: Sequence[Sequence[Any]], row_index: int) -> Tuple[int, Dict[int, str]]:
    row = rows[row_index]
    mapping: Dict[int, str] = {}
    score = 0
    found = set()
    for col, value in enumerate(row):
        canonical = _match_header(value)
        if canonical and canonical not in found:
            mapping[col] = canonical
            found.add(canonical)
            score += 3 if canonical in {"article", "description", "property_number"} else 2

    # Resolve merged shortage/overage columns from subheaders in the next 2 rows.
    for col, value in enumerate(row):
        text = _norm(value)
        if "shortage" in text or "overage" in text or text == "variance":
            mapping[col] = "shortage_qty"
            for probe_col in (col, col + 1):
                for next_idx in (row_index + 1, row_index + 2):
                    if next_idx >= len(rows) or probe_col >= len(rows[next_idx]):
                        continue
                    sub = _norm(rows[next_idx][probe_col])
                    if sub == "value" or "value" in sub:
                        mapping[probe_col] = "shortage_value"
                    elif sub == "quantity" or "quantity" in sub:
                        mapping[probe_col] = "shortage_qty"
            if col + 1 < len(row) and col + 1 not in mapping:
                mapping[col + 1] = "shortage_value"

    # Strongly prefer rows containing both article and description.
    if "article" in found and "description" in found:
        score += 8
    if "unit_value" in found:
        score += 3
    if "balance_qty" in found and "on_hand_qty" in found:
        score += 4
    return score, mapping


def detect_header(rows: Sequence[Sequence[Any]]) -> Tuple[int, Dict[int, str], Dict[int, str]]:
    """Return zero-based header row, canonical column map and display labels."""
    if not rows:
        return 0, {}, {}
    scan_limit = min(len(rows), 60)
    candidates = [_header_candidate(rows, idx) for idx in range(scan_limit)]
    best_idx = max(range(scan_limit), key=lambda idx: candidates[idx][0])
    best_score, mapping = candidates[best_idx]
    if best_score < 8:
        # Generic fallback: use the densest row near the top as the header.
        # This avoids treating a report title as the header when a future sheet
        # uses unfamiliar column names. All columns are still preserved.
        best_idx = max(
            range(scan_limit),
            key=lambda i: sum(v not in (None, "") for v in rows[i]),
        )
        mapping = {}

    labels: Dict[int, str] = {}
    header_row = rows[best_idx]
    sub_rows = rows[best_idx + 1: best_idx + 3]
    max_cols = max([len(header_row)] + [len(r) for r in sub_rows] + [0])
    for col in range(max_cols):
        main = _clean_text(header_row[col]) if col < len(header_row) else ""
        sub = ""
        for sr in sub_rows:
            if col < len(sr) and _clean_text(sr[col]):
                sub = _clean_text(sr[col])
                break
        if main:
            if ("shortage" in _norm(main) or "overage" in _norm(main) or _norm(main) == "variance") and sub:
                label = f"{main} - {sub}"
            else:
                label = main
        elif sub and _norm(sub) in {"quantity", "value", "qty", "amount"}:
            label = sub
        else:
            label = f"Column {get_column_letter(col + 1)}"
        labels[col] = label

    # Assign any recognized aliases not captured because duplicate headers were
    # encountered. The canonical map remains position based, not sheet-name based.
    for col in range(max_cols):
        if col not in mapping:
            value = header_row[col] if col < len(header_row) else None
            canonical = _match_header(value)
            if canonical and canonical not in mapping.values():
                mapping[col] = canonical
    return best_idx, mapping, labels


def _is_placeholder_property(value: Any) -> bool:
    text = _clean_text(value)
    if not text:
        return False
    digits = re.sub(r"\D", "", text)
    return bool(digits) and set(digits) == {"0"}


def _record_identity_base(record: Dict[str, Any]) -> str:
    prop = _norm(record.get("property_number"))
    if prop and not _is_placeholder_property(prop):
        return f"{_norm(record.get('source_sheet'))}|property|{prop}"
    return "|".join([
        _norm(record.get("source_sheet")),
        _norm(record.get("article")),
        _norm(record.get("description")),
        _norm(record.get("accountable_person")),
        _norm(record.get("office")),
        _norm(record.get("unit")),
    ])


def _record_identity(record: Dict[str, Any], occurrence: int) -> str:
    base = _record_identity_base(record)
    return hashlib.sha256(f"{base}|{occurrence}".encode("utf-8")).hexdigest()


def read_dynamic_workbook(path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Read every sheet and every valid row without hard-coded sheet names."""
    wb = load_workbook(path, data_only=True, read_only=True)
    all_records: List[Dict[str, Any]] = []
    sheet_meta: List[Dict[str, Any]] = []
    try:
        for ws in wb.worksheets:
            rows = list(ws.iter_rows(values_only=True))
            if not rows:
                sheet_meta.append({"sheet_name": ws.title, "header_row": 1, "row_count": 0, "columns": {}})
                continue
            header_idx, mapping, labels = detect_header(rows)
            records: List[Dict[str, Any]] = []
            identity_counts: Counter[str] = Counter()
            for zero_idx, row in enumerate(rows[header_idx + 1:], start=header_idx + 1):
                if not any(value not in (None, "") for value in row):
                    continue

                canonical = {field: "" for field in CANONICAL_FIELDS}
                raw: Dict[str, Any] = {}
                meaningful_values = 0
                max_cols = max(len(row), len(labels))
                for col in range(max_cols):
                    value = row[col] if col < len(row) else None
                    if value in (None, ""):
                        continue
                    meaningful_values += 1
                    label = labels.get(col, f"Column {get_column_letter(col + 1)}")
                    # Preserve duplicate labels without overwriting source data.
                    raw_label = label
                    suffix = 2
                    while raw_label in raw:
                        raw_label = f"{label} ({suffix})"
                        suffix += 1
                    raw[raw_label] = value
                    field = mapping.get(col)
                    if field:
                        canonical[field] = value

                if meaningful_values == 0:
                    continue
                normalized_raw_values = [_norm(v) for v in raw.values() if v not in (None, "")]
                normalized_raw_values = [v for v in normalized_raw_values if v]
                # Ignore secondary header rows such as (Quantity)/(Value), and
                # decorative signature lines made only of underscores/dashes.
                if normalized_raw_values and set(normalized_raw_values).issubset({"quantity", "value", "qty", "amount"}):
                    continue
                if raw and not any(re.search(r"[A-Za-z0-9]", str(v or "")) for v in raw.values()):
                    continue

                # Generic-schema fallback for future sheets whose headers are
                # not yet in the alias dictionary. The raw fields remain the
                # source of truth; these inferred values only make the row
                # searchable and usable as a PPMP reference without recoding.
                raw_text_candidates = []
                raw_numeric_candidates = []
                for raw_value in raw.values():
                    if raw_value in (None, ""):
                        continue
                    if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
                        raw_numeric_candidates.append(float(raw_value))
                        continue
                    raw_text = _clean_text(raw_value)
                    numeric_text = raw_text.replace('₱', '').replace('PHP', '').replace(',', '')
                    if re.fullmatch(r'-?\d+(?:\.\d+)?', numeric_text):
                        raw_numeric_candidates.append(float(numeric_text))
                    elif raw_text:
                        raw_text_candidates.append(raw_text)
                if not _clean_text(canonical.get("description")) and raw_text_candidates:
                    canonical["description"] = max(raw_text_candidates, key=len)
                if not _clean_text(canonical.get("article")):
                    canonical["article"] = ws.title
                if _to_float(canonical.get("unit_value")) == 0 and raw_numeric_candidates:
                    canonical["unit_value"] = raw_numeric_candidates[0]

                article = _clean_text(canonical.get("article"))
                description = _clean_text(canonical.get("description"))
                property_number = _clean_text(canonical.get("property_number"))
                numeric_presence = any(_to_float(canonical.get(f)) != 0 for f in (
                    "unit_value", "balance_qty", "on_hand_qty", "shortage_qty", "shortage_value"
                ))
                # Ignore report titles, repeated header rows and totals while
                # retaining sparse but valid inventory records.
                if not (article or description or property_number or numeric_presence):
                    continue
                if _norm(article) in {"article", "total", "grand total"} and not description:
                    continue
                if _norm(description) in {"description", "item description", "particulars"}:
                    continue
                footer_text = _norm(" ".join(str(v) for v in row if v not in (None, "")))
                if any(marker in footer_text for marker in (
                    "certified correct by", "approved by", "verified by",
                    "signature over printed name", "inventory committee chair",
                    "coa representative",
                )):
                    continue

                remarks = _clean_text(canonical.get("remarks"))
                office = _clean_text(canonical.get("office")) or remarks
                record: Dict[str, Any] = {
                    "source_sheet": ws.title,
                    "source_row": zero_idx + 1,
                    "article": article,
                    "description": description,
                    "property_number": property_number,
                    "unit": _clean_text(canonical.get("unit")),
                    "unit_value": _to_float(canonical.get("unit_value")),
                    "balance_qty": _to_float(canonical.get("balance_qty")),
                    "on_hand_qty": _to_float(canonical.get("on_hand_qty")),
                    "shortage_qty": _to_float(canonical.get("shortage_qty")),
                    "shortage_value": _to_float(canonical.get("shortage_value")),
                    "remarks": remarks,
                    "office": office,
                    "accountable_person": _clean_text(canonical.get("accountable_person")),
                    "raw_data": raw,
                }
                base = _record_identity_base(record)
                identity_counts[base] += 1
                record["record_key"] = _record_identity(record, identity_counts[base])
                records.append(record)
                all_records.append(record)

            sheet_meta.append({
                "sheet_name": ws.title,
                "header_row": header_idx + 1,
                "row_count": len(records),
                "columns": {str(col + 1): {"label": labels.get(col, ""), "field": mapping.get(col, "")} for col in labels},
            })
    finally:
        wb.close()
    return all_records, sheet_meta



def ensure_schema(db_path: str) -> None:
    """Create the dynamic inventory-source repository tables."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        con.executescript(
            """
            create table if not exists dataset_config(
                id integer primary key check(id=1),
                filename text,
                stored_path text,
                file_hash text,
                modified_ts real,
                last_synced text,
                sheet_count integer default 0,
                row_count integer default 0,
                model_version text,
                status text,
                last_error text
            );
            create table if not exists dataset_sheets(
                id integer primary key autoincrement,
                sheet_name text unique,
                header_row integer,
                row_count integer,
                columns_json text,
                active integer default 1,
                updated_at text
            );
            create table if not exists dataset_records(
                id integer primary key autoincrement,
                record_key text unique,
                sheet_name text,
                source_row integer,
                article text,
                description text,
                property_number text,
                unit text,
                unit_value real,
                balance_qty real,
                on_hand_qty real,
                shortage_qty real,
                shortage_value real,
                remarks text,
                office text,
                accountable_person text,
                raw_data text,
                normalized_data text,
                anomaly_score real,
                anomaly_label text,
                anomaly_reasons text,
                review_status text,
                active integer default 1,
                imported_at text,
                updated_at text
            );
            create index if not exists idx_dataset_records_sheet on dataset_records(sheet_name, active);
            create index if not exists idx_dataset_records_description on dataset_records(description);
            create index if not exists idx_dataset_records_property on dataset_records(property_number);
            """
        )
        con.execute("update dataset_records set anomaly_score=null, anomaly_label=null, anomaly_reasons=null, review_status=null")
        con.execute("update dataset_config set model_version=null")
        con.execute("drop table if exists inventory_anomaly_runs")
        con.commit()
    finally:
        con.close()


def sync_workbook(db_path: str, workbook_path: str, *, display_filename: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
    """Synchronize all worksheets and valid rows without training an AI model."""
    ensure_schema(db_path)
    if not os.path.exists(workbook_path):
        raise FileNotFoundError(workbook_path)
    file_hash = _hash_file(workbook_path)
    modified_ts = os.path.getmtime(workbook_path)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        config = con.execute("select * from dataset_config where id=1").fetchone()
        active_count = con.execute("select count(*) from dataset_records where active=1").fetchone()[0]
        if config and config["file_hash"] == file_hash and active_count and not force:
            if config["stored_path"] != workbook_path:
                con.execute("update dataset_config set stored_path=? where id=1", (workbook_path,))
                con.commit()
            return {"changed": False, "filename": config["filename"], "sheet_count": config["sheet_count"], "row_count": config["row_count"], "last_synced": config["last_synced"], "new": 0, "updated": 0, "removed": 0}
    finally:
        con.close()

    records, sheets = read_dynamic_workbook(workbook_path)
    if not records:
        raise ValueError("No valid inventory rows were detected in any worksheet.")

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    new_count = updated_count = removed_count = 0
    try:
        con.execute("begin")
        existing_rows = {r["record_key"]: r for r in con.execute("select * from dataset_records")}
        incoming_keys = {r["record_key"] for r in records}
        for record in records:
            normalized = {"article": _norm(record.get("article")), "description": _norm(record.get("description")), "property_number": _norm(record.get("property_number")), "unit": _norm(record.get("unit"))}
            common = (
                record["source_sheet"], record["source_row"], record.get("article", ""), record.get("description", ""),
                record.get("property_number", ""), record.get("unit", ""), record.get("unit_value", 0),
                record.get("balance_qty", 0), record.get("on_hand_qty", 0), record.get("shortage_qty", 0),
                record.get("shortage_value", 0), record.get("remarks", ""), record.get("office", ""),
                record.get("accountable_person", ""), json.dumps(record.get("raw_data") or {}, default=str),
                json.dumps(normalized, default=str),
            )
            if record["record_key"] in existing_rows:
                con.execute("""update dataset_records set sheet_name=?,source_row=?,article=?,description=?,property_number=?,unit=?,unit_value=?,balance_qty=?,on_hand_qty=?,shortage_qty=?,shortage_value=?,remarks=?,office=?,accountable_person=?,raw_data=?,normalized_data=?,anomaly_score=null,anomaly_label=null,anomaly_reasons=null,review_status=null,active=1,updated_at=? where record_key=?""", common + (_now(), record["record_key"]))
                updated_count += 1
            else:
                con.execute("""insert into dataset_records(sheet_name,source_row,article,description,property_number,unit,unit_value,balance_qty,on_hand_qty,shortage_qty,shortage_value,remarks,office,accountable_person,raw_data,normalized_data,anomaly_score,anomaly_label,anomaly_reasons,review_status,active,imported_at,updated_at,record_key) values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,null,null,null,null,1,?,?,?)""", common + (_now(), _now(), record["record_key"]))
                new_count += 1
        for key, old in existing_rows.items():
            if old["active"] and key not in incoming_keys:
                con.execute("update dataset_records set active=0,updated_at=? where record_key=?", (_now(), key))
                removed_count += 1
        con.execute("update dataset_sheets set active=0")
        for sheet in sheets:
            con.execute("""insert into dataset_sheets(sheet_name,header_row,row_count,columns_json,active,updated_at) values(?,?,?,?,1,?) on conflict(sheet_name) do update set header_row=excluded.header_row,row_count=excluded.row_count,columns_json=excluded.columns_json,active=1,updated_at=excluded.updated_at""", (sheet["sheet_name"], sheet["header_row"], sheet["row_count"], json.dumps(sheet["columns"], default=str), _now()))
        filename = display_filename or os.path.basename(workbook_path)
        con.execute("""insert into dataset_config(id,filename,stored_path,file_hash,modified_ts,last_synced,sheet_count,row_count,model_version,status,last_error) values(1,?,?,?,?,?,?,?,null,?,null) on conflict(id) do update set filename=excluded.filename,stored_path=excluded.stored_path,file_hash=excluded.file_hash,modified_ts=excluded.modified_ts,last_synced=excluded.last_synced,sheet_count=excluded.sheet_count,row_count=excluded.row_count,model_version=null,status=excluded.status,last_error=null""", (filename, workbook_path, file_hash, modified_ts, _now(), len(sheets), len(records), "Ready"))
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    return {"changed": True, "filename": display_filename or os.path.basename(workbook_path), "sheet_count": len(sheets), "row_count": len(records), "last_synced": _now(), "new": new_count, "updated": updated_count, "removed": removed_count}


def get_status(db_path: str) -> Dict[str, Any]:
    ensure_schema(db_path)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        config = con.execute("select * from dataset_config where id=1").fetchone()
        sheet_rows = con.execute("select * from dataset_sheets where active=1 order by sheet_name collate nocase").fetchall()
        return {"config": dict(config) if config else None, "sheets": [dict(r) for r in sheet_rows]}
    finally:
        con.close()
