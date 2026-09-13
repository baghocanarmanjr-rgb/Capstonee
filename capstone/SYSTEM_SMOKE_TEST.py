"""Run an isolated end-to-end smoke test without changing the real system database.

This test uses a temporary SQLite database and temporary upload directory. It validates:
PPMP fixture -> APP -> PR -> flexible AOQ upload -> AOQ approval -> PO approval without NLP,
then verifies that IAR is blocked until an approved NLP record exists, and continues through
IAR -> Inventory -> ICS -> Reports. It does not invent or evaluate an NLP model prediction.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

import app as system


class SmokeFailure(RuntimeError):
    pass


def expect(name: str, condition: bool, detail: object = "") -> None:
    if not condition:
        raise SmokeFailure(f"{name} failed: {detail}")
    print(f"PASS: {name}")


def main() -> int:
    original = {
        "DB_PATH": system.DB_PATH,
        "AOQ_UPLOAD_DIR": system.AOQ_UPLOAD_DIR,
        "EXCEL_PATH": system.EXCEL_PATH,
        "COA_DATASET_PATH": system.COA_DATASET_PATH,
        "DATABASE_READY": getattr(system, "_DATABASE_READY", False),
    }

    with tempfile.TemporaryDirectory(prefix="kapston_smoke_") as tmp:
        tmp_path = Path(tmp)
        system.DB_PATH = str(tmp_path / "instance" / "psms.db")
        system.AOQ_UPLOAD_DIR = str(tmp_path / "uploads" / "aoq")
        system.EXCEL_PATH = str(ROOT / "data" / "PPMP_Data.xlsx")
        # Avoid synchronizing the reference workbook into this temporary workflow database.
        system.COA_DATASET_PATH = str(tmp_path / "no_reference_workbook.xlsx")
        system._DATABASE_READY = False
        system.init_db()
        system._DATABASE_READY = True

        client = system.app.test_client()
        response = client.post(
            "/login",
            data={"username": "admin", "password": "admin123"},
            follow_redirects=True,
        )
        expect("administrator login", response.status_code == 200)

        item = {
            "Item Name": "Office Chair",
            "Description / Specification": "Ergonomic office chair, mesh back, adjustable height",
            "Quantity": 10,
            "Unit": "unit",
            "Estimated Unit Cost": 2750,
            "Estimated Budget Allocation (Php)": 27500,
            "Schedule / Milestone": "Q3",
            "Mode of Procurement": "Small Value Procurement",
        }
        ppmp_id = system.exec_db(
            """insert into ppmp(no,type,office,year,status,created_at,items,total)
               values(?,?,?,?,?,?,?,?)""",
            (
                "PPMP-2026-NONDBM-0001",
                "NON-DBM",
                "General Services Office",
                2026,
                "Draft",
                system.now(),
                json.dumps([item]),
                27500,
            ),
        )

        response = client.post("/app", data={"ppmp_id": ppmp_id}, follow_redirects=True)
        expect("APP generation", response.status_code == 200)
        app_row = system.q("select * from app_plan where ppmp_id=?", (ppmp_id,), one=True)
        expect("APP record saved", bool(app_row))
        client.post(
            f"/app/{app_row['id']}/approve",
            data={"review_confirm": "yes"},
            follow_redirects=True,
        )
        app_row = system.q("select * from app_plan where id=?", (app_row["id"],), one=True)
        expect("APP approval", app_row["status"] == "Approved", app_row["status"])

        client.post(
            "/pr",
            data={"app_id": app_row["id"], "purpose": "Procurement of ergonomic office chairs."},
            follow_redirects=True,
        )
        pr_row = system.q("select * from pr where app_id=?", (app_row["id"],), one=True)
        expect("PR created from approved APP", bool(pr_row) and pr_row["app_id"] == app_row["id"])
        client.post(
            f"/pr/{pr_row['id']}/approve",
            data={"review_confirm": "yes"},
            follow_redirects=True,
        )
        pr_row = system.q("select * from pr where id=?", (pr_row["id"],), one=True)
        expect("PR approval", pr_row["status"] == "Approved", pr_row["status"])

        sample = ROOT / "sample_files" / "SAMPLE_AOQ_ONE_WINNER.xlsx"
        with sample.open("rb") as fh:
            response = client.post(
                "/aoq",
                data={
                    "action": "extract_ocr",
                    "pr_id": str(pr_row["id"]),
                    "ocr_file": (fh, sample.name),
                },
                content_type="multipart/form-data",
                follow_redirects=True,
            )
        expect("flexible AOQ upload", response.status_code == 200)

        with client.session_transaction() as flask_session:
            preview_name = flask_session.get("aoq_preview_file")
        preview_path = Path(system.AOQ_UPLOAD_DIR) / str(preview_name or "")
        expect("AOQ extraction preview", bool(preview_name) and preview_path.exists(), preview_path)
        preview = json.loads(preview_path.read_text(encoding="utf-8"))
        suppliers = preview.get("suppliers") or []
        offer_items = preview.get("offer_items") or []
        expect("one detected AOQ supplier", len(suppliers) == 1, suppliers)
        expect("AOQ winning-offer item detected", len(offer_items) >= 1, offer_items)
        supplier = suppliers[0]
        offered = offer_items[0]

        response = client.post(
            "/aoq",
            data={
                "action": "save_ocr",
                "pr_id": str(pr_row["id"]),
                "winner_index": "0",
                "winner_custom": "",
                "remarks": "Compliant",
                "supplier_name": [supplier.get("supplier")],
                "supplier_amount": [supplier.get("amount")],
                "supplier_remarks": [supplier.get("remarks")],
                "offer_item": [offered.get("item")],
                "offer_description": [offered.get("description")],
                "offer_qty": [offered.get("qty")],
                "offer_unit": [offered.get("unit")],
                "offer_unit_cost": [2750],
                "offer_amount": [offered.get("amount")],
                "offer_remarks": [offered.get("remarks")],
            },
            follow_redirects=True,
        )
        expect("AOQ save", response.status_code == 200)
        aoq_row = system.q("select * from aoq where pr_id=?", (pr_row["id"],), one=True)
        expect("AOQ official winner", aoq_row["winner"] == "Davao Office Furniture Trading", aoq_row["winner"])
        client.post(
            f"/aoq/{aoq_row['id']}/approve",
            data={"review_confirm": "yes"},
            follow_redirects=True,
        )
        aoq_row = system.q("select * from aoq where id=?", (aoq_row["id"],), one=True)
        expect("AOQ approval", aoq_row["status"] == "Approved", aoq_row["status"])

        client.post("/po", data={"aoq_id": aoq_row["id"]}, follow_redirects=True)
        po_row = system.q("select * from po where aoq_id=?", (aoq_row["id"],), one=True)
        expect("PO generated from AOQ winner", bool(po_row) and po_row["supplier"] == aoq_row["winner"])
        expect("PO exists before NLP", not system.q("select * from nlp where po_id=?", (po_row["id"],), one=True))
        client.post(
            f"/po/{po_row['id']}/approve",
            data={"review_confirm": "yes"},
            follow_redirects=True,
        )
        po_row = system.q("select * from po where id=?", (po_row["id"],), one=True)
        expect("PO approval without NLP", po_row["status"] == "Approved")

        client.post("/iar", data={"po_id": po_row["id"]}, follow_redirects=True)
        expect("IAR blocked before approved NLP", not system.q("select * from iar where po_id=?", (po_row["id"],), one=True))

        # Test fixture only: this validates the workflow after NLP approval; it is not a model prediction.
        system.exec_db(
            """insert into nlp(no,po_id,result,confidence,staff_decision,remarks,created_at,details,status)
               values(?,?,?,?,?,?,?,?,?)""",
            (
                "NLP-2026-SMOKE",
                po_row["id"],
                "Acceptable",
                100.0,
                "Acceptable",
                "Smoke-test fixture used only to validate downstream workflow.",
                system.now(),
                "{}",
                "Approved",
            ),
        )
        client.post("/iar", data={"po_id": po_row["id"]}, follow_redirects=True)
        iar_row = system.q("select * from iar where po_id=?", (po_row["id"],), one=True)
        expect("IAR creation after approved NLP", bool(iar_row))
        client.post(
            f"/iar/{iar_row['id']}/approve",
            data={"review_confirm": "yes"},
            follow_redirects=True,
        )
        inventory_row = system.q("select * from inventory where source_po=?", (po_row["no"],), one=True)
        expect("IAR adds live inventory", bool(inventory_row) and float(inventory_row["qty"]) == 10.0)

        client.post(
            "/ris-ics",
            data={
                "type": "ICS",
                "office": "General Services Office",
                "inventory_id": [str(inventory_row["id"])],
                f"issue_qty_{inventory_row['id']}": "1",
            },
            follow_redirects=True,
        )
        issue_row = system.q("select * from ris_ics order by id desc limit 1", one=True)
        expect("ICS creation", bool(issue_row) and issue_row["type"] == "ICS")
        client.post(
            f"/ris-ics/{issue_row['id']}/approve",
            data={"review_confirm": "yes"},
            follow_redirects=True,
        )
        inventory_after = system.q("select * from inventory where id=?", (inventory_row["id"],), one=True)
        expect("ICS deducts inventory once", float(inventory_after["qty"]) == 9.0, inventory_after["qty"])
        client.post(
            f"/ris-ics/{issue_row['id']}/approve",
            data={"review_confirm": "yes"},
            follow_redirects=True,
        )
        inventory_after = system.q("select * from inventory where id=?", (inventory_row["id"],), one=True)
        expect("repeat ICS approval is idempotent", float(inventory_after["qty"]) == 9.0, inventory_after["qty"])

        major_pages = [
            "/", "/ppmp", "/app", "/pr", "/aoq", "/po", "/nlp", "/iar",
            "/inventory", "/ris-ics", "/reports", "/users", "/audit", "/nlp/algorithms",
            f"/app/{app_row['id']}/view", f"/pr/{pr_row['id']}/view",
            f"/aoq/{aoq_row['id']}/view", f"/po/{po_row['id']}/view",
            f"/iar/{iar_row['id']}/view", f"/ris-ics/{issue_row['id']}/view",
        ]
        for path in major_pages:
            page = client.get(path)
            expect(f"render {path}", page.status_code == 200, page.status_code)

        for report in ("procurement", "inventory", "issuance", "audit"):
            exported = client.get(f"/reports/export/{report}.csv")
            expect(f"CSV report {report}", exported.status_code == 200 and len(exported.data) > 20)

        with sqlite3.connect(system.DB_PATH) as con:
            expect("temporary database integrity", con.execute("pragma integrity_check").fetchone()[0] == "ok")

    system.DB_PATH = original["DB_PATH"]
    system.AOQ_UPLOAD_DIR = original["AOQ_UPLOAD_DIR"]
    system.EXCEL_PATH = original["EXCEL_PATH"]
    system.COA_DATASET_PATH = original["COA_DATASET_PATH"]
    system._DATABASE_READY = original["DATABASE_READY"]
    print("\nCOMPLETE SYSTEM SMOKE TEST PASSED.")
    print("The real database and uploads were not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
