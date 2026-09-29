---

name: procurement-workflow-check

description: Review or test the Asuncion system's procurement approvals and inventory movements. Use when checking the PPMP-to-RIS/ICS workflow or fixing workflow regressions.

---

# Procurement Workflow Check

## Locate the implementation

Read AGENTS.md and inspect the current application before testing.

Start with capstone/app.py, the relevant templates, and

capstone/SYSTEM_SMOKE_TEST.py.

## Check the affected workflow

Trace the relevant document through:

PPMP -> APP -> PR -> AOQ -> PO -> NLP -> IAR/AIR

-> Inventory -> RIS/ICS.

Check these requirements where relevant:

- PR creation requires an approved APP.

- AOQ saving requires exactly one winning supplier.

- PO items come from the approved AOQ winning offer.

- PO approval does not require NLP.

- IAR processing requires an eligible approved NLP result.

- IAR approval adds accepted quantities to inventory.

- RIS/ICS approval deducts quantities only once.

- Repeated approval requests do not duplicate inventory changes.

## Validate safely

Use a temporary database for tests.

Inspect the smoke test's isolation before running it.

Test both a valid transition and an invalid transition for

the behavior being changed.

## Report

Describe the behavior checked, evidence, and any failures.

Separate findings confirmed by tests from observations based

only on reading code.

When fixes are requested, make focused changes and rerun

the relevant checks.
