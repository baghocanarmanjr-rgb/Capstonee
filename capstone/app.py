from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, Response, abort
import sqlite3, os, json, datetime, random, re, shutil, csv, io
from openpyxl import load_workbook
from flask import session
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image, ImageOps, ImageFilter
import pytesseract
from aoq_flexible import (
    SUPPORTED_EXTENSIONS as FLEXIBLE_AOQ_EXTENSIONS,
    extract_document as extract_aoq_document,
    build_preview as build_flexible_aoq_preview,
)
from inventory_dataset import (
    ensure_schema as ensure_inventory_dataset_schema,
    sync_workbook as sync_inventory_dataset_workbook,
    get_status as get_inventory_dataset_status,
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, 'instance', 'psms.db')
EXCEL_PATH = os.path.join(BASE_DIR, 'data', 'PPMP_Data.xlsx')
COA_DATASET_PATH = os.path.join(BASE_DIR, 'data', 'COA_Inventory_Dataset.xlsx')
ALLOWED_DATASET_EXTENSIONS = {'.xlsx', '.xlsm'}
NLP_FASTTEXT_DIR = os.path.join(BASE_DIR, 'model', 'fasttext_nlp')
AOQ_UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads', 'aoq')
ALLOWED_OCR_EXTENSIONS = set(FLEXIBLE_AOQ_EXTENSIONS)

app = Flask(__name__)
app.secret_key = os.environ.get('PSMS_SECRET_KEY', 'asuncion-psms-offline-change-me')



SIGNATORIES = {
    'ppmp_prepared': ('EDEN B. PUNO', 'AAIII'),
    'ppmp_approved': ('EnP. IRIS HOPHET E. SANTIGA, MBA', 'MPDC'),
    'app_prepared': ('JULIETA C. MALABARBAS', 'Private Secretary II'),
    'app_approved': ('ATTY. EUFRACIO P. DAYADAY, JR., MPA', 'Municipal Mayor / Head of Procuring Entity'),
    'pr_requested': ('JUVY G. ALMEDA, MPA', 'MGSO'),
    'pr_cash': ('HERMES E. COSMOD, MPA', 'Municipal Treasurer'),
    'pr_approved': ('ATTY. ROSARIO R. DAYADAY', 'Municipal Mayor'),
    'po_authorized': ('ATTY. ROSARIO R. DAYADAY', 'Municipal Mayor'),
    'iar_acceptance': ('JUVY G. ALMEDA', 'Supply and/or Property Custodian'),
    'iar_inspection': ('GINA S. PERALTA', 'Inspection Officer / Inspection Committee'),
    'ris_requested': ('EUFRONIA J. MANGLE, LPT', 'PESO MANAGER/CTEC'),
    'ris_approved': ('ATTY. ROSARIO R. DAYADAY', 'Municipal Mayor'),
    'ris_issued': ('JUVY G. ALMEDA, MPA', 'MGSO'),
    'ris_received': ('EUFRONIA J. MANGLE, LPT', 'PESO MANAGER/CTEC'),
}

def conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def q(sql, params=(), one=False):
    con = conn(); cur = con.execute(sql, params); rows = cur.fetchall(); con.close(); return rows[0] if one and rows else rows

def exec_db(sql, params=()):
    con = conn(); cur = con.execute(sql, params); con.commit(); last = cur.lastrowid; con.close(); return last

def log(action, module, ref, details=''):
    exec_db('insert into audit_trail(dt,user,action,module,ref,details) values(?,?,?,?,?,?)', (now(), session.get('username', 'admin'), action, module, ref, details))

def now(): return datetime.datetime.now().strftime('%b %d, %Y %I:%M %p')
def today(): return datetime.date.today().strftime('%Y-%m-%d')


def current_role():
    return str(session.get('role') or '').strip() or 'Staff'


def is_admin():
    return current_role().lower() == 'admin'


def require_admin_access():
    if not is_admin():
        flash('Administrator access is required for that page.')
        return redirect(url_for('dashboard'))
    return None
def to_float(v):
    try:
        if v is None or v == '':
            return 0.0
        if isinstance(v, (int, float)):
            return float(v)
        cleaned = str(v).replace('₱', '').replace(',', '').strip()
        return float(cleaned or 0)
    except Exception:
        return 0.0


def current_year():
    return datetime.date.today().year


def next_document_no(kind, prefix, table, column='no', year=None, width=4):
    """Generate a readable, collision-safe identifier using a persistent sequence."""
    year = int(year or current_year())
    key = f'{kind}:{year}'
    con = conn()
    try:
        con.execute('begin immediate')
        row = con.execute('select last_number from identifier_sequence where sequence_key=?', (key,)).fetchone()
        number = int(row['last_number']) if row else 0
        while True:
            number += 1
            candidate = f'{prefix}-{year}-{number:0{width}d}'
            exists = con.execute(f'select 1 from {table} where {column}=? limit 1', (candidate,)).fetchone()
            if not exists:
                break
        con.execute(
            '''insert into identifier_sequence(sequence_key,last_number)
               values(?,?)
               on conflict(sequence_key) do update set last_number=excluded.last_number''',
            (key, number),
        )
        con.commit()
        return candidate
    finally:
        con.close()


def source_item_identifier(record_id):
    return f'SRC-INV-{int(record_id):06d}'


def assign_item_identifiers(items, field_name, document_no):
    """Copy item dictionaries and add a stable line identifier for this document."""
    result = []
    for index, item in enumerate(items or [], start=1):
        copied = dict(item or {})
        copied[field_name] = f'{document_no}-ITEM-{index:02d}'
        result.append(copied)
    return result

def configure_tesseract():
    """Locate Tesseract on Windows when it is not already on PATH."""
    env_cmd = os.environ.get('TESSERACT_CMD')
    candidates = [
        env_cmd,
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            return candidate
    return None


def preprocess_ocr_image(image):
    """Improve scanned procurement forms before OCR."""
    img = ImageOps.exif_transpose(image).convert('L')
    if img.width < 1800:
        scale = 1800 / max(img.width, 1)
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    return img


def extract_ocr_text(filepath):
    """Extract text from an AOQ/bidding-result image or PDF."""
    configure_tesseract()
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.pdf':
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError('PDF OCR requires PyMuPDF. Run: pip install -r requirements.txt') from exc
        pages = []
        with fitz.open(filepath) as doc:
            for page_index, page in enumerate(doc):
                if page_index >= 10:
                    break
                native = (page.get_text('text') or '').strip()
                if len(native) >= 20:
                    pages.append(native)
                    continue
                pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
                mode = 'RGB' if pix.n < 4 else 'RGBA'
                image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                pages.append(pytesseract.image_to_string(preprocess_ocr_image(image), config='--psm 6'))
        return '\n\n'.join(x.strip() for x in pages if x and x.strip()).strip()

    with Image.open(filepath) as image:
        return pytesseract.image_to_string(preprocess_ocr_image(image), config='--psm 6').strip()


def _clean_ocr_value(value):
    value = re.sub(r'^[\s:;|\-–—]+', '', str(value or ''))
    value = re.sub(r'\s+', ' ', value).strip()
    return value


def _find_labeled_value(lines, labels):
    label_pattern = '|'.join(re.escape(x) for x in labels)
    pattern = re.compile(rf'(?i)^\s*(?:{label_pattern})\s*(?:[:\-–—]|is)?\s*(.+?)\s*$')
    for idx, line in enumerate(lines):
        match = pattern.match(line)
        if match and _clean_ocr_value(match.group(1)):
            return _clean_ocr_value(match.group(1))
        if re.fullmatch(rf'(?i)\s*(?:{label_pattern})\s*[:\-–—]?\s*', line) and idx + 1 < len(lines):
            return _clean_ocr_value(lines[idx + 1])
    return ''


def _extract_money(value):
    if not value:
        return 0.0
    matches = re.findall(r'(?:₱|PHP|P)?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d{1,2})|[0-9]+(?:\.\d{1,2})?)', str(value), flags=re.I)
    amounts = [to_float(x) for x in matches]
    return max(amounts) if amounts else 0.0


def parse_aoq_ocr_text(text, pr_row=None):
    """Convert OCR output into editable AOQ winning-offer fields."""
    raw_lines = [re.sub(r'\s+', ' ', x).strip() for x in str(text or '').splitlines()]
    lines = [x for x in raw_lines if x]
    joined = '\n'.join(lines)

    supplier = _find_labeled_value(lines, [
        'winning supplier', 'winning bidder', 'awarded supplier', 'awarded to',
        'name of supplier', 'supplier', 'bidder', 'dealer'
    ])
    item = _find_labeled_value(lines, [
        'winning item name', 'item name', 'article', 'item', 'particulars'
    ])
    description = _find_labeled_value(lines, [
        'winning item description / specification', 'description / specification',
        'description/specification', 'item description', 'description', 'specification'
    ])
    qty_text = _find_labeled_value(lines, ['quantity', 'qty'])
    unit = _find_labeled_value(lines, ['unit of measure', 'uom', 'unit'])
    unit_cost_text = _find_labeled_value(lines, ['unit cost', 'unit price', 'bid price'])
    total_text = _find_labeled_value(lines, [
        'total winning amount', 'winning amount', 'total bid amount', 'bid amount',
        'contract amount', 'grand total', 'total amount', 'total'
    ])
    remarks = _find_labeled_value(lines, ['recommendation', 'result', 'remarks', 'award basis'])

    if not supplier:
        m = re.search(r'(?i)(?:awarded\s+to|winning\s+(?:supplier|bidder)\s*(?:is|:|-)?)[\s:,-]*([^\n]{2,100})', joined)
        if m:
            supplier = _clean_ocr_value(m.group(1))

    qty = 1.0
    qty_match = re.search(r'\d+(?:\.\d+)?', qty_text or '')
    if qty_match:
        qty = to_float(qty_match.group(0)) or 1.0

    unit_cost = _extract_money(unit_cost_text)
    total = _extract_money(total_text)
    if total <= 0:
        currency_values = re.findall(r'(?i)(?:₱|PHP|P)\s*([0-9][0-9,]*(?:\.\d{1,2})?)', joined)
        if currency_values:
            total = max(to_float(x) for x in currency_values)
    if total <= 0 and qty and unit_cost:
        total = qty * unit_cost
    if unit_cost <= 0 and total > 0 and qty > 0:
        unit_cost = total / qty

    pr_items = parse_items(pr_row['items']) if pr_row else []
    first_pr_item = pr_items[0] if pr_items else {}
    if not item:
        item = str(value_any(first_pr_item, 'Item Name', 'Item Description', 'item', default='Winning Offer Item'))
    if not description:
        description = get_item_description_for_nlp(first_pr_item, 'pr') if first_pr_item else ''
    if not unit:
        unit = str(value_any(first_pr_item, 'Unit', 'unit', default='unit')) or 'unit'
    if qty == 1.0:
        pr_qty = to_float(value_any(first_pr_item, 'Quantity', 'qty', default=0))
        if pr_qty > 0:
            qty = pr_qty

    return {
        'winner': supplier,
        'item': item,
        'description': description,
        'qty': qty,
        'unit': unit,
        'unit_cost': round(unit_cost, 2),
        'winning_amount': round(total, 2),
        'remarks': remarks or 'Extracted from uploaded AOQ/bidding result through OCR.',
        'raw_text': str(text or '').strip(),
    }


def _aoq_preview_path(filename):
    name = os.path.basename(str(filename or ''))
    return os.path.join(AOQ_UPLOAD_DIR, name) if name else ''


def load_aoq_preview():
    preview_name = os.path.basename(str(session.get('aoq_preview_file') or ''))
    if not preview_name:
        return {}
    path = _aoq_preview_path(preview_name)
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            data['_preview_file'] = preview_name
            return data
    except (OSError, ValueError, TypeError):
        pass
    session.pop('aoq_preview_file', None)
    return {}


def clear_aoq_preview(delete_upload=False):
    preview = load_aoq_preview()
    preview_name = os.path.basename(str(session.pop('aoq_preview_file', '') or ''))
    if preview_name:
        try:
            os.remove(_aoq_preview_path(preview_name))
        except OSError:
            pass
    session.pop('aoq_ocr_data', None)
    if delete_upload:
        stored_name = os.path.basename(str(preview.get('stored_filename') or ''))
        if stored_name:
            try:
                os.remove(os.path.join(AOQ_UPLOAD_DIR, stored_name))
            except OSError:
                pass
    return preview


def save_aoq_preview(preview):
    os.makedirs(AOQ_UPLOAD_DIR, exist_ok=True)
    stored_name = os.path.basename(str(preview.get('stored_filename') or 'aoq'))
    preview_name = f'{stored_name}.preview.json'
    with open(_aoq_preview_path(preview_name), 'w', encoding='utf-8') as fh:
        json.dump(preview, fh, ensure_ascii=False, indent=2, default=str)
    session['aoq_preview_file'] = preview_name
    session.modified = True
    return preview_name


def _posted_list(name):
    return [str(x or '').strip() for x in request.form.getlist(name)]


def posted_aoq_suppliers(preview):
    names = _posted_list('supplier_name')
    amounts = request.form.getlist('supplier_amount')
    remarks = _posted_list('supplier_remarks')
    original = list(preview.get('suppliers') or [])
    suppliers = []
    count = max(len(names), len(amounts), len(remarks), len(original))
    for idx in range(count):
        source = original[idx] if idx < len(original) and isinstance(original[idx], dict) else {}
        name = names[idx] if idx < len(names) else str(source.get('supplier') or '').strip()
        amount = to_float(amounts[idx]) if idx < len(amounts) else to_float(source.get('amount'))
        note = remarks[idx] if idx < len(remarks) else str(source.get('remarks') or '').strip()
        if not name and not amount and not note:
            continue
        suppliers.append({
            'supplier': name or f'Unidentified Supplier {idx + 1}',
            'amount': round(amount, 2),
            'remarks': note,
            'is_winner': False,
            'items': list(source.get('items') or []),
        })
    return suppliers


def posted_aoq_items():
    names = _posted_list('offer_item')
    descriptions = _posted_list('offer_description')
    qtys = request.form.getlist('offer_qty')
    units = _posted_list('offer_unit')
    unit_costs = request.form.getlist('offer_unit_cost')
    amounts = request.form.getlist('offer_amount')
    remarks = _posted_list('offer_remarks')
    count = max(len(names), len(descriptions), len(qtys), len(units), len(unit_costs), len(amounts), len(remarks))
    items = []
    for idx in range(count):
        item = names[idx] if idx < len(names) else ''
        desc = descriptions[idx] if idx < len(descriptions) else ''
        qty = to_float(qtys[idx]) if idx < len(qtys) else 0
        unit = units[idx] if idx < len(units) else ''
        unit_cost = to_float(unit_costs[idx]) if idx < len(unit_costs) else 0
        amount = to_float(amounts[idx]) if idx < len(amounts) else 0
        note = remarks[idx] if idx < len(remarks) else ''
        if not any([item, desc, qty, unit, unit_cost, amount, note]):
            continue
        qty = qty or 1
        unit = unit or 'unit'
        if amount <= 0 and unit_cost > 0:
            amount = qty * unit_cost
        if unit_cost <= 0 and amount > 0 and qty > 0:
            unit_cost = amount / qty
        items.append({
            'item': item or (desc[:120] if desc else 'Quoted Item'),
            'description': desc or item,
            'qty': qty,
            'unit': unit,
            'unit_cost': round(unit_cost, 2),
            'amount': round(amount, 2),
            'remarks': note,
        })
    return items


def _norm_text(value):
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9]+', ' ', str(value or '').lower())).strip()


def parse_json_object(value):
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value or '{}')
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def parse_json_list(value):
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value or '[]')
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def money(v):
    return '₱{:,.2f}'.format(to_float(v))
app.jinja_env.filters['money'] = money

ASUNCION_HEADERS = [
    'General Description and Objective of the Project to be Procured',
    'Type of the Project to be Procured',
    'Quantity and Size of the Project to be Procured',
    'Recommended Mode of Procurement',
    'Pre-Procurement Conference Applicable (Yes/No)',
    'Start of Procurement Activity',
    'End of Procurement Activity',
    'Expected Delivery/Implementation Period',
    'Source of Funds',
    'Estimated Budget Allocation (Php)',
    'Attached Supporting Documents',
    'Remarks'
]

PPMP_DESTINATIONS = ('DBM', 'NON-DBM', 'LIB')
PPMP_DATASET_META_HEADERS = [
    '_Source Item ID',
    '_Dataset Record ID',
    '_Dataset Record Key',
    '_Dataset Source Sheet',
    '_Dataset Source Row',
    '_Dataset Requesting Office',
]


def excel_meta():
    """Return the exact Excel file used by the system.

    This is shown in the PPMP screen so the user can verify they edited the
    correct file. The app ONLY reads this path:
        data/PPMP_Data.xlsx
    """
    exists = os.path.exists(EXCEL_PATH)
    info = {
        'path': EXCEL_PATH,
        'exists': exists,
        'modified': 'File not found',
        'size': 0,
    }
    if exists:
        try:
            stat = os.stat(EXCEL_PATH)
            info['size'] = stat.st_size
            info['modified'] = datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%b %d, %Y %I:%M:%S %p')
        except Exception as e:
            info['modified'] = f'Unable to read modified time: {e}'
    return info


def excel_row_signature(row):
    """Build a simple content signature for an Excel row.

    The database also stores _row for convenience, but row numbers change when
    a user deletes/insert rows in Excel. This signature helps avoid hiding the
    wrong newly-added row after Excel rows shift.
    """
    keys = [
        'General Description and Objective of the Project to be Procured',
        'Type of the Project to be Procured',
        'Quantity and Size of the Project to be Procured',
        'Recommended Mode of Procurement',
        'Source of Funds',
        'Estimated Budget Allocation (Php)',
        'Remarks',
    ]
    return '|'.join(str(row.get(k, '')).strip().lower() for k in keys)

def value_any(row, *keys, default=''):
    """Return the first non-empty value from a dictionary using possible column names."""
    for key in keys:
        val = row.get(key)
        if val not in (None, ''):
            return val
    return default

def normalize_excel_row(row, ptype):
    """Make Asuncion PPMP rows compatible with the rest of the system.

    The connected Excel follows the official Asuncion PPMP columns. This function keeps
    the original columns for printing while adding aliases used by PR, AOQ, PO and NLP.
    """
    general = value_any(row, 'General Description and Objective of the Project to be Procured', 'General Description and Objective', 'Category')
    project_type = value_any(row, 'Type of the Project to be Procured', 'Type of Project', default='Goods' if ptype != 'LIB' else 'Services')
    quantity_size = value_any(row, 'Quantity and Size of the Project to be Procured', 'Quantity and Size', 'Description / Specification', 'Description')
    method = value_any(row, 'Recommended Mode of Procurement', 'Procurement Method')
    preproc = value_any(row, 'Pre-Procurement Conference Applicable (Yes/No)', 'Pre-Procurement Conference', default='No')
    start = value_any(row, 'Start of Procurement Activity', 'Start')
    end = value_any(row, 'End of Procurement Activity', 'End')
    delivery = value_any(row, 'Expected Delivery/Implementation Period', 'Delivery Period')
    funds = value_any(row, 'Source of Funds', default='GF')
    budget = value_any(row, 'Estimated Budget Allocation (Php)', 'Estimated Total Cost', 'Total Cost', default=0)
    docs = value_any(row, 'Attached Supporting Documents', 'Supporting Documents')
    remarks = value_any(row, 'Remarks')

    # item name is derived from the quantity/specification text; this keeps the transaction pages usable
    # while the official print preview still uses the exact PPMP columns.
    item_name = str(quantity_size).split(',')[0].strip() if quantity_size else str(general).strip()
    category = str(general).strip() if general else ptype

    row['PPMP Type'] = ptype
    row['Category'] = category
    row['Item Name'] = item_name
    row['Item Description'] = item_name
    row['Description / Specification'] = str(quantity_size or '')
    row['Description'] = str(quantity_size or '')
    row['Type of Project'] = project_type
    row['Procurement Method'] = method
    row['Pre-Procurement Conference'] = preproc
    row['Start'] = start
    row['End'] = end
    row['Delivery Period'] = delivery
    row['Source of Funds'] = funds
    row['Estimated Total Cost'] = budget or 0
    row['Estimated Unit Cost'] = ''
    row['Quantity'] = ''
    row['Unit'] = ''
    row['Attached Supporting Documents'] = docs
    row['Remarks'] = remarks
    return row




ASUNCION_APP_SCHEDULE = [
    ('Jan. 14-23', 'Jan. 24', 'Jan. 26', 'Jan. 28'),
    ('Feb. 14-23', 'Feb. 21', 'Feb. 23', 'Feb. 25'),
    ('Mar. 1-7', 'Mar. 8', 'Mar. 29', 'Mar. 14'),
    ('April 4-10', 'Apr. 11', 'Apr. 26', 'Apr. 15'),
    ('May 9-15', 'May 16', 'May 28', 'May 20'),
    ('June 6-12', 'Jun. 13', 'Jun. 25', 'Jun. 17'),
    ('July 4-10', 'Jul. 20', 'Jul. 26', 'Jul. 13'),
    ('Aug. 1-7', 'Aug. 12', 'Aug. 30', 'Aug. 12'),
    ('Sept. 2-8', 'Sept. 12', 'Sept. 24', 'Sept. 14'),
    ('Oct. 3-9', 'Oct. 12', 'Oct. 29', 'Oct. 14'),
    ('Nov. 2-8', 'Nov. 11', 'Nov. 26', 'Nov. 14'),
    ('Dec. 5-11', 'Dec. 14', 'Dec. 27', 'Dec. 16'),
]

def is_capital_outlay(text):
    t = str(text or '').lower()
    return any(w in t for w in ['equipment', 'computer', 'printer', 'vehicle', 'furniture', 'aircon', 'air conditioner', 'laptop', 'desktop'])

def build_app_rows(items, office='MMO'):
    """Build one Asuncion APP line for every PPMP item.

    Important process rule:
    If the user selected 2 PPMP Excel rows, the APP printout must show 2
    procurement program/project lines. We intentionally do NOT group items here
    because grouping can make multiple selected PPMP rows look like only one APP
    item.
    """
    app_rows = []
    for idx, it in enumerate(items, start=1):
        program = value_any(
            it,
            'General Description and Objective of the Project to be Procured',
            'Category',
            'Item Name',
            default=f'Procurement Program/Project {idx}'
        )
        mode = value_any(it, 'Recommended Mode of Procurement', 'Procurement Method', default='Alternative Method/Agency to Agency')
        funds = value_any(it, 'Source of Funds', default='Gen. fund')
        remarks = value_any(
            it,
            'Remarks',
            'Quantity and Size of the Project to be Procured',
            'Description / Specification',
            default='Brief description of Program/Project'
        )
        amount = to_float(value_any(it, 'Estimated Budget Allocation (Php)', 'Estimated Total Cost', 'Total Cost', default=0))
        project_type = value_any(it, 'Type of the Project to be Procured', 'Type of Project', default='')
        is_co = is_capital_outlay(program) or is_capital_outlay(remarks) or is_capital_outlay(project_type)
        app_rows.append({
            'code': f'PAP-{idx:03d}',
            'program': str(program).strip(),
            'office': office or 'MMO',
            'mode': str(mode).strip(),
            'funds': str(funds).strip() or 'Gen. fund',
            'total': amount,
            'mooe': 0.0 if is_co else amount,
            'co': amount if is_co else 0.0,
            'remarks': str(remarks).strip(),
            'schedules': ASUNCION_APP_SCHEDULE,
        })
    return app_rows

def item_count_from_json(items_json):
    return len(parse_items(items_json))

def item_preview_from_json(items_json, limit=3):
    names = []
    for it in parse_items(items_json):
        names.append(str(it.get('Item Name') or it.get('Item Description') or it.get('General Description and Objective of the Project to be Procured') or 'Item'))
    if len(names) > limit:
        return ', '.join(names[:limit]) + f' + {len(names) - limit} more'
    return ', '.join(names)



def money_text(value):
    return f'₱{to_float(value):,.2f}'


def review_value(value):
    """Convert stored values into readable text for the read-only review screen."""
    if value is None or value == '':
        return '—'
    if isinstance(value, bool):
        return 'Yes' if value else 'No'
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def review_item_rows(items, document_no='DOC'):
    """Normalize different procurement item schemas for one consistent review table.

    The original item dictionary is also preserved as a complete field list so the
    reviewer can expand a line and verify every stored value before approval.
    """
    rows = []
    id_keys = (
        '_PPMP Item ID', '_APP Item ID', '_PR Item ID', '_AOQ Item ID',
        '_PO Item ID', '_IAR Item ID', '_Issue Item ID', 'item_code'
    )
    known_keys = {
        'Item Name', 'Item Description', 'item', 'name',
        'Description / Specification', 'Description', 'description', 'specification',
        'Quantity', 'qty', 'quantity', 'Quantity and Size of the Project to be Procured',
        'Unit', 'unit', 'Unit of Measure',
        'Unit Cost', 'unit_cost', 'Estimated Unit Cost',
        'Amount', 'amount', 'Total Cost', 'Estimated Total Cost',
        'Estimated Budget Allocation (Php)',
        'Remarks', 'remarks', 'General Description and Objective of the Project to be Procured',
    }
    for index, item in enumerate(items or [], start=1):
        item = dict(item or {})
        line_id = next((str(item.get(k)).strip() for k in id_keys if item.get(k)), '')
        if not line_id:
            line_id = f'{document_no}-ITEM-{index:02d}'
        source_id = str(item.get('_Source Item ID') or '').strip()
        item_name = value_any(
            item, 'Item Name', 'Item Description', 'item', 'name',
            'General Description and Objective of the Project to be Procured',
            default='Item'
        )
        description = value_any(
            item, 'Description / Specification', 'Description', 'description',
            'specification', 'Remarks', 'remarks',
            default=''
        )
        quantity = value_any(
            item, 'Quantity', 'qty', 'quantity',
            'Quantity and Size of the Project to be Procured', default='—'
        )
        unit = value_any(item, 'Unit', 'unit', 'Unit of Measure', default='—')
        unit_cost_raw = value_any(item, 'Unit Cost', 'unit_cost', 'Estimated Unit Cost', default='')
        amount_raw = value_any(
            item, 'Amount', 'amount', 'Total Cost', 'Estimated Total Cost',
            'Estimated Budget Allocation (Php)', default=''
        )
        unit_cost = money_text(unit_cost_raw) if unit_cost_raw not in ('', None) else '—'
        amount = money_text(amount_raw) if amount_raw not in ('', None) else '—'
        details = []
        for key, value in item.items():
            if value in (None, '', [], {}):
                continue
            label = str(key).lstrip('_').replace('_', ' ').strip()
            details.append({'label': label, 'value': review_value(value), 'is_primary': key in known_keys})
        rows.append({
            'index': index,
            'line_id': line_id,
            'source_id': source_id or '—',
            'item': review_value(item_name),
            'description': review_value(description),
            'quantity': review_value(quantity),
            'unit': review_value(unit),
            'unit_cost': unit_cost,
            'amount': amount,
            'details': details,
        })
    return rows


def review_section(title, items, document_no, note=''):
    return {
        'title': title,
        'note': note,
        'items': review_item_rows(items, document_no),
        'count': len(items or []),
    }


def render_review_page(*, active, title, document_type, document_no, status,
                       created_at='', linked_records=None, summary=None,
                       item_sections=None, text_blocks=None, approve_url=None,
                       approve_label='', approval_effect='', back_url=None,
                       print_url=None, can_approve=False):
    return render_template(
        'document_review.html', active=active, title=title,
        document_type=document_type, document_no=document_no,
        status=status or 'Unknown', created_at=created_at,
        linked_records=linked_records or [], summary=summary or [],
        item_sections=item_sections or [], text_blocks=text_blocks or [],
        approve_url=approve_url, approve_label=approve_label,
        approval_effect=approval_effect, back_url=back_url,
        print_url=print_url, can_approve=bool(can_approve),
    )


def require_review_confirmation(review_endpoint, record_id):
    """Block approval unless it was submitted from the read-only review page."""
    if request.form.get('review_confirm') != 'yes':
        flash('Open the review page and confirm that all items were checked before approving.')
        return redirect(url_for(review_endpoint, id=record_id))
    return None

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = conn()
    con.executescript('''
    create table if not exists ppmp(id integer primary key, no text, type text, office text, year integer, status text, created_at text, items text, total real);
    create table if not exists app_plan(id integer primary key, no text, ppmp_id integer, year integer, office text, status text, created_at text, total real);
    create table if not exists pr(id integer primary key, no text, app_id integer, ppmp_id integer, office text, purpose text, status text, created_at text, items text, total real);
    create table if not exists aoq(id integer primary key, no text, pr_id integer, status text, created_at text, suppliers text, winner text, winning_amount real, winning_offer text, ocr_filename text, ocr_text text);
    create table if not exists po(id integer primary key, no text, aoq_id integer, supplier text, status text, created_at text, items text, total real);
    create table if not exists nlp(id integer primary key, po_id integer, result text, confidence real, staff_decision text, remarks text, created_at text, details text, status text);
    create table if not exists iar(id integer primary key, no text, po_id integer, status text, created_at text, items text);
    create table if not exists inventory(id integer primary key, item text, description text, category text, qty real, unit text, source_po text, status text, created_at text);
    create table if not exists ris_ics(id integer primary key, no text, type text, office text, status text, created_at text, items text);
    create table if not exists audit_trail(id integer primary key, dt text, user text, action text, module text, ref text, details text);
    create table if not exists users(id integer primary key autoincrement, username text unique not null, password text not null, role text not null default 'Admin');
    create table if not exists identifier_sequence(sequence_key text primary key, last_number integer not null default 0);
    create table if not exists inventory_transactions(id integer primary key autoincrement, inventory_id integer, document_no text, transaction_type text, qty real, unit text, office text, created_at text, created_by text);
    ''')
    # Only accounts stored in the users table can access the system.
    # Seed the single administrator account if the database has no users yet.
    existing_user = con.execute('select count(*) as c from users').fetchone()['c']
    if existing_user == 0:
        con.execute(
            'insert into users(username, password, role) values (?, ?, ?)',
            ('admin', generate_password_hash('admin123'), 'Admin')
        )
    # Safely migrate legacy plaintext passwords to Werkzeug password hashes.
    for existing in con.execute('select id,password from users').fetchall():
        stored = str(existing['password'] or '')
        if not (stored.startswith('scrypt:') or stored.startswith('pbkdf2:')):
            con.execute('update users set password=? where id=?', (generate_password_hash(stored or 'changeme'), existing['id']))
    # Migration for connected item-level APP records. Older databases may have
    # app_plan without an items column, which caused APP to appear as only one
    # saved item even when the PPMP contained multiple selected Excel rows.
    cols = [r[1] for r in con.execute('pragma table_info(app_plan)').fetchall()]
    if 'items' not in cols:
        con.execute('alter table app_plan add column items text')
    pr_cols = [r[1] for r in con.execute('pragma table_info(pr)').fetchall()]
    if 'ppmp_id' not in pr_cols:
        con.execute('alter table pr add column ppmp_id integer')
    aoq_cols = [r[1] for r in con.execute('pragma table_info(aoq)').fetchall()]
    if 'ocr_filename' not in aoq_cols:
        con.execute('alter table aoq add column ocr_filename text')
    if 'ocr_text' not in aoq_cols:
        con.execute('alter table aoq add column ocr_text text')
    for name, definition in [
        ('stored_filename', 'text'),
        ('source_type', 'text'),
        ('extraction_method', 'text'),
        ('extraction_profile', 'text'),
        ('extraction_warnings', 'text'),
    ]:
        if name not in aoq_cols:
            con.execute(f'alter table aoq add column {name} {definition}')
    nlp_cols = [r[1] for r in con.execute('pragma table_info(nlp)').fetchall()]
    if 'status' not in nlp_cols:
        con.execute('alter table nlp add column status text')
        con.execute('update nlp set status="For Review" where status is null or status=""')
    if 'no' not in nlp_cols:
        con.execute('alter table nlp add column no text')
    for legacy in con.execute('select id from nlp where no is null or trim(no)=""').fetchall():
        candidate = f'NLP-{current_year()}-{int(legacy["id"]):04d}'
        suffix = 1
        while con.execute('select 1 from nlp where no=? and id<>?', (candidate, legacy['id'])).fetchone():
            candidate = f'NLP-{current_year()}-{int(legacy["id"]):04d}-{suffix}'
            suffix += 1
        con.execute('update nlp set no=? where id=?', (candidate, legacy['id']))

    # Dynamic inventory-source fields. Inventory records are references used to
    # prepare PPMP items; no anomaly decision is made at this stage.
    ppmp_cols = [r[1] for r in con.execute('pragma table_info(ppmp)').fetchall()]
    if 'source_dataset_record_id' not in ppmp_cols:
        con.execute('alter table ppmp add column source_dataset_record_id integer')

    inventory_cols = [r[1] for r in con.execute('pragma table_info(inventory)').fetchall()]
    for name, definition in [
        ('unit_cost', 'real'),
        ('source_dataset_record_id', 'integer'),
        ('item_code', 'text'),
    ]:
        if name not in inventory_cols:
            con.execute(f'alter table inventory add column {name} {definition}')
    for legacy in con.execute('select id from inventory where item_code is null or trim(item_code)=""').fetchall():
        candidate = f'INV-{current_year()}-{int(legacy["id"]):05d}'
        suffix = 1
        while con.execute('select 1 from inventory where item_code=? and id<>?', (candidate, legacy['id'])).fetchone():
            candidate = f'INV-{current_year()}-{int(legacy["id"]):05d}-{suffix}'
            suffix += 1
        con.execute('update inventory set item_code=? where id=?', (candidate, legacy['id']))
    con.execute('create index if not exists idx_nlp_no on nlp(no)')
    con.execute('create index if not exists idx_inventory_item_code on inventory(item_code)')

    # Backfill stable line identifiers for records created by older builds.
    for table_name, json_column, item_field in [
        ('ppmp', 'items', '_PPMP Item ID'),
        ('app_plan', 'items', '_APP Item ID'),
        ('pr', 'items', '_PR Item ID'),
        ('aoq', 'winning_offer', '_AOQ Item ID'),
        ('po', 'items', '_PO Item ID'),
        ('iar', 'items', '_IAR Item ID'),
        ('ris_ics', 'items', '_Issue Item ID'),
    ]:
        for doc in con.execute(f'select id,no,{json_column} from {table_name} where {json_column} is not null').fetchall():
            try:
                parsed = json.loads(doc[json_column] or '[]')
            except Exception:
                parsed = []
            if not isinstance(parsed, list):
                continue
            changed = False
            for index, item in enumerate(parsed, start=1):
                if not isinstance(item, dict):
                    continue
                if not item.get(item_field):
                    item[item_field] = f'{doc["no"]}-ITEM-{index:02d}'
                    changed = True
                dataset_id = item.get('_Dataset Record ID')
                if dataset_id and not item.get('_Source Item ID'):
                    try:
                        item['_Source Item ID'] = source_item_identifier(dataset_id)
                        changed = True
                    except Exception:
                        pass
            if changed:
                con.execute(f'update {table_name} set {json_column}=? where id=?', (json.dumps(parsed, default=str), doc['id']))

    # Clear values created by older builds that scored anomalies in PPMP or
    # Inventory. The single trained discrepancy check now runs only in the
    # PR-to-PO NLP Verification module.
    for table in ('ppmp', 'inventory'):
        cols = [r[1] for r in con.execute(f'pragma table_info({table})').fetchall()]
        clear_cols = [c for c in ('anomaly_score','anomaly_label','anomaly_reasons','anomaly_review_status','anomaly_model_version') if c in cols]
        if clear_cols:
            con.execute(f"update {table} set " + ','.join(f'{c}=null' for c in clear_cols))

    con.commit(); con.close()

    ensure_inventory_dataset_schema(DB_PATH)
    # The bundled inventory workbook is synchronized automatically whenever
    # its sheets or rows change. This only updates the item source repository.
    if os.path.exists(COA_DATASET_PATH):
        try:
            sync_inventory_dataset_workbook(
                DB_PATH, COA_DATASET_PATH,
                display_filename=os.path.basename(COA_DATASET_PATH), force=False
            )
        except Exception as exc:
            print(f'Inventory dataset initialization warning: {exc}')

def read_excel(sheet):
    """Read DBM, NON-DBM or LIB from data/PPMP_Data.xlsx.

    Compatible with the Asuncion PPMP Excel format shown in the official form.
    Row 1 must contain the PPMP columns. Personnel/signatories are intentionally
    excluded from Excel and are injected only in print preview.
    """
    if not os.path.exists(EXCEL_PATH):
        return []

    wb = load_workbook(EXCEL_PATH, data_only=True, read_only=True)
    try:
        # Match DBM, NON-DBM, or LIB even if the Excel sheet casing/spaces differ.
        requested = str(sheet or '').strip().lower()
        sheet_name = next((name for name in wb.sheetnames if str(name).strip().lower() == requested), wb.sheetnames[0])
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return []

        headers = [str(h).strip() if h else '' for h in rows[0]]
        data = []
        for i, r in enumerate(rows[1:], start=2):
            if not any(r):
                continue
            raw = {headers[j]: r[j] for j in range(min(len(headers), len(r))) if headers[j]}
            raw['_row'] = i
            raw = normalize_excel_row(raw, sheet_name)
            raw['_excel_sheet'] = sheet_name
            raw['_excel_signature'] = excel_row_signature(raw)
            data.append(raw)
        return data
    finally:
        wb.close()


def ppmp_has_downstream_app(ppmp_id):
    return q('select id from app_plan where ppmp_id=? limit 1', (ppmp_id,), one=True) is not None


def sync_draft_ppmp_from_excel(ptype):
    """Synchronize Draft PPMP records with the current Excel file.

    Why this is needed:
    - The PPMP screen hides Excel rows that were already selected.
    - If the user edits/deletes those rows in Excel, the old saved PPMP JSON
      would stay unchanged unless we update it here.
    - Only Draft PPMP records with no generated APP are changed. Records that
      already moved to APP/PR/PO are preserved for audit trail consistency.
    """
    excel_rows = read_excel(ptype)
    by_row = {str(r.get('_row')): r for r in excel_rows}
    by_sig = {str(r.get('_excel_signature')): r for r in excel_rows if r.get('_excel_signature')}
    updated = deleted = 0
    for rec in q('select * from ppmp where type=? and status="Draft" order by id', (ptype,)):
        if ppmp_has_downstream_app(rec['id']):
            continue
        old_items = parse_items(rec['items'])
        new_items = []
        missing = False
        for old in old_items:
            row_key = str(old.get('_row', ''))
            sig_key = str(old.get('_excel_signature', ''))
            fresh = by_row.get(row_key) or by_sig.get(sig_key)
            if fresh:
                new_items.append(fresh)
            else:
                missing = True
        if missing and not new_items:
            exec_db('delete from ppmp where id=?', (rec['id'],))
            deleted += 1
        elif new_items:
            new_total = sum(to_float(x.get('Estimated Budget Allocation (Php)') or x.get('Estimated Total Cost') or x.get('Total Cost') or 0) for x in new_items)
            if json.dumps(old_items, default=str, sort_keys=True) != json.dumps(new_items, default=str, sort_keys=True) or to_float(rec['total']) != new_total:
                exec_db('update ppmp set items=?, total=? where id=?', (json.dumps(new_items, default=str), new_total, rec['id']))
                updated += 1
    return {'updated': updated, 'deleted': deleted, 'excel_rows': len(excel_rows)}


def used_excel_rows_for_ppmp(ptype):
    """Return Excel rows/signatures already converted into PPMP.

    Current records save both _row and _excel_signature. We mainly use row numbers
    to keep selected items hidden. This function is recalculated every request,
    so editing the Excel file then pressing Refresh re-reads the current workbook.
    """
    used_rows = set()
    used_signatures = set()
    for row in q('select items from ppmp where type=?', (ptype,)):
        for item in parse_items(row['items']):
            if item.get('_row') not in (None, ''):
                used_rows.add(str(item.get('_row')))
            if item.get('_excel_signature'):
                used_signatures.add(str(item.get('_excel_signature')))
    return used_rows, used_signatures


def available_excel_rows(ptype):
    """Read Excel and hide rows already saved to PPMP."""
    rows = read_excel(ptype)
    used_rows, used_signatures = used_excel_rows_for_ppmp(ptype)
    available = []
    for r in rows:
        row_key = str(r.get('_row'))
        sig_key = str(r.get('_excel_signature', ''))
        if row_key in used_rows or (sig_key and sig_key in used_signatures):
            continue
        available.append(r)
    return available, used_rows, len(rows)


def parse_items(items_json):
    try: return json.loads(items_json or '[]')
    except: return []

def next_available_records():
    """Return only records that are allowed for the next connected process.
    This prevents users from creating disconnected duplicate documents.
    """
    ppmps = q('''select p.* from ppmp p
                 left join app_plan a on a.ppmp_id=p.id
                 where a.id is null
                 order by p.id desc''')
    apps = q('''select a.* from app_plan a
                left join pr r on r.app_id=a.id
                where a.status="Approved" and r.id is null
                order by a.id desc''')
    prs = q('''select r.* from pr r
               left join aoq aq on aq.pr_id=r.id
               where r.status="Approved" and aq.id is null
               order by r.id desc''')
    aoqs = q('''select aq.* from aoq aq
                left join po p on p.aoq_id=aq.id
                where aq.status="Approved" and p.id is null
                order by aq.id desc''')
    pos_for_nlp = q('''select p.* from po p
                       left join nlp n on n.po_id=p.id
                       where p.status="Approved" and n.id is null
                       order by p.id desc''')
    pos_for_iar = q('''select p.* from po p
                       join nlp n on n.po_id=p.id
                       left join iar i on i.po_id=p.id
                       where i.id is null
                         and n.status="Approved"
                         and n.result in ("Acceptable","Needs Review")
                       order by p.id desc''')
    return ppmps, apps, prs, aoqs, pos_for_nlp, pos_for_iar

def trace_for_po(po_row):
    if not po_row: return {}
    aoq_row = q('select * from aoq where id=?',(po_row['aoq_id'],),one=True)
    pr_row = q('select * from pr where id=?',(aoq_row['pr_id'],),one=True) if aoq_row else None
    app_row = q('select * from app_plan where id=?',(pr_row['app_id'],),one=True) if pr_row and pr_row['app_id'] else None
    ppmp_row = None
    if pr_row and 'ppmp_id' in pr_row.keys() and pr_row['ppmp_id']:
        ppmp_row = q('select * from ppmp where id=?',(pr_row['ppmp_id'],),one=True)
    elif app_row:
        ppmp_row = q('select * from ppmp where id=?',(app_row['ppmp_id'],),one=True)
    return {'ppmp':ppmp_row,'app':app_row,'pr':pr_row,'aoq':aoq_row}

def get_item_description_for_nlp(item, side='pr'):
    """Return the complete natural-language description/specification."""
    if not item:
        return ''
    preferred = [
        'Description / Specification', 'Item Description', 'Description', 'description',
        'Quantity and Size of the Project to be Procured', 'Item Name', 'item', 'Remarks',
    ]
    values = []
    seen = set()
    for key in preferred:
        value = item.get(key)
        text = re.sub(r'\s+', ' ', str(value or '')).strip()
        if text and text.lower() not in seen:
            values.append(text)
            seen.add(text.lower())
    # Preserve other meaningful specification fields instead of forcing a fixed
    # procurement vocabulary. The trained FastText model receives the original wording.
    ignored = {
        '_row', '_source', '_source_record_id', '_source_item_id', '_source_sheet',
        '_PR Item ID', '_PO Item ID', '_AOQ Item ID', '_IAR Item ID',
        'Quantity', 'quantity', 'qty', 'QTY', 'Requested Quantity', 'Awarded Quantity',
        'Unit', 'unit', 'Unit of Measure', 'unit_of_measure', 'uom',
    }
    for key, value in (item or {}).items():
        if key in preferred or key in ignored or str(key).startswith('_'):
            continue
        text = re.sub(r'\s+', ' ', str(value or '')).strip()
        if not text or text.lower() in seen:
            continue
        key_text = re.sub(r'[_\-]+', ' ', str(key)).strip()
        values.append(f'{key_text}: {text}')
        seen.add(text.lower())
    return ' | '.join(values)


def get_item_quantity_for_nlp(item):
    """Read quantity as text input without deciding whether it is acceptable."""
    if not item:
        return 'not provided'
    for key in ('Quantity', 'quantity', 'qty', 'QTY', 'Requested Quantity', 'Awarded Quantity'):
        value = item.get(key)
        if value not in (None, ''):
            return re.sub(r'\s+', ' ', str(value)).strip()
    return 'not provided'


def get_item_unit_for_nlp(item):
    """Read the original unit wording; synonym understanding belongs to the model."""
    if not item:
        return 'not provided'
    for key in ('Unit', 'unit', 'Unit of Measure', 'unit_of_measure', 'uom'):
        value = item.get(key)
        if value not in (None, ''):
            return re.sub(r'\s+', ' ', str(value)).strip()
    return 'not provided'


def serialize_nlp_items(items, side):
    """Serialize the complete approved document as a natural-language model input."""
    document_name = 'PURCHASE REQUEST' if side == 'pr' else 'PURCHASE ORDER'
    parts = [document_name]
    for index, item in enumerate(items or [], start=1):
        description = get_item_description_for_nlp(item, side) or 'description not provided'
        quantity = get_item_quantity_for_nlp(item)
        unit = get_item_unit_for_nlp(item)
        parts.append(
            f'ITEM {index}. DESCRIPTION AND SPECIFICATION: {description}. '
            f'QUANTITY: {quantity}. UNIT: {unit}.'
        )
    if len(parts) == 1:
        parts.append('NO ITEM INFORMATION WAS PROVIDED.')
    return '\n'.join(parts)


def build_pr_text_for_trained_nlp(items):
    return serialize_nlp_items(items, 'pr')


def build_po_text_for_trained_nlp(items):
    return serialize_nlp_items(items, 'po')


def run_trained_nlp_prediction(pr_text, po_text):
    """Run the Colab-trained AI-only FastText model."""
    from fasttext_nlp_runtime import predict_pair
    return predict_pair(NLP_FASTTEXT_DIR, pr_text, po_text)


def nlp_predict(pr_items, po_items):
    """Return a direct trained-model PR-to-PO prediction.

    There is no hardcoded discrepancy rule, similarity score, hybrid decision,
    threshold-to-class conversion, or fallback model. If the exported Colab
    model is unavailable, verification stops and no NLP record is created.
    """
    pr_text = build_pr_text_for_trained_nlp(pr_items)
    po_text = build_po_text_for_trained_nlp(po_items)
    prediction = run_trained_nlp_prediction(pr_text, po_text)
    manifest = prediction.get('model_manifest') or {}
    details = {
        'model': manifest.get('deployed_model') or 'FastText + Logistic Regression',
        'architecture': manifest.get('architecture') or 'Pair-aware FastText embeddings with trained status and issue classifiers',
        'decision_mode': 'AI_ONLY_FASTTEXT_DUAL_CLASSIFIER',
        'final_decision_source': 'Direct prediction from the trained FastText status classifier',
        'trained_model_used': True,
        'runtime_rules_used': False,
        'similarity_score_used': False,
        'hybrid_score_used': False,
        'rule_fallback_used': False,
        'manual_threshold_used': False,
        'trained_status_label': prediction['status'],
        'trained_status_confidence': prediction['status_confidence'],
        'trained_status_probabilities': prediction['status_probabilities'],
        'trained_primary_issue': prediction['issue'],
        'trained_issue_confidence': prediction['issue_confidence'],
        'trained_issue_probabilities': prediction['issue_probabilities'],
        'pr_token_count': prediction.get('pr_token_count'),
        'po_token_count': prediction.get('po_token_count'),
        'feature_dimension': prediction.get('feature_dimension'),
        'model_metadata': manifest,
        'model_path': NLP_FASTTEXT_DIR,
        'pr_text': pr_text,
        'po_text': po_text,
    }
    return prediction['status'], prediction['status_confidence'], details


def nlp_short_finding(result, details):
    """Display the trained issue classifier's direct prediction."""
    try:
        if isinstance(details, str):
            details = json.loads(details or '{}')
    except Exception:
        details = {}
    issue = str((details or {}).get('trained_primary_issue') or '').strip()
    confidence = to_float((details or {}).get('trained_issue_confidence'))
    if issue:
        return f'Trained issue classifier predicted: {issue} ({confidence:.2f}% model probability).'
    return f'Trained status classifier predicted {result or "a result"}.'


def nlp_dynamic_remarks(result, details=None):
    """Show model outputs without adding a second rule-based NLP decision."""
    return (
        f'Trained AI status: {result or "Unavailable"}. '
        f'{nlp_short_finding(result, details)} Review the full PR and PO before approving the recorded result.'
    )


def trained_nlp_model_status():
    """Validate the exported FastText model without loading a fallback."""
    try:
        from fasttext_nlp_runtime import model_status
        return model_status(NLP_FASTTEXT_DIR)
    except Exception as error:
        return {'active': False, 'message': f'Unable to inspect FastText model: {error}'}


def algorithm_comparison():
    """Show the June three-model comparison and the deployed FastText model."""
    model_state = trained_nlp_model_status()
    active = bool(model_state.get('active'))
    manifest = model_state.get('manifest') or {}
    comparison = manifest.get('model_comparison') or []
    comparison_by_name = {
        str(row.get('model') or ''): row
        for row in comparison
        if isinstance(row, dict)
    }

    def metric_text(model_name):
        row = comparison_by_name.get(model_name) or {}
        if not row:
            return 'Evaluation unavailable until the Colab model is installed'
        return (
            f"Status Macro F1 {to_float(row.get('status_macro_f1'))*100.0:.2f}% | "
            f"Status Accuracy {to_float(row.get('status_accuracy'))*100.0:.2f}% | "
            f"Issue Macro F1 {to_float(row.get('issue_macro_f1'))*100.0:.2f}%"
        )

    selected = manifest.get('deployed_model') or 'FastText + Logistic Regression'
    return [
        {
            'name': 'TF-IDF + Logistic Regression',
            'type': 'Word/character TF-IDF representation with trained status and issue classifiers',
            'role': 'June comparison baseline',
            'status': 'Compared in Colab' if comparison else 'Awaiting Colab comparison',
            'accuracy': metric_text('TF-IDF + Logistic Regression'),
            'notes': 'Evaluated using the same group-held-out PR–PO records. It is retained for model-comparison evidence but is not used during runtime verification.',
        },
        {
            'name': 'FastText + Logistic Regression',
            'type': 'Pair-aware subword embeddings with trained status and issue classifiers',
            'role': 'Selected and deployed NLP model',
            'status': 'Active' if active and selected == 'FastText + Logistic Regression' else ('Selected; model not installed' if not active else 'Installed'),
            'accuracy': metric_text('FastText + Logistic Regression'),
            'notes': 'The June-selected approach is retained and enhanced using semantic requirement wording, local COA terminology, OCR-like variations, and optional staff-validated pairs. It directly predicts the workflow status and primary issue; no hardcoded or hybrid runtime decision overrides it.',
        },
        {
            'name': 'DistilBERT Embeddings + Logistic Regression',
            'type': 'Frozen DistilBERT sentence-pair embeddings with trained status and issue classifiers',
            'role': 'June transformer comparison approach',
            'status': 'Compared in Colab' if comparison else 'Awaiting Colab comparison',
            'accuracy': metric_text('DistilBERT Embeddings + Logistic Regression'),
            'notes': 'DistilBERT is used as a frozen embedding extractor, matching the original June methodology. It is evaluated but not loaded by the offline production runtime.',
        },
    ]


# ---------------------------
# Editable PPMP Table Save Route
# ---------------------------
# Editable PPMP Table Save Route
@app.route('/ppmp/<ptype>/edit_table', methods=['POST'])
def ppmp_edit_table(ptype):
    """Save PPMP table/modal edits back to Excel and Draft PPMP records.

    The PPMP page displays available rows directly from data/PPMP_Data.xlsx.
    Modal edits therefore must update the Excel row itself, not only a saved
    PPMP database record. Any Draft PPMP JSON that still points to the same
    Excel row is also refreshed so the next APP/PR/AOQ/PO/IAR flow stays intact.
    """
    selected_rows = [str(x) for x in request.form.getlist('rows') if str(x).strip()]
    if not selected_rows:
        flash('Select at least one PPMP item before saving changes.')
        return redirect(url_for('ppmp_type', ptype=ptype))

    field_map = {
        'general': 'General Description and Objective of the Project to be Procured',
        'type': 'Type of the Project to be Procured',
        'quantity': 'Quantity and Size of the Project to be Procured',
        'mode': 'Recommended Mode of Procurement',
        'preproc': 'Pre-Procurement Conference Applicable (Yes/No)',
        'start': 'Start of Procurement Activity',
        'end': 'End of Procurement Activity',
        'delivery': 'Expected Delivery/Implementation Period',
        'source': 'Source of Funds',
        'budget': 'Estimated Budget Allocation (Php)',
        'docs': 'Attached Supporting Documents',
        'remarks': 'Remarks',
    }

    edited_rows = {}
    for key, value in request.form.items():
        if '_' not in key:
            continue
        col, row_id = key.rsplit('_', 1)
        row_id = str(row_id)
        if row_id in selected_rows and col in field_map:
            edited_rows.setdefault(row_id, {})[field_map[col]] = value

    if not edited_rows:
        flash('No editable fields were submitted. Open Edit Table, change the selected item, then click Save Changes.')
        return redirect(url_for('ppmp_type', ptype=ptype))

    # Save edits to the real Excel sheet used by PPMP.
    try:
        wb = load_workbook(EXCEL_PATH)
        requested = str(ptype or '').strip().lower()
        sheet_name = next((name for name in wb.sheetnames if str(name).strip().lower() == requested), None)
        if not sheet_name:
            wb.close()
            flash(f'Excel sheet for {ptype} was not found.')
            return redirect(url_for('ppmp_type', ptype=ptype))
        ws = wb[sheet_name]
        headers = [str(c.value).strip() if c.value else '' for c in ws[1]]
        header_to_col = {h: idx + 1 for idx, h in enumerate(headers) if h}
        for row_id, fields in edited_rows.items():
            excel_row = int(row_id)
            for header, value in fields.items():
                col_index = header_to_col.get(header)
                if col_index:
                    ws.cell(row=excel_row, column=col_index).value = to_float(value) if header == 'Estimated Budget Allocation (Php)' else value
        wb.save(EXCEL_PATH)
        wb.close()
    except Exception as e:
        flash(f'PPMP Excel save failed: {e}')
        return redirect(url_for('ppmp_type', ptype=ptype))

    # Refresh any existing Draft PPMP records that came from the edited rows.
    fresh_rows = {str(r.get('_row')): r for r in read_excel(ptype)}
    updated_db = 0
    for rec in q('select * from ppmp where type=? and status="Draft"', (ptype,)):
        items = parse_items(rec['items'])
        changed = False
        for idx, item in enumerate(items):
            row_key = str(item.get('_row', ''))
            if row_key in edited_rows and row_key in fresh_rows:
                items[idx] = fresh_rows[row_key]
                changed = True
        if changed:
            new_total = sum(to_float(x.get('Estimated Budget Allocation (Php)') or x.get('Estimated Total Cost') or x.get('Total Cost') or 0) for x in items)
            exec_db('update ppmp set items=?, total=? where id=?', (json.dumps(items, default=str), new_total, rec['id']))
            updated_db += 1

    # NLP training is intentionally not performed inside the deployed system.
    # Train and validate in Google Colab, then install the exported model folder.
    flash('Saved successfully.')

    return redirect(url_for('ppmp_type', ptype=ptype))

# ---------------------------
# OCR Upload Route
# ---------------------------
# OCR Upload Route
@app.route('/ppmp/<ptype>/upload_ocr', methods=['POST'])
def ppmp_ocr_upload(ptype):
    file = request.files.get('ocr_file')
    if not file:
        flash('No file uploaded.')
        return redirect(url_for('ppmp_type', ptype=ptype))
    
    filename = secure_filename(file.filename)
    upload_dir = os.path.join(BASE_DIR, 'uploads', 'ppmp')
    os.makedirs(upload_dir, exist_ok=True)
    filepath = os.path.join(upload_dir, filename)
    file.save(filepath)

    try:
        text = pytesseract.image_to_string(Image.open(filepath))
        session['ocr_text'] = text
        flash('OCR completed.')
    except Exception as e:
        flash(f'OCR failed: {e}')

    return redirect(url_for('ppmp_type', ptype=ptype))

_DATABASE_READY = False

@app.before_request
def ensure():
    global _DATABASE_READY
    if not _DATABASE_READY:
        init_db()
        _DATABASE_READY = True
    allowed = {'login', 'static'}
    if request.endpoint in allowed:
        return None
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return None

@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = request.form.get('password') or ''
        user = q('select * from users where username=?', (username,), one=True)
        valid = False
        if user:
            stored = str(user['password'] or '')
            try:
                valid = check_password_hash(stored, password)
            except Exception:
                valid = stored == password
            if valid and not (stored.startswith('scrypt:') or stored.startswith('pbkdf2:')):
                exec_db('update users set password=? where id=?', (generate_password_hash(password), user['id']))
        if valid:
            session.clear()
            session['logged_in'] = True
            session['username'] = user['username']
            session['role'] = user['role']
            flash('Login successful.')
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.')
    return render_template('login.html', title='Login')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.')
    return redirect(url_for('login'))

@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def dashboard():
    stats = {
        'ppmp': q('select count(*) c from ppmp', one=True)['c'],
        'app': q('select count(*) c from app_plan', one=True)['c'],
        'pr': q('select count(*) c from pr', one=True)['c'],
        'aoq': q('select count(*) c from aoq', one=True)['c'],
        'po': q('select count(*) c from po', one=True)['c'],
        'nlp': q('select count(*) c from nlp', one=True)['c'],
        'iar': q('select count(*) c from iar', one=True)['c'],
        'inventory': q('select count(*) c from inventory', one=True)['c'],
        'ris_ics': q('select count(*) c from ris_ics', one=True)['c'],
    }
    pending = {
        'app': q('select count(*) c from app_plan where status<>"Approved"', one=True)['c'],
        'pr': q('select count(*) c from pr where status<>"Approved"', one=True)['c'],
        'aoq': q('select count(*) c from aoq where status<>"Approved"', one=True)['c'],
        'po': q('select count(*) c from po where status<>"Approved"', one=True)['c'],
        'nlp': q('select count(*) c from nlp where coalesce(status,"")<>"Approved"', one=True)['c'],
        'iar': q('select count(*) c from iar where status<>"Approved"', one=True)['c'],
    }
    recent = q('select * from audit_trail order by id desc limit 10')
    model_state = trained_nlp_model_status()
    return render_template('dashboard.html', active='dashboard', stats=stats, pending=pending,
                           recent=recent, model_state=model_state, title='Dashboard')

@app.route('/excel-status')
def excel_status():
    meta = excel_meta()
    sheets = []
    error = ''
    if meta['exists']:
        try:
            wb = load_workbook(EXCEL_PATH, read_only=True, data_only=True)
            sheets = wb.sheetnames
            wb.close()
        except Exception as e:
            error = str(e)
    return {'excel_path': meta['path'], 'exists': meta['exists'], 'modified': meta['modified'], 'size': meta['size'], 'sheets': sheets, 'error': error}

def parse_json_list(value):
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value or '[]')
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def dataset_reference_unit_cost(record):
    value = to_float(record.get('unit_value'))
    qty = to_float(record.get('balance_qty')) or to_float(record.get('on_hand_qty'))
    return round(value / qty, 2) if value > 0 and qty > 1 else round(value, 2)


def sync_active_coa_dataset(force=False, display_filename=None):
    if not os.path.exists(COA_DATASET_PATH):
        return None
    return sync_inventory_dataset_workbook(
        DB_PATH,
        COA_DATASET_PATH,
        display_filename=display_filename or os.path.basename(COA_DATASET_PATH),
        force=force,
    )

def dataset_record_to_ppmp_item(record, requested_qty, estimated_unit_cost, ptype):
    '''Convert one inventory-source row into official PPMP columns.'''
    ptype = str(ptype or '').strip().upper()
    if ptype not in PPMP_DESTINATIONS:
        raise ValueError('PPMP destination must be DBM, NON-DBM, or LIB.')
    requested_qty = max(to_float(requested_qty), 1)
    estimated_unit_cost = max(to_float(estimated_unit_cost), 0)
    unit = str(record.get('unit') or 'unit').strip() or 'unit'
    description = str(record.get('description') or record.get('article') or 'Inventory item').strip()
    article = str(record.get('article') or 'Goods').strip()
    total = requested_qty * estimated_unit_cost
    source_note = f"Inventory source: {record.get('sheet_name')} row {record.get('source_row')}"
    procurement_method = {
        'DBM': 'Agency to Agency',
        'NON-DBM': 'Small Value Procurement',
        'LIB': 'Direct Contracting',
    }[ptype]
    project_type = 'Services' if ptype == 'LIB' else 'Goods'
    return {
        'General Description and Objective of the Project to be Procured': description,
        'Type of the Project to be Procured': project_type,
        'Quantity and Size of the Project to be Procured': f'{requested_qty:g} {unit} - {description}',
        'Recommended Mode of Procurement': procurement_method,
        'Pre-Procurement Conference Applicable (Yes/No)': 'No',
        'Start of Procurement Activity': '',
        'End of Procurement Activity': '',
        'Expected Delivery/Implementation Period': '',
        'Source of Funds': 'General Fund',
        'Estimated Budget Allocation (Php)': round(total, 2),
        'Attached Supporting Documents': 'Inventory reference dataset',
        'Remarks': source_note,
        'PPMP Type': ptype,
        'Category': article,
        'Item Name': description,
        'Item Description': description,
        'Description / Specification': description,
        'Description': description,
        'Quantity': requested_qty,
        'Unit': unit,
        'Estimated Unit Cost': round(estimated_unit_cost, 2),
        'Unit Cost': round(estimated_unit_cost, 2),
        'Estimated Total Cost': round(total, 2),
        'Historical Unit Value': to_float(record.get('unit_value')),
        'Historical Balance Quantity': to_float(record.get('balance_qty')),
        'Historical On Hand Quantity': to_float(record.get('on_hand_qty')),
        'Property Number': record.get('property_number') or '',
        'Office': record.get('office') or record.get('remarks') or '',
        'Person Accountable': record.get('accountable_person') or '',
        '_Source Item ID': source_item_identifier(record.get('id')),
        '_Dataset Record ID': record.get('id'),
        '_Dataset Record Key': record.get('record_key') or '',
        '_Dataset Source Sheet': record.get('sheet_name') or '',
        '_Dataset Source Row': record.get('source_row') or '',
    }

def append_dataset_items_to_ppmp_excel(ptype, items, requesting_office):
    """Append selected inventory items to the chosen PPMP worksheet.

    Sheet names and row positions are resolved at runtime. Metadata columns are
    added once so the inventory source remains traceable when the user
    later generates the PPMP record from the selected category page.
    """
    ptype = str(ptype or '').strip().upper()
    if ptype not in PPMP_DESTINATIONS:
        raise ValueError('Choose DBM, NON-DBM, or LIB as the PPMP destination.')
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f'PPMP workbook was not found: {EXCEL_PATH}')

    wb = load_workbook(EXCEL_PATH)
    try:
        requested = ptype.lower()
        sheet_name = next((name for name in wb.sheetnames if str(name).strip().lower() == requested), None)
        if not sheet_name:
            raise ValueError(f'The {ptype} worksheet was not found in PPMP_Data.xlsx.')
        ws = wb[sheet_name]

        headers = [str(cell.value).strip() if cell.value else '' for cell in ws[1]]
        for header in ASUNCION_HEADERS + PPMP_DATASET_META_HEADERS:
            if header not in headers:
                headers.append(header)
                ws.cell(row=1, column=len(headers), value=header)
        header_index = {header: idx + 1 for idx, header in enumerate(headers) if header}

        key_col = header_index['_Dataset Record Key']
        id_col = header_index['_Dataset Record ID']
        existing_keys = set()
        existing_ids = set()
        for row_no in range(2, ws.max_row + 1):
            key = ws.cell(row=row_no, column=key_col).value
            record_id = ws.cell(row=row_no, column=id_col).value
            if key not in (None, ''):
                existing_keys.add(str(key).strip())
            if record_id not in (None, ''):
                existing_ids.add(str(record_id).strip())

        added = 0
        skipped = 0
        added_rows = []
        for item in items:
            key = str(item.get('_Dataset Record Key') or '').strip()
            record_id = str(item.get('_Dataset Record ID') or '').strip()
            if (key and key in existing_keys) or (record_id and record_id in existing_ids):
                skipped += 1
                continue
            row_no = ws.max_row + 1
            item['_Dataset Requesting Office'] = requesting_office
            for header, col_no in header_index.items():
                value = item.get(header, '')
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, default=str)
                ws.cell(row=row_no, column=col_no, value=value)
            if key:
                existing_keys.add(key)
            if record_id:
                existing_ids.add(record_id)
            added += 1
            added_rows.append(row_no)

        if added:
            wb.save(EXCEL_PATH)
        return {'added': added, 'skipped': skipped, 'rows': added_rows, 'sheet': sheet_name}
    finally:
        wb.close()


@app.route('/ppmp/dataset', methods=['GET', 'POST'])
def ppmp_dataset():
    '''Dynamic multi-sheet inventory source for the existing PPMP workflow.'''
    os.makedirs(os.path.dirname(COA_DATASET_PATH), exist_ok=True)
    if request.method == 'POST':
        action = request.form.get('action') or ''

        if action == 'upload_dataset':
            uploaded = request.files.get('dataset_file')
            if not uploaded or not uploaded.filename:
                flash('Choose an Excel workbook to upload.')
                return redirect(url_for('ppmp_dataset'))
            original_name = secure_filename(uploaded.filename)
            ext = os.path.splitext(original_name)[1].lower()
            if ext not in ALLOWED_DATASET_EXTENSIONS:
                flash('Upload an .xlsx or .xlsm workbook.')
                return redirect(url_for('ppmp_dataset'))
            temp_path = COA_DATASET_PATH + '.uploading' + ext
            backup_path = COA_DATASET_PATH + '.backup'
            uploaded.save(temp_path)
            if os.path.exists(COA_DATASET_PATH):
                shutil.copy2(COA_DATASET_PATH, backup_path)
            try:
                os.replace(temp_path, COA_DATASET_PATH)
                summary = sync_active_coa_dataset(force=True, display_filename=original_name)
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                log('Imported Inventory Dataset', 'PPMP Inventory Source', original_name,
                    f'{summary["sheet_count"]} sheet(s), {summary["row_count"]} valid row(s)')
                flash(f'Inventory dataset imported: {summary["sheet_count"]} sheet(s) and {summary["row_count"]} valid row(s) were read automatically.')
            except Exception as exc:
                if os.path.exists(backup_path):
                    os.replace(backup_path, COA_DATASET_PATH)
                elif os.path.exists(COA_DATASET_PATH):
                    os.remove(COA_DATASET_PATH)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                flash(f'Dataset import failed: {exc}')
            return redirect(url_for('ppmp_dataset'))

        if action == 'refresh_dataset':
            try:
                summary = sync_active_coa_dataset(force=True)
                flash(f'Inventory source refreshed: {summary["row_count"]} active rows from {summary["sheet_count"]} sheet(s).')
                log('Refreshed Inventory Dataset', 'PPMP Inventory Source', summary['filename'],
                    f'Rows {summary["row_count"]}; sheets {summary["sheet_count"]}')
            except Exception as exc:
                flash(f'Unable to refresh the inventory dataset: {exc}')
            return redirect(url_for('ppmp_dataset'))

        destination_actions = {
            'pass_dbm': 'DBM',
            'pass_non_dbm': 'NON-DBM',
            'pass_lib': 'LIB',
        }
        if action in destination_actions or action == 'generate_ppmp':
            destination = destination_actions.get(action) or str(request.form.get('ppmp_destination') or '').strip().upper()
            if destination not in PPMP_DESTINATIONS:
                flash('Choose where the selected inventory items should appear: DBM, NON-DBM, or LIB.')
                return redirect(request.referrer or url_for('ppmp_dataset'))
            selected = [int(x) for x in request.form.getlist('record_ids') if str(x).isdigit()]
            if not selected:
                flash(f'Select at least one inventory item to display in {destination}.')
                return redirect(request.referrer or url_for('ppmp_dataset'))
            office = (request.form.get('requesting_office') or 'General Services Office').strip()
            items = []
            source_details = []
            for record_id in selected:
                row = q('select * from dataset_records where id=? and active=1', (record_id,), one=True)
                if not row:
                    continue
                record = dict(row)
                requested_qty = to_float(request.form.get(f'qty_{record_id}')) or 1
                estimated_unit_cost = to_float(request.form.get(f'cost_{record_id}'))
                if estimated_unit_cost <= 0:
                    estimated_unit_cost = dataset_reference_unit_cost(record)
                items.append(dataset_record_to_ppmp_item(
                    record, requested_qty, estimated_unit_cost, destination
                ))
                source_details.append(f'{record["sheet_name"]} row {record["source_row"]}')

            if not items:
                flash('The selected inventory records are no longer active. Refresh the dataset and try again.')
                return redirect(url_for('ppmp_dataset'))
            try:
                result = append_dataset_items_to_ppmp_excel(destination, items, office)
            except Exception as exc:
                flash(f'Unable to add the inventory items to {destination}: {exc}')
                return redirect(url_for('ppmp_dataset'))

            if result['added']:
                log('Passed Inventory Items to PPMP Category', 'PPMP Inventory Source', destination,
                    f'{result["added"]} item(s) added to {destination}; sources: {", ".join(source_details)}')
            message = f'{result["added"]} selected inventory item(s) added to the {destination} PPMP list.'
            if result['skipped']:
                message += f' {result["skipped"]} duplicate item(s) were already in that category and were skipped.'
            message += ' Review the displayed items, select them, then click Generate PPMP.'
            flash(message)
            return redirect(url_for('ppmp_type', ptype=destination))

    auto_sync_message = None
    if os.path.exists(COA_DATASET_PATH):
        try:
            sync_summary = sync_active_coa_dataset(force=False)
            if sync_summary and sync_summary.get('changed'):
                auto_sync_message = f'Workbook changes detected: {sync_summary["row_count"]} rows synchronized.'
        except Exception as exc:
            auto_sync_message = f'Automatic dataset synchronization failed: {exc}'

    status = get_inventory_dataset_status(DB_PATH)
    selected_sheet = (request.args.get('sheet') or '').strip()
    search_text = (request.args.get('q') or '').strip()
    try:
        page = max(int(request.args.get('page') or 1), 1)
    except Exception:
        page = 1
    per_page = 100
    where = ['active=1']
    params = []
    if selected_sheet:
        where.append('sheet_name=?')
        params.append(selected_sheet)
    if search_text:
        like = f'%{search_text.lower()}%'
        where.append('''(
            lower(coalesce(article,'')) like ? or lower(coalesce(description,'')) like ? or
            lower(coalesce(property_number,'')) like ? or lower(coalesce(unit,'')) like ? or
            lower(coalesce(remarks,'')) like ? or lower(coalesce(office,'')) like ? or
            lower(coalesce(accountable_person,'')) like ? or lower(coalesce(raw_data,'')) like ?
        )''')
        params.extend([like] * 8)
    where_sql = ' and '.join(where)
    total = q(f'select count(*) c from dataset_records where {where_sql}', tuple(params), one=True)['c']
    offset = (page - 1) * per_page
    raw_rows = q(
        f'''select * from dataset_records where {where_sql}
        order by sheet_name collate nocase, source_row
        limit ? offset ?''',
        tuple(params + [per_page, offset])
    )
    records = []
    for row in raw_rows:
        d = dict(row)
        d['reference_unit_cost'] = dataset_reference_unit_cost(d)
        d['source_id'] = source_item_identifier(d['id'])
        records.append(d)
    page_count = max((total + per_page - 1) // per_page, 1)
    return render_template(
        'ppmp_dataset.html', active='ppmp', records=records, dataset_status=status,
        selected_sheet=selected_sheet, search_text=search_text,
        page=page, page_count=page_count, total=total,
        auto_sync_message=auto_sync_message, title='Inventory Item Source'
    )


@app.route('/ppmp')
def ppmp_home():
    raw_rows=q('select * from ppmp order by id desc')
    rows=[]
    for r in raw_rows:
        d=dict(r)
        d['item_count']=item_count_from_json(d.get('items'))
        d['item_preview']=item_preview_from_json(d.get('items'), limit=2)
        d['items_list']=parse_items(d.get('items'))
        rows.append(d)
    dataset_status = get_inventory_dataset_status(DB_PATH)
    return render_template('ppmp_home.html', active='ppmp', rows=rows, dataset_status=dataset_status, title='PPMP')

@app.route('/ppmp/<ptype>', methods=['GET','POST'])
def ppmp_type(ptype):
    # Always read Excel from disk, then hide rows that are already saved into PPMP.
    # This keeps the process clean: Excel item -> PPMP -> APP -> PR -> AOQ -> PO.
    refresh = request.args.get('refresh') == '1'
    sync_info = {'updated': 0, 'deleted': 0, 'excel_rows': 0}
    if refresh:
        sync_info = sync_draft_ppmp_from_excel(ptype)
    data, used_rows, total_excel_rows = available_excel_rows(ptype)
    meta = excel_meta()
    # Refresh silently reloads the Excel data without showing a long status message.
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'ocr':
            return redirect(url_for('ppmp_ocr_upload', ptype=ptype))
        elif action == 'generate_ppmp':
            # Your original logic for creating new PPMP from selected Excel rows
            selected = request.form.getlist('rows')
            selected_set = {str(x) for x in selected}
            items = [x for x in data if str(x.get('_row')) in selected_set]
            if not items:
                flash('Select at least one available item from Excel. Already-selected items are hidden and cannot be selected again.')
                return redirect(url_for('ppmp_type', ptype=ptype))
            # ... continue your original PPMP generation code here ...
        elif action == 'save_table':
            return redirect(url_for('ppmp_edit_table', ptype=ptype))

        # IMPORTANT CONNECTED-PROCESS FIX:
        # Each selected Excel row becomes its own PPMP record.
        # This makes the visible PPMP list, APP dropdown, and the next process match
        # what the user selected. If the user selects 2 Excel rows, the system saves
        # 2 PPMP records, then APP can generate 2 connected APP records.
        created_nos = []
        for item in items:
            item_total = to_float(item.get('Estimated Budget Allocation (Php)') or item.get('Estimated Total Cost') or item.get('Total Cost') or 0)
            row_no = item.get('_row')
            year = current_year()
            ppmp_prefix = f'PPMP-{ptype.replace("-", "")}'
            no = next_document_no(f'PPMP-{ptype}', ppmp_prefix, 'ppmp', year=year)
            source_dataset_record_id = int(to_float(item.get('_Dataset Record ID'))) or None
            item_office = str(item.get('_Dataset Requesting Office') or 'General Services Office').strip()
            ppmp_items = assign_item_identifiers([item], '_PPMP Item ID', no)
            exec_db(
                '''insert into ppmp(no,type,office,year,status,created_at,items,total,source_dataset_record_id)
                values(?,?,?,?,?,?,?,?,?)''',
                (no, ptype, item_office, year, 'Draft', now(), json.dumps(ppmp_items, default=str), item_total,
                 source_dataset_record_id)
            )
            created_nos.append(no)
            source_text = 'COA inventory dataset' if source_dataset_record_id else f'{ptype} Excel'
            log('Created PPMP', 'PPMP', no, f'1 item from {source_text}; category {ptype}; Excel row: {row_no}')
        flash('PPMP generated successfully.')
        return redirect(url_for('ppmp_home'))
    return render_template('ppmp_type.html', active='ppmp', ptype=ptype, data=data, used_count=len(used_rows), total_excel_rows=total_excel_rows, title=f'PPMP - {ptype}', excel=os.path.basename(EXCEL_PATH), excel_meta=meta)

@app.route('/ppmp/<int:id>/items', methods=['POST'])
def ppmp_update_saved_items(id):
    row = q('select * from ppmp where id=?', (id,), one=True)
    if not row:
        flash('PPMP record not found.')
        return redirect(url_for('ppmp_home'))
    items = parse_items(row['items'])
    if not items:
        flash('No PPMP items found to update.')
        return redirect(url_for('ppmp_home'))
    try:
        selected_index = int(request.form.get('item_index') or 0)
    except Exception:
        selected_index = 0
    if selected_index < 0 or selected_index >= len(items):
        selected_index = 0
    field_map = {
        'general': 'General Description and Objective of the Project to be Procured',
        'type': 'Type of the Project to be Procured',
        'quantity': 'Quantity and Size of the Project to be Procured',
        'mode': 'Recommended Mode of Procurement',
        'preproc': 'Pre-Procurement Conference Applicable (Yes/No)',
        'start': 'Start of Procurement Activity',
        'end': 'End of Procurement Activity',
        'delivery': 'Expected Delivery/Implementation Period',
        'source': 'Source of Funds',
        'budget': 'Estimated Budget Allocation (Php)',
        'docs': 'Attached Supporting Documents',
        'remarks': 'Remarks',
    }
    for form_key, item_key in field_map.items():
        if form_key in request.form:
            value = request.form.get(form_key)
            items[selected_index][item_key] = to_float(value) if form_key == 'budget' else value
    items[selected_index] = normalize_excel_row(items[selected_index], row['type'])
    new_total = sum(to_float(x.get('Estimated Budget Allocation (Php)') or x.get('Estimated Total Cost') or 0) for x in items)
    exec_db('update ppmp set items=?, total=? where id=?', (json.dumps(items, default=str), new_total, id))
    log('Updated PPMP Item', 'PPMP', row['no'], f'Updated saved PPMP item {selected_index + 1}')
    flash('PPMP item updated successfully.')
    return redirect(url_for('ppmp_home'))

@app.route('/ppmp/<int:id>/print')
def ppmp_print(id):
    row=q('select * from ppmp where id=?',(id,),one=True); items=parse_items(row['items'])
    return render_template('print_ppmp.html', row=row, items=items, sig=SIGNATORIES, title='Print PPMP')

@app.route('/app', methods=['GET','POST'])
def app_plan():
    if request.method=='POST':
        pid = request.form.get('ppmp_id')
        p = q('select * from ppmp where id=?', (pid,), one=True)
        if p:
            items = parse_items(p['items'])
            if not items:
                flash('This PPMP has no saved items, so APP cannot be generated.')
                return redirect(url_for('app_plan'))

            # IMPORTANT FIX:
            # APP is item-level. If PPMP has 2 selected Excel rows, create 2 APP rows.
            # This prevents the next processes from carrying only the first item.
            created = 0
            created_nos = []
            for item in items:
                item_total = to_float(value_any(item, 'Estimated Budget Allocation (Php)', 'Estimated Total Cost', 'Total Cost', default=0))
                app_no = next_document_no('APP', 'APP', 'app_plan', year=p['year'] or current_year())
                app_items = assign_item_identifiers([item], '_APP Item ID', app_no)
                exec_db(
                    'insert into app_plan(no,ppmp_id,year,office,status,created_at,total,items) values(?,?,?,?,?,?,?,?)',
                    (app_no, p['id'], p['year'], p['office'], 'For Review', now(), item_total, json.dumps(app_items, default=str))
                )
                created_nos.append(app_no)
                created += 1

            exec_db('update ppmp set status=? where id=?', ('For APP', pid))
            log('Generated APP', 'APP', ', '.join(created_nos), f'from {p["no"]}; {created} APP item row(s) created from {len(items)} selected PPMP item(s)')
            flash('APP generated successfully.')
        return redirect(url_for('app_plan'))

    raw_plans = q("""select a.*, p.no ppmp_no, p.type ppmp_type, p.items ppmp_items, a.items app_items
                     from app_plan a
                     left join ppmp p on p.id=a.ppmp_id
                     order by a.id desc""")
    plans = []
    for r in raw_plans:
        d = dict(r)
        source_items = d.get('app_items') or d.get('ppmp_items')
        d['item_count'] = item_count_from_json(source_items)
        d['item_preview'] = item_preview_from_json(source_items)
        plans.append(d)
    raw_ppmps, _, _, _, _, _ = next_available_records()
    ppmps = []
    for p in raw_ppmps:
        d = dict(p)
        d['item_count'] = item_count_from_json(d.get('items'))
        d['item_preview'] = item_preview_from_json(d.get('items'), limit=1)
        ppmps.append(d)
    return render_template('app.html', active='app', plans=plans, ppmps=ppmps, title='APP')

@app.route('/app/<int:id>/view')
def app_review(id):
    row = q('''select a.*, p.no ppmp_no, p.type ppmp_type, p.items ppmp_items
               from app_plan a left join ppmp p on p.id=a.ppmp_id
               where a.id=?''', (id,), one=True)
    if not row:
        flash('APP record not found.')
        return redirect(url_for('app_plan'))
    d = dict(row)
    app_items = parse_items(d.get('items') or d.get('ppmp_items'))
    ppmp_items = parse_items(d.get('ppmp_items'))
    sections = [review_section('APP Items for Approval', app_items, d['no'],
                               'Verify every procurement item, amount, schedule, and source information.')]
    if ppmp_items and d.get('items'):
        sections.append(review_section('Source PPMP Items', ppmp_items, d.get('ppmp_no') or 'PPMP',
                                       'Read-only source used to create this APP.'))
    return render_review_page(
        active='app', title=f'Review APP {d["no"]}', document_type='Annual Procurement Plan',
        document_no=d['no'], status=d['status'], created_at=d.get('created_at'),
        linked_records=[
            {'label': 'Source PPMP', 'value': d.get('ppmp_no') or 'No linked PPMP'},
            {'label': 'PPMP Category', 'value': d.get('ppmp_type') or 'Not recorded'},
        ],
        summary=[
            {'label': 'Office', 'value': d.get('office') or '—'},
            {'label': 'Fiscal Year', 'value': d.get('year') or '—'},
            {'label': 'Number of Items', 'value': len(app_items)},
            {'label': 'APP Total', 'value': money_text(d.get('total'))},
        ],
        item_sections=sections,
        approve_url=url_for('app_approve', id=id),
        approve_label=f'Approve APP {d["no"]}',
        approval_effect='Approval finalizes this APP and makes its planning record officially approved.',
        back_url=url_for('app_plan'), print_url=url_for('app_print', id=id),
        can_approve=d.get('status') != 'Approved',
    )

@app.route('/app/<int:id>/approve', methods=['POST'])
def app_approve(id):
    row=q('select * from app_plan where id=?',(id,),one=True)
    if not row:
        flash('APP record not found.')
        return redirect(url_for('app_plan'))
    guard = require_review_confirmation('app_review', id)
    if guard:
        return guard
    if row['status'] == 'Approved':
        flash('APP is already approved.')
        return redirect(url_for('app_review', id=id))
    exec_db('update app_plan set status=? where id=?',('Approved',id))
    if row['ppmp_id']:
        exec_db('update ppmp set status=? where id=?', ('APP Approved', row['ppmp_id']))
    log('Approved APP','APP',row['no'],'APP approved after full item review')
    flash(f'APP {row["no"]} approved after reviewing all items.')
    return redirect(url_for('app_plan'))

@app.route('/app/<int:id>/print')
def app_print(id):
    row = q('select * from app_plan where id=?', (id,), one=True)
    p = q('select * from ppmp where id=?', (row['ppmp_id'],), one=True) if row else None
    # Use the APP row's own item first. Older records without app_plan.items fall back to PPMP items.
    items = parse_items(row['items']) if row and 'items' in row.keys() and row['items'] else (parse_items(p['items']) if p else [])
    app_rows = build_app_rows(items, row['office'] if row else 'MMO')
    return render_template('print_app.html', row=row, items=items, app_rows=app_rows, sig=SIGNATORIES, title='Print APP')

@app.route('/pr', methods=['GET','POST'])
def pr():
    if request.method == 'POST':
        app_id = request.form.get('app_id')
        app_row = q('''select a.*, p.no ppmp_no, p.type ppmp_type, p.items ppmp_items
                       from app_plan a left join ppmp p on p.id=a.ppmp_id
                       where a.id=? and a.status="Approved"
                         and not exists (select 1 from pr r where r.app_id=a.id)''', (app_id,), one=True)
        if not app_row:
            flash('Select an approved APP record that does not already have a Purchase Request.')
            return redirect(url_for('pr'))
        items = parse_items(app_row['items'] or app_row['ppmp_items'])
        if not items:
            flash('No approved APP items were found for the selected record.')
            return redirect(url_for('pr'))
        no = next_document_no('PR', 'PR', 'pr', year=app_row['year'] or current_year())
        pr_items = assign_item_identifiers(items, '_PR Item ID', no)
        total = sum(to_float(value_any(item, 'Estimated Budget Allocation (Php)', 'Estimated Total Cost', 'Total Cost', 'amount', default=0)) for item in items)
        if total <= 0:
            total = to_float(app_row['total'])
        purpose = (request.form.get('purpose') or 'Procurement of approved supplies, equipment, and services.').strip()
        exec_db('''insert into pr(no,app_id,ppmp_id,office,purpose,status,created_at,items,total)
                   values(?,?,?,?,?,?,?,?,?)''',
                (no, app_row['id'], app_row['ppmp_id'], app_row['office'], purpose,
                 'For Review', now(), json.dumps(pr_items, default=str), total))
        exec_db('update ppmp set status=? where id=?', ('For PR Review', app_row['ppmp_id']))
        log('Created PR', 'PR', no, f'Created from approved APP {app_row["no"]}; source PPMP {app_row["ppmp_no"]}; {len(items)} item(s)')
        flash(f'Purchase Request {no} created from approved APP {app_row["no"]}.')
        return redirect(url_for('pr'))

    raw_prs = q('''select r.*, p.no ppmp_no, p.type ppmp_type, a.no app_no
                   from pr r
                   left join ppmp p on p.id=r.ppmp_id
                   left join app_plan a on a.id=r.app_id
                   order by r.id desc''')
    prs = []
    for row in raw_prs:
        d = dict(row)
        d['item_preview'] = item_preview_from_json(d.get('items'), limit=2)
        d['item_count'] = item_count_from_json(d.get('items'))
        prs.append(d)

    _, raw_apps, _, _, _, _ = next_available_records()
    apps = []
    for row in raw_apps:
        d = dict(row)
        source = d.get('items')
        d['item_preview'] = item_preview_from_json(source, limit=2)
        d['item_count'] = item_count_from_json(source)
        ppmp_row = q('select no,type from ppmp where id=?', (d.get('ppmp_id'),), one=True)
        d['ppmp_no'] = ppmp_row['no'] if ppmp_row else 'No linked PPMP'
        d['ppmp_type'] = ppmp_row['type'] if ppmp_row else ''
        apps.append(d)
    return render_template('pr.html', active='pr', prs=prs, apps=apps, title='PR')

@app.route('/pr/<int:id>/view')
def pr_review(id):
    row = q('''select r.*, p.no ppmp_no, p.type ppmp_type, p.items ppmp_items,
                      a.no app_no
               from pr r
               left join ppmp p on p.id=r.ppmp_id
               left join app_plan a on a.id=r.app_id
               where r.id=?''', (id,), one=True)
    if not row:
        flash('PR record not found.')
        return redirect(url_for('pr'))
    d = dict(row)
    pr_items = parse_items(d.get('items'))
    source_items = parse_items(d.get('ppmp_items'))
    sections = [review_section('Purchase Request Items', pr_items, d['no'],
                               'These are the final requirements that will be used for bidding and NLP comparison.')]
    if source_items:
        sections.append(review_section('Source PPMP Items', source_items, d.get('ppmp_no') or 'PPMP',
                                       'Confirm that the PR still follows its selected PPMP source.'))
    return render_review_page(
        active='pr', title=f'Review PR {d["no"]}', document_type='Purchase Request',
        document_no=d['no'], status=d['status'], created_at=d.get('created_at'),
        linked_records=[
            {'label': 'Source PPMP', 'value': d.get('ppmp_no') or 'No linked PPMP'},
            {'label': 'Related APP', 'value': d.get('app_no') or 'Not linked'},
            {'label': 'PPMP Category', 'value': d.get('ppmp_type') or 'Not recorded'},
        ],
        summary=[
            {'label': 'Requesting Office', 'value': d.get('office') or '—'},
            {'label': 'Purpose', 'value': d.get('purpose') or '—'},
            {'label': 'Number of Items', 'value': len(pr_items)},
            {'label': 'Requested Total', 'value': money_text(d.get('total'))},
        ],
        item_sections=sections,
        approve_url=url_for('pr_approve', id=id), approve_label=f'Approve PR {d["no"]}',
        approval_effect='Approval locks this PR as the official requirement and makes it available for AOQ bidding-result upload.',
        back_url=url_for('pr'), print_url=url_for('pr_print', id=id),
        can_approve=d.get('status') != 'Approved',
    )

@app.route('/pr/<int:id>/approve', methods=['POST'])
def pr_approve(id):
    row = q('select * from pr where id=?', (id,), one=True)
    if not row:
        flash('PR record not found.')
        return redirect(url_for('pr'))
    guard = require_review_confirmation('pr_review', id)
    if guard:
        return guard
    if row['status'] == 'Approved':
        flash('PR is already approved.')
        return redirect(url_for('pr_review', id=id))
    exec_db('update pr set status=? where id=?', ('Approved', id))
    if row['ppmp_id']:
        exec_db('update ppmp set status=? where id=?', ('PR Approved', row['ppmp_id']))
    log('Approved PR', 'PR', row['no'], 'Purchase Request approved')
    flash('PR approved. Continue to AOQ to upload and review the bidding result.')
    return redirect(url_for('aoq'))

@app.route('/pr/<int:id>/print')
def pr_print(id):
    row=q('select * from pr where id=?',(id,),one=True); items=parse_items(row['items'])
    return render_template('print_pr.html', row=row, items=items, sig=SIGNATORIES, title='Print PR')

@app.route('/aoq', methods=['GET','POST'])
def aoq():
    # Layout-flexible AOQ ingestion. The source document may use labels, rows,
    # supplier columns, multiple sheets, or scanned pages. Every extraction is
    # normalized into an editable review payload before it becomes an AOQ.
    if request.method == 'POST':
        action = request.form.get('action') or 'extract_ocr'

        if action == 'clear_ocr':
            clear_aoq_preview(delete_upload=True)
            flash('AOQ extraction preview and temporary upload cleared.')
            return redirect(url_for('aoq'))

        if action == 'extract_ocr':
            pr_id = request.form.get('pr_id')
            pr_row = q('''select r.* from pr r where r.id=? and r.status="Approved"
                        and not exists (select 1 from aoq aq where aq.pr_id=r.id)''', (pr_id,), one=True)
            if not pr_row:
                flash('Select an approved PR before uploading the AOQ or bidding result.')
                return redirect(url_for('aoq'))

            uploaded = request.files.get('ocr_file')
            if not uploaded or not uploaded.filename:
                flash('Choose an AOQ or bidding-result document to upload.')
                return redirect(url_for('aoq'))

            original_name = secure_filename(uploaded.filename)
            ext = os.path.splitext(original_name)[1].lower()
            if ext not in ALLOWED_OCR_EXTENSIONS:
                supported = ', '.join(sorted(x.upper().lstrip('.') for x in ALLOWED_OCR_EXTENSIONS))
                flash(f'Unsupported AOQ file. Supported formats: {supported}.')
                return redirect(url_for('aoq'))

            clear_aoq_preview(delete_upload=True)
            os.makedirs(AOQ_UPLOAD_DIR, exist_ok=True)
            stored_name = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_{random.randint(1000,9999)}{ext}'
            filepath = os.path.join(AOQ_UPLOAD_DIR, stored_name)
            uploaded.save(filepath)

            try:
                document = extract_aoq_document(filepath)
                preview = build_flexible_aoq_preview(document, parse_items(pr_row['items']))
                preview.update({
                    'pr_id': int(pr_row['id']),
                    'pr_no': pr_row['no'],
                    'ocr_filename': original_name,
                    'stored_filename': stored_name,
                })
                save_aoq_preview(preview)
                supplier_count = len(preview.get('suppliers') or [])
                item_count = len(preview.get('offer_items') or [])
                flash(
                    f'{preview.get("document_type", "AOQ document")} extracted using '
                    f'{preview.get("extraction_method", "automatic detection")}. '
                    f'Review {supplier_count} supplier(s) and {item_count} winning-offer item(s) before saving.'
                )
            except Exception as exc:
                try:
                    os.remove(filepath)
                except OSError:
                    pass
                flash(f'AOQ document extraction failed: {exc}')
            return redirect(url_for('aoq'))

        if action == 'save_ocr':
            preview = load_aoq_preview()
            pr_id = preview.get('pr_id') or request.form.get('pr_id')
            pr_row = q('''select r.* from pr r where r.id=? and r.status="Approved"
                        and not exists (select 1 from aoq aq where aq.pr_id=r.id)''', (pr_id,), one=True)
            if not preview or not pr_row:
                flash('Upload and extract an AOQ or bidding-result document first.')
                return redirect(url_for('aoq'))

            suppliers = posted_aoq_suppliers(preview)
            offer_items = posted_aoq_items()
            try:
                winner_index = int(request.form.get('winner_index') or -1)
            except ValueError:
                winner_index = -1
            custom_winner = (request.form.get('winner_custom') or '').strip()
            general_remarks = (request.form.get('remarks') or '').strip()

            if not offer_items:
                flash('Add at least one reviewed winning-offer item before saving the AOQ.')
                return redirect(url_for('aoq'))

            if 0 <= winner_index < len(suppliers):
                winner = suppliers[winner_index]['supplier']
            elif custom_winner:
                winner = custom_winner
            elif len(suppliers) == 1:
                winner_index = 0
                winner = suppliers[0]['supplier']
            else:
                winner = ''
            if not winner:
                flash('Select or enter the official winning supplier before saving.')
                return redirect(url_for('aoq'))

            winning_amount = round(sum(to_float(item.get('amount')) for item in offer_items), 2)
            if winning_amount <= 0 and 0 <= winner_index < len(suppliers):
                winning_amount = round(to_float(suppliers[winner_index].get('amount')), 2)
            if winning_amount <= 0:
                flash('Enter the winning unit costs or amounts before saving the AOQ.')
                return redirect(url_for('aoq'))

            matched = False
            for idx, supplier in enumerate(suppliers):
                supplier['is_winner'] = (idx == winner_index) or (_norm_text(supplier.get('supplier')) == _norm_text(winner))
                if supplier['is_winner']:
                    supplier['items'] = offer_items
                    supplier['amount'] = winning_amount
                    matched = True
            if not matched:
                suppliers.append({
                    'supplier': winner,
                    'amount': winning_amount,
                    'remarks': general_remarks,
                    'is_winner': True,
                    'items': offer_items,
                })

            no = next_document_no('AOQ', 'AOQ', 'aoq')
            offer_items = assign_item_identifiers(offer_items, '_AOQ Item ID', no)
            for supplier in suppliers:
                if supplier.get('is_winner'):
                    supplier['items'] = offer_items

            profile = {
                'document_type': preview.get('document_type'),
                'extraction_method': preview.get('extraction_method'),
                'extraction_confidence': preview.get('confidence'),
                'table_count': preview.get('table_count'),
                'parsed_table_sources': preview.get('parsed_table_sources') or [],
                'detected_tables': preview.get('detected_tables') or [],
            }
            warnings = preview.get('warnings') or []
            exec_db(
                '''insert into aoq(
                       no,pr_id,status,created_at,suppliers,winner,winning_amount,winning_offer,
                       ocr_filename,ocr_text,stored_filename,source_type,extraction_method,
                       extraction_profile,extraction_warnings
                   ) values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (
                    no, pr_row['id'], 'For Review', now(), json.dumps(suppliers, default=str),
                    winner, winning_amount, json.dumps(offer_items, default=str),
                    preview.get('ocr_filename', ''), preview.get('raw_text', ''),
                    preview.get('stored_filename', ''), preview.get('document_type', ''),
                    preview.get('extraction_method', ''), json.dumps(profile, ensure_ascii=False, default=str),
                    json.dumps(warnings, ensure_ascii=False, default=str),
                )
            )
            log(
                'Created flexible AOQ extraction', 'AOQ', no,
                f'File: {preview.get("ocr_filename", "")}; type: {preview.get("document_type", "")}; '
                f'winner: {winner}; suppliers: {len(suppliers)}; items: {len(offer_items)}'
            )
            clear_aoq_preview(delete_upload=False)
            flash(f'AOQ {no} saved from the reviewed flexible extraction. Open its viewing mode before approval.')
            return redirect(url_for('aoq'))

    raw_aoq_rows = q('''select aq.*, r.no pr_no, r.office pr_office from aoq aq
                left join pr r on r.id=aq.pr_id
                order by aq.id desc''')
    rows=[]
    for aq in raw_aoq_rows:
        d=dict(aq)
        d['item_preview']=item_preview_from_json(d.get('winning_offer'),limit=2)
        rows.append(d)
    _, _, prs, _, _, _ = next_available_records()
    pr_previews = []
    for r in prs:
        d = dict(r)
        d['item_preview'] = item_preview_from_json(d.get('items'), limit=2)
        pr_previews.append(d)
    ocr_data = load_aoq_preview()
    selected_pr = None
    if ocr_data and ocr_data.get('pr_id'):
        selected_pr_row = q('select * from pr where id=?', (ocr_data.get('pr_id'),), one=True)
        if selected_pr_row:
            selected_pr = dict(selected_pr_row)
            selected_pr['items_list'] = parse_items(selected_pr.get('items'))
    return render_template(
        'aoq.html', active='aoq', rows=rows, prs=pr_previews,
        ocr_data=ocr_data or None, selected_pr=selected_pr, title='AOQ',
        supported_aoq_formats=', '.join(sorted(x.upper().lstrip('.') for x in ALLOWED_OCR_EXTENSIONS)),
    )


@app.route('/aoq/<int:id>/view')
def aoq_review(id):
    row = q('''select aq.*, r.no pr_no, r.office pr_office, r.items pr_items,
                      r.total pr_total
               from aoq aq left join pr r on r.id=aq.pr_id
               where aq.id=?''', (id,), one=True)
    if not row:
        flash('AOQ record not found.')
        return redirect(url_for('aoq'))
    d = dict(row)
    pr_items = parse_items(d.get('pr_items'))
    offer_items = parse_items(d.get('winning_offer'))
    suppliers = parse_items(d.get('suppliers'))
    return render_review_page(
        active='aoq', title=f'Review AOQ {d["no"]}', document_type='AOQ / Bidding Result',
        document_no=d['no'], status=d['status'], created_at=d.get('created_at'),
        linked_records=[
            {'label': 'Related PR', 'value': d.get('pr_no') or 'No linked PR'},
            {'label': 'Source File', 'value': d.get('ocr_filename') or 'Legacy/manual record',
             'url': url_for('aoq_source_file', id=id) if d.get('stored_filename') else ''},
            {'label': 'Detected Document Type', 'value': d.get('source_type') or 'Legacy/manual record'},
            {'label': 'Extraction Method', 'value': d.get('extraction_method') or 'Legacy/manual extraction'},
        ],
        summary=[
            {'label': 'Requesting Office', 'value': d.get('pr_office') or '—'},
            {'label': 'Winning Supplier', 'value': d.get('winner') or '—'},
            {'label': 'Winning Amount', 'value': money_text(d.get('winning_amount'))},
            {'label': 'Winning Offer Items', 'value': len(offer_items)},
        ],
        item_sections=[
            review_section('Approved PR Requirements', pr_items, d.get('pr_no') or 'PR',
                           'Use this section to confirm what the procurement originally requested.'),
            review_section('OCR-Reviewed Winning Offer', offer_items, d['no'],
                           'Double-check the supplier offer, description, specification, quantity, unit, and price before approval.'),
        ],
        text_blocks=[
            {'title': 'Supplier / Quotation Records', 'content': json.dumps(suppliers, indent=2, ensure_ascii=False, default=str) if suppliers else 'No supplier list stored.'},
            {'title': 'Flexible Extraction Profile', 'content': json.dumps(parse_json_object(d.get('extraction_profile')), indent=2, ensure_ascii=False, default=str) if d.get('extraction_profile') else 'No extraction profile stored.'},
            {'title': 'Extraction Warnings', 'content': json.dumps(parse_json_list(d.get('extraction_warnings')), indent=2, ensure_ascii=False, default=str) if d.get('extraction_warnings') else 'No extraction warnings stored.'},
            {'title': 'Raw Extracted Text', 'content': d.get('ocr_text') or 'No raw extracted text stored.'},
        ],
        approve_url=url_for('aoq_approve', id=id), approve_label=f'Approve AOQ {d["no"]}',
        approval_effect='Approval locks the reviewed winning offer and makes this AOQ available for Purchase Order generation.',
        back_url=url_for('aoq'), print_url=None, can_approve=d.get('status') != 'Approved',
    )

@app.route('/aoq/<int:id>/source')
def aoq_source_file(id):
    row = q('select ocr_filename, stored_filename from aoq where id=?', (id,), one=True)
    if not row or not row['stored_filename']:
        flash('The original AOQ source file is not available for this record.')
        return redirect(url_for('aoq_review', id=id))
    stored = os.path.basename(str(row['stored_filename']))
    path = os.path.join(AOQ_UPLOAD_DIR, stored)
    if not os.path.exists(path):
        flash('The original AOQ source file could not be found.')
        return redirect(url_for('aoq_review', id=id))
    return send_from_directory(
        AOQ_UPLOAD_DIR, stored, as_attachment=False,
        download_name=row['ocr_filename'] or stored,
    )


@app.route('/aoq/<int:id>/approve', methods=['POST'])
def aoq_approve(id):
    row = q('select * from aoq where id=?', (id,), one=True)
    if not row:
        flash('AOQ record not found.')
        return redirect(url_for('aoq'))
    guard = require_review_confirmation('aoq_review', id)
    if guard:
        return guard
    if row['status'] == 'Approved':
        flash('AOQ is already approved.')
        return redirect(url_for('aoq_review', id=id))
    exec_db('update aoq set status=? where id=?', ('Approved', id))
    log('Approved AOQ', 'AOQ', row['no'], 'OCR-created AOQ approved')
    flash('AOQ approved. Continue to Purchase Order generation.')
    return redirect(url_for('po'))

@app.route('/po', methods=['GET','POST'])
def po():
    if request.method=='POST':
        aoq_id=request.form.get('aoq_id')
        a=q('''select aq.* from aoq aq where aq.id=? and aq.status="Approved"
               and not exists (select 1 from po p where p.aoq_id=aq.id)''',(aoq_id,),one=True)
        if a:
            # Correct rule: PO is generated from the approved OCR-created AOQ winning offer.
            # It must NOT copy the PR items, otherwise NLP compares PR against itself.
            po_items=parse_items(a['winning_offer'])
            if not po_items:
                po_items=[{
                    'item':'AOQ Winning Offer',
                    'description':a['winning_offer'] or '',
                    'qty':1,
                    'unit':'unit',
                    'unit_cost':to_float(a['winning_amount']),
                    'amount':to_float(a['winning_amount'])
                }]
            total=sum(to_float(x.get('amount') or (to_float(x.get('qty') or 1) * to_float(x.get('unit_cost') or 0))) for x in po_items)
            if total <= 0:
                total=to_float(a['winning_amount'])
            no = next_document_no('PO', 'PO', 'po')
            po_items = assign_item_identifiers(po_items, '_PO Item ID', no)
            exec_db('insert into po(no,aoq_id,supplier,status,created_at,items,total) values(?,?,?,?,?,?,?)',(no,a['id'],a['winner'],'For Review',now(),json.dumps(po_items,default=str),total))
            log('Generated PO','PO',no,f'from approved OCR AOQ {a["no"]}; {len(po_items)} winning offer item(s)')
            flash('PO generated successfully.')
        else:
            flash('Select an approved AOQ that does not already have a PO.')
        return redirect(url_for('po'))
    rows=q("""select p.*, aq.no aoq_no, aq.winner, r.no pr_no
              from po p
              left join aoq aq on aq.id=p.aoq_id
              left join pr r on r.id=aq.pr_id
              order by p.id desc""")
    _,_,_,raw_aoqs,_,_ = next_available_records()
    aoqs=[]
    for a in raw_aoqs:
        d=dict(a)
        pr_row=q('select no,items,office from pr where id=?',(a['pr_id'],),one=True)
        d['pr_no']=pr_row['no'] if pr_row else 'No linked PR'
        d['item_preview']=item_preview_from_json(a['winning_offer'],limit=1)
        aoqs.append(d)
    return render_template('po.html', active='po', rows=rows, aoqs=aoqs, title='PO')

@app.route('/po/<int:id>/view')
def po_review(id):
    row = q('''select p.*, aq.no aoq_no, aq.winner aoq_winner,
                      aq.winning_offer aoq_items, r.no pr_no, r.items pr_items,
                      r.office pr_office
               from po p
               left join aoq aq on aq.id=p.aoq_id
               left join pr r on r.id=aq.pr_id
               where p.id=?''', (id,), one=True)
    if not row:
        flash('PO record not found.')
        return redirect(url_for('po'))
    d = dict(row)
    po_items = parse_items(d.get('items'))
    return render_review_page(
        active='po', title=f'Review PO {d["no"]}', document_type='Purchase Order',
        document_no=d['no'], status=d['status'], created_at=d.get('created_at'),
        linked_records=[
            {'label': 'Related PR', 'value': d.get('pr_no') or 'No linked PR'},
            {'label': 'Source AOQ', 'value': d.get('aoq_no') or 'No linked AOQ'},
        ],
        summary=[
            {'label': 'Supplier', 'value': d.get('supplier') or '—'},
            {'label': 'Requesting Office', 'value': d.get('pr_office') or '—'},
            {'label': 'Number of PO Items', 'value': len(po_items)},
            {'label': 'PO Total', 'value': money_text(d.get('total'))},
        ],
        item_sections=[
            review_section('Approved PR Requirements', parse_items(d.get('pr_items')), d.get('pr_no') or 'PR',
                           'The NLP module will compare these requirements against the final PO.'),
            review_section('Approved AOQ Winning Offer', parse_items(d.get('aoq_items')), d.get('aoq_no') or 'AOQ',
                           'Verify that the PO was generated from this approved offer.'),
            review_section('Purchase Order Items for Approval', po_items, d['no'],
                           'Double-check all descriptions, specifications, quantities, units, prices, and totals.'),
        ],
        approve_url=url_for('po_approve', id=id), approve_label=f'Approve PO {d["no"]}',
        approval_effect='Approval locks this PO and makes the linked PR–PO pair available for NLP Verification.',
        back_url=url_for('po'), print_url=url_for('po_print', id=id),
        can_approve=d.get('status') != 'Approved',
    )

@app.route('/po/<int:id>/approve', methods=['POST'])
def po_approve(id):
    row = q('select * from po where id=?', (id,), one=True)
    if not row:
        flash('PO record not found.')
        return redirect(url_for('po'))
    guard = require_review_confirmation('po_review', id)
    if guard:
        return guard
    if row['status'] == 'Approved':
        flash('PO is already approved.')
        return redirect(url_for('po_review', id=id))
    exec_db('update po set status=? where id=?', ('Approved', id))
    log('Approved PO', 'PO', row['no'], 'Purchase Order approved')
    flash('PO approved. The procurement process is complete through PO. Open PR–PO NLP Verification when you are ready to continue to receiving and inventory.')
    return redirect(url_for('po'))

@app.route('/po/<int:id>/print')
def po_print(id):
    row=q('select * from po where id=?',(id,),one=True); items=parse_items(row['items'])
    return render_template('print_po.html', row=row, items=items, sig=SIGNATORIES, title='Print PO')

@app.route('/nlp', methods=['GET','POST'])
def nlp():
    if request.method=='POST':
        po_id=request.form.get('po_id'); po_row=q('''select p.* from po p where p.id=? and p.status="Approved"
               and not exists (select 1 from nlp n where n.po_id=p.id)''',(po_id,),one=True)
        if po_row:
            a=q('select * from aoq where id=?',(po_row['aoq_id'],),one=True); r=q('select * from pr where id=?',(a['pr_id'],),one=True)
            try:
                result, conf, details = nlp_predict(parse_items(r['items']), parse_items(po_row['items']))
            except Exception as error:
                log('PR–PO NLP Check Failed', 'NLP', po_row['no'], str(error))
                flash(f'NLP verification stopped: {error}. No fallback or hardcoded result was created.')
                return redirect(url_for('nlp'))
            finding = nlp_short_finding(result, details)
            nlp_no = next_document_no('NLP', 'NLP', 'nlp')
            exec_db('insert into nlp(no,po_id,result,confidence,staff_decision,remarks,created_at,details,status) values(?,?,?,?,?,?,?,?,?)',(nlp_no,po_id,result,conf,result,nlp_dynamic_remarks(result, details),now(),json.dumps(details, default=str),'For Review'))
            log('PR–PO FastText NLP Check Completed','NLP',nlp_no,f'PR {r["no"]} versus PO {po_row["no"]}: {result} {conf}% - {finding}'); flash(f'FastText NLP verification {nlp_no} completed for PR {r["no"]} and PO {po_row["no"]}.')
        else:
            flash('Select an approved PO that has not yet been verified.')
        return redirect(url_for('nlp'))
    raw_rows=q('''select n.*, p.no po_no, p.supplier, aq.no aoq_no, r.no pr_no
                  from nlp n
                  join po p on p.id=n.po_id
                  left join aoq aq on aq.id=p.aoq_id
                  left join pr r on r.id=aq.pr_id
                  order by n.id desc''')
    rows=[]
    for row in raw_rows:
        d=dict(row)
        d['findings'] = nlp_short_finding(d.get('result'), d.get('details'))
        d['remarks'] = nlp_dynamic_remarks(d.get('result'), d.get('details'))
        rows.append(d)
    _,_,_,_,raw_pos,_ = next_available_records()
    pos = []
    for p in raw_pos:
        d = dict(p)
        trace = trace_for_po(p)
        d['pr_no'] = trace.get('pr')['no'] if trace.get('pr') else 'No linked PR'
        d['aoq_no'] = trace.get('aoq')['no'] if trace.get('aoq') else 'No linked AOQ'
        d['item_preview'] = item_preview_from_json(d.get('items'), limit=1)
        pos.append(d)
    return render_template('nlp.html', active='nlp', rows=rows, pos=pos, model_state=trained_nlp_model_status(), title='PR–PO NLP Verification')

@app.route('/nlp/<int:id>/view')
def nlp_review(id):
    row = q('''select n.*, p.no po_no, p.items po_items, p.supplier,
                      aq.no aoq_no, r.no pr_no, r.items pr_items
               from nlp n
               join po p on p.id=n.po_id
               left join aoq aq on aq.id=p.aoq_id
               left join pr r on r.id=aq.pr_id
               where n.id=?''', (id,), one=True)
    if not row:
        flash('NLP result not found.')
        return redirect(url_for('nlp'))
    d = dict(row)
    try:
        details = json.loads(d.get('details') or '{}')
    except Exception:
        details = {}
    finding = nlp_short_finding(d.get('result'), details)
    remarks = nlp_dynamic_remarks(d.get('result'), details)
    source = details.get('final_decision_source') or 'AI-only FastText PR–PO verification'
    return render_review_page(
        active='nlp', title=f'Review NLP {d.get("no") or id}', document_type='PR–PO NLP Verification',
        document_no=d.get('no') or f'NLP-{id}', status=d.get('status') or 'For Review',
        created_at=d.get('created_at'),
        linked_records=[
            {'label': 'Compared PR', 'value': d.get('pr_no') or 'No linked PR'},
            {'label': 'Supporting AOQ', 'value': d.get('aoq_no') or 'No linked AOQ'},
            {'label': 'Compared PO', 'value': d.get('po_no') or 'No linked PO'},
        ],
        summary=[
            {'label': 'Supplier', 'value': d.get('supplier') or '—'},
            {'label': 'NLP Result', 'value': d.get('result') or '—'},
            {'label': 'Match / Certainty Score', 'value': f'{to_float(d.get("confidence")):.2f}%'},
            {'label': 'Decision Source', 'value': source},
        ],
        item_sections=[
            review_section('Approved PR Requirements', parse_items(d.get('pr_items')), d.get('pr_no') or 'PR',
                           'Original requirements supplied to the NLP model.'),
            review_section('Approved PO Items', parse_items(d.get('po_items')), d.get('po_no') or 'PO',
                           'Final purchase-order values compared by the NLP model.'),
        ],
        text_blocks=[
            {'title': 'Primary NLP Finding', 'content': finding},
            {'title': 'Recommended Staff Action', 'content': remarks},
            {'title': 'Complete NLP Comparison Details', 'content': json.dumps(details, indent=2, ensure_ascii=False, default=str)},
        ],
        approve_url=url_for('nlp_approve', id=id),
        approve_label=f'Approve NLP Result {d.get("no") or id}',
        approval_effect=('Approval accepts this recorded NLP decision. Acceptable/Needs Review results may proceed to IAR; '
                         'Needs Recanvass/Reject results remain blocked for corrective action.'),
        back_url=url_for('nlp'), print_url=None,
        can_approve=d.get('status') != 'Approved',
    )

@app.route('/nlp/<int:id>/approve', methods=['POST'])
def nlp_approve(id):
    row = q('''select n.*, p.no po_no from nlp n
               join po p on p.id=n.po_id where n.id=?''', (id,), one=True)
    if not row:
        flash('NLP result not found.')
        return redirect(url_for('nlp'))
    guard = require_review_confirmation('nlp_review', id)
    if guard:
        return guard
    if row['status'] == 'Approved':
        flash('NLP result is already approved.')
        return redirect(url_for('nlp_review', id=id))
    exec_db('update nlp set status=?, staff_decision=? where id=?', ('Approved', row['result'], id))
    log('Approved NLP Result', 'NLP', row['po_no'], f'Approved prediction: {row["result"]}')
    if row['result'] in ('Acceptable', 'Needs Review'):
        flash('NLP result approved. The PO is now available for IAR creation.')
    else:
        flash('NLP result approved, but this result blocks IAR and requires corrective procurement action.')
    return redirect(url_for('nlp'))

@app.route('/nlp/algorithms')
def nlp_algorithms():
    return render_template(
        'nlp_algorithms.html',
        active='nlp',
        algorithms=algorithm_comparison(),
        title='NLP Algorithms',
    )

@app.route('/iar', methods=['GET','POST'])
def iar():
    if request.method=='POST':
        po_id=request.form.get('po_id'); p=q('''select p.* from po p where p.id=? and p.status="Approved"
               and exists (select 1 from nlp n where n.po_id=p.id and n.status="Approved" and n.result in ("Acceptable","Needs Review"))
               and not exists (select 1 from iar i where i.po_id=p.id)''',(po_id,),one=True)
        if p:
            no = next_document_no('IAR', 'IAR', 'iar')
            iar_items = assign_item_identifiers(parse_items(p['items']), '_IAR Item ID', no)
            exec_db('insert into iar(no,po_id,status,created_at,items) values(?,?,?,?,?)',(no,p['id'],'For Review',now(),json.dumps(iar_items, default=str)))
            log('Created IAR','IAR',no,'IAR created for review; inventory is updated only after approval')
            flash('IAR created. Approve it to accept the delivery and add the items to inventory.')
        else:
            flash('The selected PO is not eligible for IAR creation or already has an IAR.')
        return redirect(url_for('iar'))
    rows=q('''select i.*, p.no po_no, p.supplier, n.no nlp_no, aq.no aoq_no, r.no pr_no
              from iar i
              left join po p on p.id=i.po_id
              left join nlp n on n.po_id=p.id
              left join aoq aq on aq.id=p.aoq_id
              left join pr r on r.id=aq.pr_id
              order by i.id desc''')
    _,_,_,_,_,raw_pos = next_available_records()
    pos=[]
    for p in raw_pos:
        d=dict(p)
        trace=trace_for_po(p)
        n=q('select no,result from nlp where po_id=? and status="Approved" order by id desc limit 1',(p['id'],),one=True)
        d['pr_no']=trace.get('pr')['no'] if trace.get('pr') else 'No linked PR'
        d['nlp_no']=n['no'] if n else 'No approved NLP'
        d['nlp_result']=n['result'] if n else ''
        d['item_preview']=item_preview_from_json(d.get('items'),limit=1)
        pos.append(d)
    return render_template('iar.html', active='iar', rows=rows, pos=pos, title='IAR/AIR')

@app.route('/iar/<int:id>/view')
def iar_review(id):
    row = q('''select i.*, p.no po_no, p.items po_items, p.supplier,
                      aq.no aoq_no, r.no pr_no, n.no nlp_no, n.result nlp_result
               from iar i
               join po p on p.id=i.po_id
               left join aoq aq on aq.id=p.aoq_id
               left join pr r on r.id=aq.pr_id
               left join nlp n on n.po_id=p.id and n.status='Approved'
               where i.id=? order by n.id desc limit 1''', (id,), one=True)
    if not row:
        flash('IAR record not found.')
        return redirect(url_for('iar'))
    d = dict(row)
    iar_items = parse_items(d.get('items'))
    return render_review_page(
        active='iar', title=f'Review IAR {d["no"]}', document_type='Inspection and Acceptance Report',
        document_no=d['no'], status=d['status'], created_at=d.get('created_at'),
        linked_records=[
            {'label': 'Related PR', 'value': d.get('pr_no') or 'No linked PR'},
            {'label': 'Related AOQ', 'value': d.get('aoq_no') or 'No linked AOQ'},
            {'label': 'Related PO', 'value': d.get('po_no') or 'No linked PO'},
            {'label': 'Approved NLP', 'value': f'{d.get("nlp_no") or "—"} — {d.get("nlp_result") or "—"}'},
        ],
        summary=[
            {'label': 'Supplier', 'value': d.get('supplier') or '—'},
            {'label': 'Delivery Items', 'value': len(iar_items)},
            {'label': 'Inventory Effect', 'value': 'Accepted items will be added to Inventory after approval.'},
        ],
        item_sections=[
            review_section('Purchase Order Items', parse_items(d.get('po_items')), d.get('po_no') or 'PO',
                           'Confirm what was ordered.'),
            review_section('IAR Delivery / Acceptance Items', iar_items, d['no'],
                           'Verify every delivered item and quantity before accepting the IAR.'),
        ],
        approve_url=url_for('iar_approve', id=id), approve_label=f'Approve IAR {d["no"]}',
        approval_effect='Approval accepts the delivery and adds every listed IAR item to the system Inventory.',
        back_url=url_for('iar'), print_url=url_for('iar_print', id=id),
        can_approve=d.get('status') != 'Approved',
    )

@app.route('/iar/<int:id>/approve', methods=['POST'])
def iar_approve(id):
    row = q('''select i.*, p.no po_no from iar i
               join po p on p.id=i.po_id where i.id=?''', (id,), one=True)
    if not row:
        flash('IAR record not found.')
        return redirect(url_for('iar'))
    guard = require_review_confirmation('iar_review', id)
    if guard:
        return guard
    if row['status'] == 'Approved':
        flash('IAR is already approved.')
        return redirect(url_for('iar_review', id=id))

    for it in parse_items(row['items']):
        item_name = it.get('item') or it.get('Item Name') or 'Item'
        description = it.get('description') or it.get('Description / Specification') or it.get('Description') or ''
        qty = to_float(it.get('qty') or it.get('Quantity') or 1) or 1
        unit = it.get('unit') or it.get('Unit') or 'unit'
        unit_cost = to_float(it.get('unit_cost') or it.get('Unit Cost') or it.get('Estimated Unit Cost') or 0)
        cat = 'Equipment' if any(w in str(item_name).lower() for w in ['laptop','printer','chair','projector','air conditioner']) else 'Consumable'
        source_dataset_id = it.get('_Dataset Record ID') or it.get('_dataset_record_id')
        item_code = next_document_no('INVENTORY', 'INV', 'inventory', column='item_code', width=5)
        exec_db(
            '''insert into inventory(item_code,item,description,category,qty,unit,source_po,status,created_at,unit_cost,source_dataset_record_id)
            values(?,?,?,?,?,?,?,?,?,?,?)''',
            (item_code, item_name, description, cat, qty, unit, row['po_no'], 'Available', now(), unit_cost, source_dataset_id)
        )
    exec_db('update iar set status=? where id=?', ('Approved', id))
    log('Approved IAR', 'IAR', row['no'], 'Delivery accepted and items added to inventory')
    flash('IAR approved. Delivery items were added to inventory.')
    return redirect(url_for('iar'))

@app.route('/iar/<int:id>/print')
def iar_print(id):
    row=q('select * from iar where id=?',(id,),one=True); po_row=q('select * from po where id=?',(row['po_id'],),one=True); items=parse_items(row['items'])
    return render_template('print_iar.html', row=row, po=po_row, items=items, sig=SIGNATORIES, title='Print IAR')

@app.route('/inventory')
def inventory():
    rows = q('select * from inventory order by id desc')
    transactions = q('''select t.*, i.item_code, i.item from inventory_transactions t
                        left join inventory i on i.id=t.inventory_id order by t.id desc limit 50''')
    return render_template('inventory.html', active='inventory', rows=rows, transactions=transactions, title='Inventory')

@app.route('/ris-ics', methods=['GET','POST'])
def ris_ics():
    if request.method == 'POST':
        typ = (request.form.get('type') or '').upper().strip()
        if typ not in ('RIS', 'ICS'):
            flash('Choose RIS for consumables or ICS for equipment.')
            return redirect(url_for('ris_ics'))
        office = (request.form.get('office') or 'General Services Office').strip()
        selected_ids = [int(x) for x in request.form.getlist('inventory_id') if str(x).isdigit()]
        if not selected_ids:
            flash('Select at least one available inventory item.')
            return redirect(url_for('ris_ics'))
        expected_category = 'Equipment' if typ == 'ICS' else 'Consumable'
        items = []
        for inventory_id in selected_ids:
            inv = q('select * from inventory where id=? and qty>0 and category=?', (inventory_id, expected_category), one=True)
            if not inv:
                continue
            requested_qty = to_float(request.form.get(f'issue_qty_{inventory_id}')) or 1
            if typ == 'ICS':
                requested_qty = min(requested_qty, 1.0)
            requested_qty = min(requested_qty, to_float(inv['qty']))
            if requested_qty <= 0:
                continue
            items.append({
                'inventory_id': inv['id'],
                'item_code': inv['item_code'],
                'item': inv['item'],
                'description': inv['description'] or '',
                'category': inv['category'],
                'qty': requested_qty,
                'unit': inv['unit'] or 'unit',
                'unit_cost': to_float(inv['unit_cost']),
                'available_before': to_float(inv['qty']),
                'source_po': inv['source_po'] or '',
            })
        if not items:
            flash(f'No available {expected_category.lower()} inventory items were selected.')
            return redirect(url_for('ris_ics'))
        no = next_document_no(typ, typ, 'ris_ics')
        items = assign_item_identifiers(items, '_Issue Item ID', no)
        exec_db('insert into ris_ics(no,type,office,status,created_at,items) values(?,?,?,?,?,?)',
                (no, typ, office, 'For Review', now(), json.dumps(items, default=str)))
        log(f'Created {typ}', 'RIS/ICS', no, f'{len(items)} inventory item(s) prepared for {office}')
        flash(f'{no} created. Review all selected inventory items before approval.')
        return redirect(url_for('ris_ics'))

    rows = q('select * from ris_ics order by id desc')
    consumables = q('select * from inventory where category="Consumable" and qty>0 order by item collate nocase')
    equipment = q('select * from inventory where category="Equipment" and qty>0 order by item collate nocase')
    return render_template('ris_ics.html', active='ris', rows=rows, consumables=consumables,
                           equipment=equipment, title='RIS/ICS')

@app.route('/ris-ics/<int:id>/view')
def ris_ics_review(id):
    row = q('select * from ris_ics where id=?', (id,), one=True)
    if not row:
        flash('RIS/ICS record not found.')
        return redirect(url_for('ris_ics'))
    d = dict(row)
    items = parse_items(d.get('items'))
    full_name = 'Requisition and Issue Slip' if d.get('type') == 'RIS' else 'Inventory Custodian Slip'
    return render_review_page(
        active='ris', title=f'Review {d["no"]}', document_type=full_name,
        document_no=d['no'], status=d['status'], created_at=d.get('created_at'),
        linked_records=[{'label': 'Document Type', 'value': d.get('type') or '—'}],
        summary=[
            {'label': 'Receiving Office / Custodian', 'value': d.get('office') or '—'},
            {'label': 'Number of Inventory Items', 'value': len(items)},
            {'label': 'Inventory Effect', 'value': 'Approval deducts the listed quantities from live inventory.'},
        ],
        item_sections=[review_section(f'{d.get("type")} Inventory Items', items, d['no'],
                                      'Verify inventory identifiers, available quantities, issued quantities, units, and receiving office.')],
        approve_url=url_for('ris_ics_approve', id=id), approve_label=f'Approve {d["no"]}',
        approval_effect=f'Approval finalizes this {d.get("type")} and deducts the issued quantities from Inventory.',
        back_url=url_for('ris_ics'), print_url=url_for('ris_ics_print', id=id),
        can_approve=d.get('status') != 'Approved',
    )

@app.route('/ris-ics/<int:id>/approve', methods=['POST'])
def ris_ics_approve(id):
    row = q('select * from ris_ics where id=?', (id,), one=True)
    if not row:
        flash('RIS/ICS record not found.')
        return redirect(url_for('ris_ics'))
    guard = require_review_confirmation('ris_ics_review', id)
    if guard:
        return guard
    if row['status'] == 'Approved':
        flash('RIS/ICS document is already approved; inventory was not deducted again.')
        return redirect(url_for('ris_ics_review', id=id))
    items = parse_items(row['items'])
    con = conn()
    try:
        con.execute('begin immediate')
        for item in items:
            inventory_id = int(to_float(item.get('inventory_id')))
            issue_qty = to_float(item.get('qty'))
            inv = con.execute('select * from inventory where id=?', (inventory_id,)).fetchone()
            if not inv:
                raise ValueError(f'Inventory item {item.get("item_code") or inventory_id} no longer exists.')
            available = to_float(inv['qty'])
            if issue_qty <= 0 or issue_qty > available:
                raise ValueError(f'Insufficient quantity for {inv["item_code"]}: requested {issue_qty:g}, available {available:g}.')
            remaining = round(available - issue_qty, 4)
            new_status = 'Available' if remaining > 0 else ('Assigned' if row['type'] == 'ICS' else 'Issued')
            con.execute('update inventory set qty=?, status=? where id=?', (remaining, new_status, inventory_id))
            con.execute('''insert into inventory_transactions(inventory_id,document_no,transaction_type,qty,unit,office,created_at,created_by)
                           values(?,?,?,?,?,?,?,?)''',
                        (inventory_id, row['no'], row['type'], issue_qty, inv['unit'], row['office'], now(), session.get('username', 'admin')))
        con.execute('update ris_ics set status=? where id=?', ('Approved', id))
        con.commit()
    except Exception as exc:
        con.rollback()
        con.close()
        flash(f'Approval stopped: {exc}')
        return redirect(url_for('ris_ics_review', id=id))
    con.close()
    log(f'Approved {row["type"]}', 'RIS/ICS', row['no'], f'Inventory quantities deducted for {len(items)} item(s)')
    flash(f'{row["no"]} approved and live inventory quantities were updated.')
    return redirect(url_for('ris_ics'))

@app.route('/ris-ics/<int:id>/print')
def ris_ics_print(id):
    row=q('select * from ris_ics where id=?',(id,),one=True); items=parse_items(row['items'])
    return render_template('print_ris_ics.html', row=row, items=items, sig=SIGNATORIES, title='Print RIS/ICS')

@app.route('/reports')
def reports():
    totals = {
        'ppmp_total': to_float(q('select coalesce(sum(total),0) v from ppmp', one=True)['v']),
        'pr_total': to_float(q('select coalesce(sum(total),0) v from pr', one=True)['v']),
        'po_total': to_float(q('select coalesce(sum(total),0) v from po', one=True)['v']),
        'inventory_value': to_float(q('select coalesce(sum(qty*coalesce(unit_cost,0)),0) v from inventory', one=True)['v']),
    }
    counts = {name: q(f'select count(*) c from {table}', one=True)['c'] for name, table in [
        ('PPMP','ppmp'),('APP','app_plan'),('PR','pr'),('AOQ','aoq'),('PO','po'),
        ('NLP','nlp'),('IAR','iar'),('Inventory','inventory'),('RIS/ICS','ris_ics')
    ]}
    return render_template('reports.html', active='reports', totals=totals, counts=counts, title='Reports')


def _csv_response(filename, headers, rows):
    output = io.StringIO(newline='')
    writer = csv.writer(output)
    writer.writerow(headers)
    for row in rows:
        writer.writerow(row)
    return Response('\ufeff' + output.getvalue(), mimetype='text/csv; charset=utf-8',
                    headers={'Content-Disposition': f'attachment; filename="{filename}"'})


@app.route('/reports/export/<report_type>.csv')
def report_export(report_type):
    report_type = report_type.lower().strip()
    if report_type == 'procurement':
        rows = q('''select r.no pr_no, aq.no aoq_no, aq.winner, p.no po_no, p.total, p.status,
                           n.no nlp_no, n.result nlp_result, n.status nlp_status
                    from pr r left join aoq aq on aq.pr_id=r.id
                    left join po p on p.aoq_id=aq.id left join nlp n on n.po_id=p.id
                    order by r.id desc''')
        return _csv_response('procurement_chain_report.csv',
            ['PR Number','AOQ Number','Winning Supplier','PO Number','PO Total','PO Status','NLP Number','NLP Result','NLP Status'],
            [[r['pr_no'],r['aoq_no'],r['winner'],r['po_no'],r['total'],r['status'],r['nlp_no'],r['nlp_result'],r['nlp_status']] for r in rows])
    if report_type == 'inventory':
        rows = q('select item_code,item,description,category,qty,unit,unit_cost,source_po,status from inventory order by item')
        return _csv_response('inventory_report.csv',
            ['Item Code','Item','Description','Category','Available Quantity','Unit','Unit Cost','Source PO','Status'],
            [[r['item_code'],r['item'],r['description'],r['category'],r['qty'],r['unit'],r['unit_cost'],r['source_po'],r['status']] for r in rows])
    if report_type == 'issuance':
        rows = q('select no,type,office,status,created_at,items from ris_ics order by id desc')
        return _csv_response('ris_ics_report.csv',
            ['Document Number','Type','Office/Custodian','Status','Created At','Items JSON'],
            [[r['no'],r['type'],r['office'],r['status'],r['created_at'],r['items']] for r in rows])
    if report_type == 'audit':
        rows = q('select dt,user,action,module,ref,details from audit_trail order by id desc')
        return _csv_response('audit_trail_report.csv',
            ['Date/Time','User','Action','Module','Reference','Details'],
            [[r['dt'],r['user'],r['action'],r['module'],r['ref'],r['details']] for r in rows])
    abort(404)


@app.route('/users', methods=['GET','POST'])
def users():
    guard = require_admin_access()
    if guard:
        return guard
    if request.method == 'POST':
        action = request.form.get('action') or 'create'
        if action == 'create':
            username = (request.form.get('username') or '').strip()
            password = request.form.get('password') or ''
            role = (request.form.get('role') or 'Staff').strip()
            if role not in ('Admin', 'Staff', 'Inspector'):
                role = 'Staff'
            if len(username) < 3 or len(password) < 8:
                flash('Username must have at least 3 characters and password at least 8 characters.')
            elif q('select id from users where username=?', (username,), one=True):
                flash('That username already exists.')
            else:
                exec_db('insert into users(username,password,role) values(?,?,?)',
                        (username, generate_password_hash(password), role))
                log('Created User', 'User Management', username, f'Role: {role}')
                flash(f'User {username} created.')
        elif action == 'reset_password':
            user_id = request.form.get('user_id')
            password = request.form.get('password') or ''
            user = q('select * from users where id=?', (user_id,), one=True)
            if not user or len(password) < 8:
                flash('Select a valid user and enter a password with at least 8 characters.')
            else:
                exec_db('update users set password=? where id=?', (generate_password_hash(password), user_id))
                log('Reset Password', 'User Management', user['username'], 'Administrator reset account password')
                flash(f'Password reset for {user["username"]}.')
        return redirect(url_for('users'))
    rows = q('select id,username,role from users order by username collate nocase')
    return render_template('users.html', active='users', rows=rows, title='User Management')


@app.route('/audit')
def audit():
    rows = q('select * from audit_trail order by id desc limit 500')
    return render_template('audit.html', active='audit', rows=rows, title='Audit Trail')


@app.route('/health')
def health():
    model_state = trained_nlp_model_status()
    return {
        'application': 'ready',
        'database': os.path.exists(DB_PATH),
        'nlp_model_active': bool(model_state.get('active')),
        'nlp_message': model_state.get('message'),
    }

if __name__ == '__main__':
    init_db()
    _DATABASE_READY = True
    print('Running PSMS at http://127.0.0.1:5000')
    print('Pre-deployment mode: Waitress WSGI server (offline/local)')
    try:
        from waitress import serve
        serve(app, host='127.0.0.1', port=5000, threads=4)
    except ImportError:
        print('Waitress is not installed. Run: pip install -r requirements.txt')
        print('Fallback server started with debug disabled.')
        app.run(debug=False, host='127.0.0.1', port=5000)


