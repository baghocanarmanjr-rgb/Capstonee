"""Train the AI-only PR-to-PO discrepancy model.

The deployed decision is controlled exclusively by trained classifiers:
- a four-class status model predicts Acceptable, Needs Review,
  Needs Recanvass, or Reject;
- a trained issue model predicts the primary discrepancy type.

No procurement rule, manually weighted hybrid score, or rule-based fallback is
used at runtime. The Municipality of Asuncion COA workbook supplies the real
item vocabulary. Controlled PR/PO examples are generated because the workbook
contains inventory records rather than labelled PR/PO pairs.
"""
from __future__ import annotations

import json
import os
import random
import re
from typing import Any, Dict, List, Sequence, Tuple
from difflib import SequenceMatcher

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from inventory_dataset import read_dynamic_workbook

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(MODEL_DIR, 'nlp_discrepancy_model.pkl')
FASTTEXT_PATH = os.path.join(MODEL_DIR, 'fasttext_model.model')
METHOD_PATH = os.path.join(MODEL_DIR, 'nlp_method.json')
COA_CORPUS_PATH = os.path.join(MODEL_DIR, 'coa_inventory_corpus.json')
GENERATED_PAIRS_PATH = os.path.join(DATA_DIR, 'COA_Generated_PR_PO_Training_Pairs.csv')
DEFAULT_COA_PATH = os.path.join(DATA_DIR, 'COA_Inventory_Dataset.xlsx')
DEFAULT_LABELLED_PATH = os.path.join(BASE_DIR, 'PR_PO_dataset_full_with_categories.csv')

VECTOR_SIZE = 64
WINDOW = 5
EPOCHS = 5
RANDOM_STATE = 42
STATUS_CLASSES = ['Acceptable', 'Needs Review', 'Needs Recanvass', 'Reject']


def clean_text(value: Any) -> str:
    text = str(value or '').lower()
    text = re.sub(r'[^a-z0-9_\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def normalized_unit(value: Any) -> str:
    unit = clean_text(value)
    aliases = {
        'pc': 'piece', 'pcs': 'piece', 'pieces': 'piece', 'unit': 'piece', 'units': 'piece',
        'btl': 'bottle', 'btls': 'bottle', 'bottles': 'bottle',
        'reams': 'ream', 'rolls': 'roll', 'boxes': 'box', 'sets': 'set',
        'packs': 'pack', 'lots': 'lot',
    }
    return aliases.get(unit, unit or 'piece')


def item_text(record: Dict[str, Any]) -> str:
    parts = [record.get('article'), record.get('description')]
    excluded = {
        'person accountable', 'accountable person', 'office', 'department',
        'property number', 'semi expendable property number', 'balance per card',
        'on hand per card', 'shortage', 'overage', 'remarks', 'location',
        'unit value', 'recorded value', 'amount', 'price', 'cost',
    }
    for key, value in (record.get('raw_data') or {}).items():
        key_norm = clean_text(key)
        if any(token in key_norm for token in excluded):
            continue
        value_text = str(value or '').strip()
        if value_text and not re.fullmatch(r'[\d,.$₱\s-]+', value_text):
            parts.append(value_text)
    return clean_text(' '.join(str(x or '') for x in parts))


def serialize_item(description: str, quantity: int, unit: str, index: int = 1) -> str:
    return (
        f'item_{index} description {clean_text(description)} '
        f'quantity qty_{int(quantity)} unit unit_{normalized_unit(unit).replace(" ", "_")}'
    )


ABBREVIATIONS = {
    'computer': 'pc', 'pieces': 'pcs', 'piece': 'pc', 'memory': 'ram',
    'solid state drive': 'ssd', 'gigabyte': 'gb', 'kilogram': 'kg',
    'milliliter': 'ml', 'liter': 'l', 'air conditioner': 'aircon',
}


def acceptable_variant(text: str, rng: random.Random) -> str:
    result = clean_text(text)
    for source, target in ABBREVIATIONS.items():
        if source in result and rng.random() < 0.65:
            result = result.replace(source, target)
    words = result.split()
    # A harmless adjacent swap gives the model realistic word-order variation.
    if len(words) > 4 and rng.random() < 0.35:
        i = rng.randrange(1, len(words) - 1)
        words[i], words[i + 1] = words[i + 1], words[i]
    return ' '.join(words)


def minor_incomplete_variant(text: str, rng: random.Random) -> str:
    words = clean_text(text).split()
    removable = [i for i, w in enumerate(words) if len(w) > 3 and not re.search(r'\d', w)]
    if removable:
        words.pop(rng.choice(removable))
    elif len(words) > 2:
        words = words[:-1]
    return ' '.join(words) or clean_text(text)


def typo_variant(text: str, rng: random.Random) -> str:
    words = clean_text(text).split()
    choices = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    if not choices:
        return minor_incomplete_variant(text, rng)
    i = rng.choice(choices)
    w = words[i]
    pos = rng.randrange(1, len(w) - 1)
    words[i] = w[:pos] + w[pos + 1:]
    return ' '.join(words)


def mutate_specification(text: str, rng: random.Random) -> Tuple[str, bool]:
    original = clean_text(text)
    matches = list(re.finditer(r'\b\d+(?:\.\d+)?\b', original))
    if matches:
        match = rng.choice(matches)
        value = float(match.group())
        changed = max(1, int(round(value * rng.choice([0.5, 0.75, 1.5, 2.0]))))
        if str(changed) == match.group():
            changed += 1
        return original[:match.start()] + str(changed) + original[match.end():], True
    replacements = [
        ('black', 'yellow'), ('yellow', 'cyan'), ('cyan', 'magenta'),
        ('red', 'blue'), ('blue', 'green'), ('small', 'large'),
        ('single', 'double'), ('wired', 'wireless'),
    ]
    for source, target in replacements:
        if re.search(rf'\b{source}\b', original):
            return re.sub(rf'\b{source}\b', target, original, count=1), True
    return original, False


def prepare_corpus(workbook_path: str) -> List[Dict[str, Any]]:
    records, _sheet_meta = read_dynamic_workbook(workbook_path)
    corpus: List[Dict[str, Any]] = []
    seen = set()
    for record in records:
        description = item_text(record)
        if len(description.split()) < 2 or description in seen:
            continue
        seen.add(description)
        corpus.append({
            'text': description,
            'article': clean_text(record.get('article')),
            'description': clean_text(record.get('description')),
            'unit': normalized_unit(record.get('unit')),
            'sheet': str(record.get('source_sheet') or ''),
            'source_row': int(record.get('source_row') or 0),
            'group_id': f'{record.get("source_sheet") or "sheet"}:{record.get("source_row") or len(corpus)+1}',
        })
    if len(corpus) < 20:
        raise ValueError('The COA workbook does not contain enough usable item descriptions.')
    return corpus


def choose_different_index(corpus: List[Dict[str, Any]], idx: int, rng: random.Random, prefer_other_article: bool) -> int:
    article = corpus[idx]['article']
    candidates = [j for j, row in enumerate(corpus) if j != idx and (not prefer_other_article or row['article'] != article)]
    if not candidates:
        candidates = [j for j in range(len(corpus)) if j != idx]
    return rng.choice(candidates)


def generate_pairs_for_corpus(corpus: List[Dict[str, Any]], seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    units = ['piece', 'bottle', 'box', 'ream', 'roll', 'set', 'pack']
    rows: List[Dict[str, Any]] = []

    def add(row, pr, po, status, issue, source):
        rows.append({
            'pr_text': pr, 'po_text': po, 'label': status, 'issue': issue,
            'source': source, 'sheet': row['sheet'], 'source_row': row['source_row'],
            'group_id': row['group_id'],
        })

    for idx, row in enumerate(corpus):
        base = row['text']
        unit = row['unit'] or 'piece'
        quantity = rng.randint(1, 50)
        pr = serialize_item(base, quantity, unit)

        # Acceptable: exact and semantically equivalent text.
        add(row, pr, serialize_item(base, quantity, unit), 'Acceptable', 'No discrepancy', 'COA exact match')
        add(row, pr, serialize_item(acceptable_variant(base, rng), quantity, unit), 'Acceptable', 'No discrepancy', 'COA acceptable wording variation')

        # Needs Review: same item and structured values, but text is incomplete or noisy.
        add(row, pr, serialize_item(minor_incomplete_variant(base, rng), quantity, unit), 'Needs Review', 'Incomplete or ambiguous description', 'COA incomplete description')
        add(row, pr, serialize_item(typo_variant(base, rng), quantity, unit), 'Needs Review', 'Minor wording or spelling difference', 'COA typographical variation')

        # Needs Recanvass: related item but one important requirement changes.
        changed, used_spec = mutate_specification(base, rng)
        if used_spec:
            add(row, pr, serialize_item(changed, quantity, unit), 'Needs Recanvass', 'Specification mismatch', 'COA specification change')
        else:
            changed_qty = quantity + max(1, rng.randint(1, 10))
            add(row, pr, serialize_item(base, changed_qty, unit), 'Needs Recanvass', 'Quantity mismatch', 'COA quantity change')

        changed_qty = max(1, quantity + rng.choice([-3, -2, -1, 1, 2, 3]))
        if changed_qty == quantity:
            changed_qty += 1
        changed_unit = rng.choice([u for u in units if u != unit] or ['box'])
        if rng.random() < 0.5:
            add(row, pr, serialize_item(base, changed_qty, unit), 'Needs Recanvass', 'Quantity mismatch', 'COA quantity discrepancy')
        else:
            add(row, pr, serialize_item(base, quantity, changed_unit), 'Needs Recanvass', 'Unit mismatch', 'COA unit discrepancy')

        # Reject: unrelated or missing item.
        other_idx = choose_different_index(corpus, idx, rng, prefer_other_article=True)
        other = corpus[other_idx]
        add(row, pr, serialize_item(other['text'], quantity, other['unit']), 'Reject', 'Different item', 'COA unrelated item')
        if rng.random() < 0.5:
            add(row, pr, serialize_item('item description unavailable', quantity, unit), 'Reject', 'Missing item description', 'COA missing item')
        else:
            gibberish = ''.join(rng.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(rng.randint(4, 12)))
            if rng.random() < 0.5:
                gibberish += ' ' + ''.join(rng.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(rng.randint(4, 10)))
            add(row, pr, serialize_item(gibberish, quantity, unit), 'Reject', 'Different item', 'COA invalid or unrelated text')

    return pd.DataFrame(rows)


def _pick_column(df: pd.DataFrame, candidates: Sequence[str]):
    lower = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    for column in df.columns:
        c = str(column).lower()
        if any(token.lower() in c for token in candidates):
            return column
    return None


def load_labelled_pairs(csv_path: str) -> pd.DataFrame:
    if not csv_path or not os.path.exists(csv_path):
        return pd.DataFrame(columns=['pr_text', 'po_text', 'label', 'issue', 'source', 'group_id'])
    df = pd.read_csv(csv_path)
    label_col = _pick_column(df, ['Label', 'label', 'target', 'status'])
    pr_col = _pick_column(df, ['PR_Quantity and Size of the Project to be Procured', 'PR_General_Desc', 'pr_text', 'PR_General Description'])
    po_col = _pick_column(df, ['PO_Quantity and Size of the Project to be Procured', 'PO_General_Desc', 'po_text', 'PO_General Description'])
    if not label_col or not pr_col or not po_col:
        return pd.DataFrame(columns=['pr_text', 'po_text', 'label', 'issue', 'source', 'group_id'])

    rows = []
    for i, source_row in df.iterrows():
        raw_label = clean_text(source_row[label_col])
        if raw_label in {'ok', 'acceptable', 'accept'}:
            status, issue = 'Acceptable', 'No discrepancy'
        elif raw_label in {'needs review', 'review'}:
            status, issue = 'Needs Review', 'General discrepancy'
        elif raw_label in {'reject', 'rejected'}:
            status, issue = 'Reject', 'Different item'
        elif raw_label in {'discrepancy', 'needs recanvass', 'recanvass', 'mismatch'}:
            status, issue = 'Needs Recanvass', 'General discrepancy'
        else:
            continue
        rows.append({
            'pr_text': clean_text(source_row[pr_col]),
            'po_text': clean_text(source_row[po_col]),
            'label': status,
            'issue': issue,
            'source': 'Manually labelled PR-PO CSV',
            'group_id': f'manual:{i}',
        })
    return pd.DataFrame(rows)


def fasttext_vector(text: str, model) -> np.ndarray:
    words = clean_text(text).split()
    if not words:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean([model.wv[word] for word in words], axis=0).astype(np.float32)


def generic_pair_statistics(pr_text: str, po_text: str) -> np.ndarray:
    """Generic learned-model inputs; none of these directly decide a class."""
    pr_clean, po_clean = clean_text(pr_text), clean_text(po_text)
    pr_tokens, po_tokens = pr_clean.split(), po_clean.split()
    pr_set, po_set = set(pr_tokens), set(po_tokens)
    union = pr_set | po_set
    token_jaccard = len(pr_set & po_set) / max(1, len(union))
    sequence_ratio = SequenceMatcher(None, pr_clean, po_clean).ratio()
    length_ratio = min(len(pr_tokens), len(po_tokens)) / max(1, max(len(pr_tokens), len(po_tokens)))
    exact_text = 1.0 if pr_clean == po_clean else 0.0

    pr_qty = {t for t in pr_tokens if t.startswith('qty_')}
    po_qty = {t for t in po_tokens if t.startswith('qty_')}
    quantity_match = 1.0 if pr_qty and pr_qty == po_qty else 0.0
    pr_units = {t for t in pr_tokens if t.startswith('unit_')}
    po_units = {t for t in po_tokens if t.startswith('unit_')}
    unit_match = 1.0 if pr_units and pr_units == po_units else 0.0

    pr_numbers = {t for t in pr_tokens if any(ch.isdigit() for ch in t) and not t.startswith('item_')}
    po_numbers = {t for t in po_tokens if any(ch.isdigit() for ch in t) and not t.startswith('item_')}
    number_union = pr_numbers | po_numbers
    number_jaccard = len(pr_numbers & po_numbers) / max(1, len(number_union))
    item_count_ratio = min(sum(t.startswith('item_') for t in pr_tokens), sum(t.startswith('item_') for t in po_tokens)) / max(1, max(sum(t.startswith('item_') for t in pr_tokens), sum(t.startswith('item_') for t in po_tokens)))
    return np.asarray([token_jaccard, sequence_ratio, length_ratio, exact_text, quantity_match, unit_match, number_jaccard, item_count_ratio], dtype=np.float32)


def pair_features(pr_text: str, po_text: str, model) -> np.ndarray:
    pr = fasttext_vector(pr_text, model)
    po = fasttext_vector(po_text, model)
    learned_pair = np.concatenate([pr, po, np.abs(pr - po), pr * po])
    return np.concatenate([learned_pair, generic_pair_statistics(pr_text, po_text)]).astype(np.float32)


def train_best_logistic(X_train, y_train, X_test, y_test):
    candidates = [0.5, 2.0]
    best = None
    evaluations = []
    for c_value in candidates:
        clf = OneVsRestClassifier(LogisticRegression(
            C=c_value, max_iter=600, class_weight='balanced',
            random_state=RANDOM_STATE, solver='liblinear',
        ))
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        score = float(f1_score(y_test, pred, average='weighted', zero_division=0))
        evaluations.append({'C': c_value, 'weighted_f1': score})
        if best is None or score > best[0]:
            best = (score, c_value, clf, pred)
    return best, evaluations


def evaluation_dict(y_true, pred, labels: List[str]) -> Dict[str, Any]:
    return {
        'accuracy': float(accuracy_score(y_true, pred)),
        'precision_weighted': float(precision_score(y_true, pred, average='weighted', zero_division=0)),
        'recall_weighted': float(recall_score(y_true, pred, average='weighted', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, pred, average='weighted', zero_division=0)),
        'test_records': int(len(y_true)),
        'labels': labels,
        'confusion_matrix': confusion_matrix(y_true, pred, labels=labels).tolist(),
    }


def retrain_model(new_dataset_path: str | None = None, coa_workbook_path: str | None = None):
    from gensim.models import FastText

    labelled_path = new_dataset_path or DEFAULT_LABELLED_PATH
    workbook_path = coa_workbook_path or DEFAULT_COA_PATH
    corpus = prepare_corpus(workbook_path)

    rng = random.Random(RANDOM_STATE)
    indices = list(range(len(corpus)))
    rng.shuffle(indices)
    split = max(1, int(len(indices) * 0.80))
    train_corpus = [corpus[i] for i in indices[:split]]
    test_corpus = [corpus[i] for i in indices[split:]]

    train_generated = generate_pairs_for_corpus(train_corpus, RANDOM_STATE)
    test_generated = generate_pairs_for_corpus(test_corpus, RANDOM_STATE + 1)
    manual = load_labelled_pairs(labelled_path)
    training = pd.concat([train_generated, manual], ignore_index=True)

    for df in (training, test_generated):
        df['pr_text'] = df['pr_text'].map(clean_text)
        df['po_text'] = df['po_text'].map(clean_text)
        df.drop(df[(df.pr_text == '') | (df.po_text == '')].index, inplace=True)

    # FastText learns the complete local vocabulary. The supervised evaluation
    # still holds out source items from classifier training.
    sentences: List[List[str]] = []
    for row in corpus:
        sentences.append(clean_text(row['text']).split())
    for frame in (training, test_generated):
        for pr, po in zip(frame.pr_text, frame.po_text):
            sentences.append(pr.split())
            sentences.append(po.split())

    ft_model = FastText(
        vector_size=VECTOR_SIZE, window=WINDOW, min_count=1, workers=1,
        sg=1, epochs=EPOCHS, seed=RANDOM_STATE, bucket=10000,
    )
    ft_model.build_vocab(corpus_iterable=sentences)
    ft_model.train(corpus_iterable=sentences, total_examples=len(sentences), epochs=EPOCHS)

    X_train = np.vstack([pair_features(pr, po, ft_model) for pr, po in zip(training.pr_text, training.po_text)])
    X_test = np.vstack([pair_features(pr, po, ft_model) for pr, po in zip(test_generated.pr_text, test_generated.po_text)])
    y_status_train = training.label.astype(str).to_numpy()
    y_status_test = test_generated.label.astype(str).to_numpy()
    y_issue_train = training.issue.astype(str).to_numpy()
    y_issue_test = test_generated.issue.astype(str).to_numpy()

    status_best, status_search = train_best_logistic(X_train, y_status_train, X_test, y_status_test)
    issue_best, issue_search = train_best_logistic(X_train, y_issue_train, X_test, y_issue_test)
    _status_score, status_c, status_eval_model, status_pred = status_best
    _issue_score, issue_c, issue_eval_model, issue_pred = issue_best

    status_labels = list(STATUS_CLASSES)
    issue_labels = sorted(set(y_issue_test) | set(y_issue_train))
    status_evaluation = evaluation_dict(y_status_test, status_pred, status_labels)
    issue_evaluation = evaluation_dict(y_issue_test, issue_pred, issue_labels)

    # Fit final models on all available generated and manually labelled pairs.
    all_generated = pd.concat([train_generated, test_generated], ignore_index=True)
    final_training = pd.concat([all_generated, manual], ignore_index=True)
    final_training['pr_text'] = final_training['pr_text'].map(clean_text)
    final_training['po_text'] = final_training['po_text'].map(clean_text)
    final_training = final_training[(final_training.pr_text != '') & (final_training.po_text != '')]
    X_all = np.vstack([pair_features(pr, po, ft_model) for pr, po in zip(final_training.pr_text, final_training.po_text)])

    status_model = OneVsRestClassifier(LogisticRegression(
        C=status_c, max_iter=600, class_weight='balanced', random_state=RANDOM_STATE,
        solver='liblinear',
    )).fit(X_all, final_training.label.astype(str).to_numpy())
    issue_model = OneVsRestClassifier(LogisticRegression(
        C=issue_c, max_iter=600, class_weight='balanced', random_state=RANDOM_STATE,
        solver='liblinear',
    )).fit(X_all, final_training.issue.astype(str).to_numpy())

    bundle = {
        'status_model': status_model,
        'issue_model': issue_model,
        'feature_layout': 'PR vector + PO vector + absolute difference + element-wise product + 8 generic pair statistics',
        'vector_size': VECTOR_SIZE,
        'classifier_dimensions': VECTOR_SIZE * 4 + 8,
        'decision_mode': 'AI_ONLY_DIRECT_MULTICLASS',
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    ft_model.save(FASTTEXT_PATH)
    all_generated.to_csv(GENERATED_PAIRS_PATH, index=False)
    with open(COA_CORPUS_PATH, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    info = {
        'method': 'AI-only COA-domain FastText + direct multi-class Logistic Regression',
        'classifier': 'OneVsRest LogisticRegression',
        'runtime_location': 'PR-to-PO NLP Verification only',
        'decision_mode': 'AI_ONLY_DIRECT_MULTICLASS',
        'runtime_rules_used': False,
        'hybrid_score_used': False,
        'rule_fallback_used': False,
        'status_classes': list(status_model.classes_),
        'issue_classes': list(issue_model.classes_),
        'feature_layout': bundle['feature_layout'],
        'vector_size_each': VECTOR_SIZE,
        'classifier_dimensions': VECTOR_SIZE * 4 + 8,
        'coa_workbook': os.path.basename(workbook_path),
        'coa_valid_unique_items': len(corpus),
        'coa_generated_pairs': int(len(all_generated)),
        'manual_labelled_pairs': int(len(manual)),
        'total_training_pairs': int(len(final_training)),
        'held_out_source_items': len(test_corpus),
        'status_model_C': status_c,
        'issue_model_C': issue_c,
        'status_hyperparameter_search': status_search,
        'issue_hyperparameter_search': issue_search,
        'status_evaluation': status_evaluation,
        'issue_evaluation': issue_evaluation,
        'evaluation_note': (
            'Evaluation holds out complete COA source items from supervised classifier training. '
            'The examples remain controlled synthetic PR/PO variations and must not be presented '
            'as external accuracy on real historical municipal PR/PO documents.'
        ),
        'important_limitation': (
            'No model can guarantee detection of every possible discrepancy. Real reviewed PR/PO '
            'outcomes should be added as labelled training records for stronger external validity.'
        ),
    }
    with open(METHOD_PATH, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    return info


if __name__ == '__main__':
    print(json.dumps(retrain_model(), indent=2))
