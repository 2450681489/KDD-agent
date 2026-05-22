import os
import pandas as pd
import json
import re
from collections import Counter
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

PRED_DIR = "tmp_output"
GOLD_DIR = "data/public/output"

NULL_STRINGS = {"", "null", "none", "nan", "nat", "<na>"}
DECIMAL_QUANT = Decimal("0.01")
DATE_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
DATETIME_HINT_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}[ T]\d{1,2}:\d{1,2}")
LETTER_RE = re.compile(r"[A-Za-z\u4e00-\u9fff]")


# =========================
# normalization
# =========================
def normalize_value(v):
    """
    Normalize values before building column signatures.
    """
    if pd.isna(v):
        return ""

    text = str(v).strip(" \r\n\t")
    if text.lower() in NULL_STRINGS:
        return ""

    date_value = _normalize_date(text)
    if date_value is not None:
        return date_value

    datetime_value = _normalize_datetime(text)
    if datetime_value is not None:
        return datetime_value

    number_value = _normalize_number(text)
    if number_value is not None:
        return number_value

    return text


def _normalize_number(text):
    try:
        value = Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    return str(value.quantize(DECIMAL_QUANT, rounding=ROUND_HALF_UP))


def _normalize_date(text):
    match = DATE_RE.match(text)
    if match is None:
        return None
    try:
        year, month, day = (int(part) for part in match.groups())
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _normalize_datetime(text):
    if DATETIME_HINT_RE.match(text) is None:
        return None

    candidate = text.replace(" ", "T", 1)
    if candidate.endswith("Z"):
        candidate = f"{candidate[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.isoformat()

    utc_value = parsed.astimezone(timezone.utc)
    rendered = utc_value.isoformat().replace("+00:00", "Z")
    return rendered


def load_csv(path):
    df = pd.read_csv(path, keep_default_na=False)
    return df


def df_to_serializable(df):
    return {
        "columns": df.columns.tolist(),
        "rows": df.values.tolist()
    }


# =========================
# core comparison logic
# =========================
def compare_by_content_inclusion(pred_df, gold_df):
    """
    Compare by column-level content signatures.

    Column names and column order are ignored. Row order is ignored by sorting
    normalized cell values inside each column signature. Duplicate values inside
    a column and duplicate columns are both preserved.
    """

    pred_signatures = build_column_signatures(pred_df)
    gold_signatures = build_column_signatures(gold_df)
    matched_count, missing_signatures = count_signature_coverage(
        pred_signatures,
        gold_signatures,
    )
    total_gold_columns = len(gold_signatures)
    accuracy = matched_count / total_gold_columns if total_gold_columns else 1.0

    if matched_count == total_gold_columns:
        return True, None, None, 1.0

    # 名称字段允许拆列/合列；这里只用内容生成候选签名，不依赖列名。
    if _match_gold_signatures(
        build_column_match_options(gold_df),
        build_column_match_options(pred_df),
        len(gold_df.columns),
    ):
        return True, None, None, 1.0

    return False, "missing_or_mismatched_gold_column", {
        "pred_columns": pred_df.columns.tolist(),
        "gold_columns": gold_df.columns.tolist(),
        "pred_data": df_to_serializable(pred_df),
        "gold_data": df_to_serializable(gold_df),
        "pred_signatures": _signature_debug(pred_signatures),
        "gold_signatures": _signature_debug(gold_signatures),
        "missing_signatures": _missing_signature_debug(missing_signatures),
    }, accuracy


def build_column_signatures(df):
    signatures = []
    normalized_columns = [
        [normalize_value(v) for v in df[column].tolist()]
        for column in df.columns
    ]

    for index, values in enumerate(normalized_columns):
        signatures.append({
            "columns": frozenset({index}),
            "label": df.columns[index],
            "kind": "single",
            "values": _column_signature(values),
        })

    return signatures


def build_column_match_options(df):
    signatures = build_column_signatures(df)
    normalized_columns = [
        [normalize_value(v) for v in df[column].tolist()]
        for column in df.columns
    ]

    for left_index in range(len(normalized_columns)):
        for right_index in range(left_index + 1, len(normalized_columns)):
            left_values = normalized_columns[left_index]
            right_values = normalized_columns[right_index]
            if not _looks_like_name_pair(left_values, right_values):
                continue

            for first_index, last_index in (
                (left_index, right_index),
                (right_index, left_index),
            ):
                full_names = [
                    _join_name_parts(first, last)
                    for first, last in zip(
                        normalized_columns[first_index],
                        normalized_columns[last_index],
                        strict=False,
                    )
                ]
                signatures.append({
                    "columns": frozenset({left_index, right_index}),
                    "label": f"col#{first_index}+col#{last_index}",
                    "kind": "full_name",
                    "values": _column_signature(full_names),
                })

    return signatures


def _column_signature(values):
    return tuple(sorted(values))


def count_signature_coverage(pred_signatures, gold_signatures):
    pred_counts = Counter(signature["values"] for signature in pred_signatures)
    gold_counts = Counter(signature["values"] for signature in gold_signatures)

    matched_count = 0
    missing = {}
    for signature, gold_count in gold_counts.items():
        pred_count = pred_counts.get(signature, 0)
        matched_count += min(pred_count, gold_count)
        if pred_count < gold_count:
            missing[signature] = gold_count - pred_count

    return matched_count, missing


def _looks_like_name_pair(left_values, right_values):
    checked = 0
    name_like = 0
    for left, right in zip(left_values, right_values, strict=False):
        if not left or not right:
            continue
        checked += 1
        if _looks_like_name_part(left) and _looks_like_name_part(right):
            name_like += 1

    return checked > 0 and name_like / checked >= 0.8


def _looks_like_name_part(value):
    text = str(value).strip()
    if not text:
        return False
    if any(char.isdigit() for char in text):
        return False
    return LETTER_RE.search(text) is not None


def _join_name_parts(first, last):
    if first and last:
        return f"{first} {last}"
    return first or last


def _match_gold_signatures(gold_signatures, pred_signatures, gold_column_count):
    target_columns = frozenset(range(gold_column_count))

    def search(covered_gold_columns, used_pred_columns):
        if covered_gold_columns == target_columns:
            return True

        next_column = min(target_columns - covered_gold_columns)
        candidate_gold_signatures = [
            signature for signature in gold_signatures
            if next_column in signature["columns"]
        ]
        candidate_gold_signatures.sort(key=lambda item: len(item["columns"]), reverse=True)

        for gold_signature in candidate_gold_signatures:
            for pred_signature in pred_signatures:
                if pred_signature["columns"] & used_pred_columns:
                    continue
                if gold_signature["values"] != pred_signature["values"]:
                    continue
                if search(
                    covered_gold_columns | gold_signature["columns"],
                    used_pred_columns | pred_signature["columns"],
                ):
                    return True
        return False

    return search(frozenset(), frozenset())


def _signature_debug(signatures):
    return [
        {
            "label": signature["label"],
            "kind": signature["kind"],
            "columns": sorted(signature["columns"]),
            "values": list(signature["values"]),
        }
        for signature in signatures
    ]


def _missing_signature_debug(missing_signatures):
    return [
        {
            "missing_count": count,
            "values": list(signature),
        }
        for signature, count in missing_signatures.items()
    ]


# =========================
# main evaluation loop
# =========================
def main():
    tasks = [
        t for t in os.listdir(PRED_DIR)
        if os.path.isdir(os.path.join(PRED_DIR, t))
        and t.startswith("task_")
    ]

    report = {}

    match_tasks = []
    mismatch_tasks = []
    mismatch_details = {}

    for task in sorted(tasks):
        pred_path = os.path.join(PRED_DIR, task, "prediction.csv")
        gold_path = os.path.join(GOLD_DIR, task, "gold.csv")

        if not os.path.exists(pred_path):
            mismatch_tasks.append(task)
            mismatch_details[task] = {
                "correct": False,
                "error_type": "missing_prediction",
                "error_detail": "prediction.csv not found",
                "accuracy": 0.0
            }
            continue

        try:
            pred_df = load_csv(pred_path)
            gold_df = load_csv(gold_path)

            ok, err_type, detail, acc = compare_by_content_inclusion(pred_df, gold_df)

            if ok:
                match_tasks.append(task)
            else:
                mismatch_tasks.append(task)
                mismatch_details[task] = {
                    "correct": False,
                    "error_type": err_type,
                    "error_detail": detail,
                    "accuracy": acc
                }

        except Exception as e:
            mismatch_tasks.append(task)
            mismatch_details[task] = {
                "correct": False,
                "error_type": "runtime_error",
                "error_detail": str(e),
                "accuracy": 0.0
            }

    # =========================
    # 构建最终报告
    # =========================
    final_report = {
        "match": {
            "count": len(match_tasks),
            "tasks": match_tasks
        },
        "mismatch": {
            "count": len(mismatch_tasks),
            "tasks": mismatch_tasks,
            "details": mismatch_details   # ✅ 只在这里放详细信息
        }
    }

    # =========================
    # console summary
    # =========================
    print("\n===== SUMMARY =====")
    print(f"Match: {len(match_tasks)}")
    print(f"Mismatch: {len(mismatch_tasks)}")

    print("\n===== MISMATCH TASKS =====")
    for t in mismatch_tasks:
        print(t)

    # =========================
    # save report
    # =========================
    with open("per_task_report_0518_2.json", "w") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print("\nSaved: per_task_report_0518_2.json")
    
if __name__ == "__main__":
    main()
