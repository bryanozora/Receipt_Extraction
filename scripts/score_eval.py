"""Step 9, part 2: score saved extraction results against ground truth.

Run from the repo root:
    python scripts/score_eval.py <run_id>

Reads whatever extraction results already exist under
outputs/eval_runs/<run_id>/ (written by run_eval_extractions.py) and
data/ground_truth/ -- purely a comparison of two sets of JSON files already
on disk, no API calls and no dependency on the extraction code itself. A
receipt with no saved result yet (run still in progress) is simply skipped
and counted as "pending" rather than erroring, so partial progress can be
scored without waiting for the full run to finish.

Ground-truth readiness
------------------------
A ground truth file where every top-level field is still the blank
{"value": null, ...} template (not yet hand-labeled) is skipped entirely
rather than scored -- treating it literally would mean "ground truth says
no field has a value anywhere", so every real extracted value would be
misclassified as hallucinated. A ground truth entry only counts as ready
once at least one top-level field has been filled in.

Field-level categories
-------------------------
Every top-level field, and every field within an aligned line-item row, is
classified as exactly one of:
- correct: both sides agree there's a value and it matches, OR both sides
  agree there's no value and agree on why (same status).
- wrong_value: both sides have a value, but it doesn't match.
- hallucinated: ground truth says there's no value (not_present, or
  illegible -- unverifiable either way) but the extraction produced one.
- wrongly_abstained: ground truth has a real value but the extraction
  didn't produce one.
- missed_status: both sides agree there's no value, but disagree on why
  (e.g. ground truth says not_present, extraction says illegible).

Line items are aligned first (by description similarity + amount
closeness, since row order/count can differ), then matched rows are scored
field-by-field the same way as top-level fields. Unmatched rows count too
(an unmatched ground-truth row's populated fields are wrongly_abstained; an
unmatched extracted row's populated fields are hallucinated) -- excluding
them would let a system that just drops line items score perfectly -- and
are additionally listed by index so they're visible on their own, not only
folded into the aggregate counts.

Precision/recall per field name treats "correct" (with a real value) as a
true positive, hallucinated/wrong_value as contributing false positives,
and wrongly_abstained/wrong_value as contributing false negatives. Two
correctly-agreeing abstentions (correct with no value, or missed_status)
are true negatives and don't enter precision/recall at all.
"""

import argparse
import difflib
import re
import json
import os

GROUND_TRUTH_DIR = os.path.join("data", "ground_truth")
EVAL_RUNS_DIR = os.path.join("outputs", "eval_runs")

NUMERIC_TOLERANCE = 1.0
MIN_LINE_ITEM_MATCH_SCORE = 0.3

TOP_LEVEL_STRING_FIELDS = ["vendor_name", "receipt_number", "currency"]
TOP_LEVEL_DATE_FIELDS = ["transaction_date"]
TOP_LEVEL_NUMERIC_FIELDS = ["subtotal", "tax", "total"]
TOP_LEVEL_FIELDS = TOP_LEVEL_STRING_FIELDS + TOP_LEVEL_DATE_FIELDS + TOP_LEVEL_NUMERIC_FIELDS

LINE_ITEM_STRING_FIELDS = ["description"]
LINE_ITEM_NUMERIC_FIELDS = ["quantity", "unit_price", "amount"]
LINE_ITEM_FIELDS = LINE_ITEM_STRING_FIELDS + LINE_ITEM_NUMERIC_FIELDS

CATEGORIES = ["correct", "wrong_value", "hallucinated", "wrongly_abstained", "missed_status"]


# --------------------------------------------------------------------------
# Field-level value matching
# --------------------------------------------------------------------------


def _normalize_str(s):
    return s.strip().lower() if isinstance(s, str) else s


# Cosmetic punctuation that shows up inconsistently between ground truth and
# extracted text for the same real content -- an inch mark rendered as a
# straight quote (") on one side and an asterisk (*) on the other, a
# trailing "**" promo marker, a stray apostrophe/backtick, etc. Stripped only
# for string-field *correctness* matching (below), not for _normalize_str
# itself, which line-item alignment also uses and which shouldn't change.
_COSMETIC_SYMBOLS_PATTERN = re.compile(r"[*\"'`´′″]+")


def _strip_cosmetic_symbols(s):
    if not isinstance(s, str):
        return s
    return re.sub(r"\s+", " ", _COSMETIC_SYMBOLS_PATTERN.sub("", s)).strip()


def _normalize_for_matching(s):
    return _strip_cosmetic_symbols(_normalize_str(s))


def _string_matches(ex_value, gt_value, gt_alternatives) -> bool:
    candidates = [gt_value, *(gt_alternatives or [])]
    ex_norm = _normalize_for_matching(ex_value)
    return any(ex_norm == _normalize_for_matching(c) for c in candidates if c is not None)


def _numeric_matches(ex_value, gt_value, gt_alternatives) -> bool:
    candidates = [gt_value, *(gt_alternatives or [])]
    try:
        return any(c is not None and abs(float(ex_value) - float(c)) <= NUMERIC_TOLERANCE for c in candidates)
    except (TypeError, ValueError):
        return False


def _date_matches(ex_value, gt_value, gt_alternatives) -> bool:
    candidates = [gt_value, *(gt_alternatives or [])]
    return any(ex_value == c for c in candidates if c is not None)


FIELD_MATCHERS = {}
for _f in TOP_LEVEL_STRING_FIELDS + LINE_ITEM_STRING_FIELDS:
    FIELD_MATCHERS[_f] = _string_matches
for _f in TOP_LEVEL_DATE_FIELDS:
    FIELD_MATCHERS[_f] = _date_matches
for _f in TOP_LEVEL_NUMERIC_FIELDS + LINE_ITEM_NUMERIC_FIELDS:
    FIELD_MATCHERS[_f] = _numeric_matches


def _has_value(field: dict) -> bool:
    return field.get("status") == "present" and field.get("value") is not None


def _classify_field(ex_field: dict, gt_field: dict, matcher) -> str:
    gt_has = _has_value(gt_field)
    ex_has = _has_value(ex_field)

    if gt_has and ex_has:
        try:
            matched = matcher(ex_field["value"], gt_field["value"], gt_field.get("acceptable_alternatives"))
        except Exception:
            matched = False
        return "correct" if matched else "wrong_value"
    if gt_has and not ex_has:
        return "wrongly_abstained"
    if not gt_has and ex_has:
        return "hallucinated"
    # Neither side has a value -- ground truth "illegible" also lands here,
    # since it means no verifiable value exists to check against either.
    return "correct" if ex_field.get("status") == gt_field.get("status") else "missed_status"


# --------------------------------------------------------------------------
# Line-item row alignment
# --------------------------------------------------------------------------


def _line_item_similarity(gt_item: dict, ex_item: dict) -> float:
    desc_sim = 0.0
    gt_desc, ex_desc = gt_item["description"].get("value"), ex_item["description"].get("value")
    if isinstance(gt_desc, str) and isinstance(ex_desc, str):
        desc_sim = difflib.SequenceMatcher(None, _normalize_str(gt_desc), _normalize_str(ex_desc)).ratio()

    amount_match = 0.0
    gt_amount, ex_amount = gt_item["amount"].get("value"), ex_item["amount"].get("value")
    if isinstance(gt_amount, (int, float)) and isinstance(ex_amount, (int, float)):
        if abs(gt_amount - ex_amount) <= NUMERIC_TOLERANCE:
            amount_match = 1.0

    return 0.6 * desc_sim + 0.4 * amount_match


def _match_line_items(gt_items: list, ex_items: list):
    """Greedy highest-similarity-first bipartite matching (small n per receipt)."""
    candidates = []
    for gi, gt_item in enumerate(gt_items):
        for ei, ex_item in enumerate(ex_items):
            score = _line_item_similarity(gt_item, ex_item)
            if score >= MIN_LINE_ITEM_MATCH_SCORE:
                candidates.append((score, gi, ei))
    candidates.sort(key=lambda c: c[0], reverse=True)

    matched_gt, matched_ex, matches = set(), set(), []
    for score, gi, ei in candidates:
        if gi in matched_gt or ei in matched_ex:
            continue
        matched_gt.add(gi)
        matched_ex.add(ei)
        matches.append((gi, ei, score))

    unmatched_gt = [i for i in range(len(gt_items)) if i not in matched_gt]
    unmatched_ex = [i for i in range(len(ex_items)) if i not in matched_ex]
    return matches, unmatched_gt, unmatched_ex


# --------------------------------------------------------------------------
# Accumulation (category counts + per-field precision/recall)
# --------------------------------------------------------------------------


class Accumulator:
    def __init__(self):
        self.category_counts = {c: 0 for c in CATEGORIES}
        self.field_stats: dict[str, dict[str, int]] = {}

    def record(self, field_name: str, category: str) -> None:
        self.category_counts[category] += 1
        stats = self.field_stats.setdefault(field_name, {"tp": 0, "fp": 0, "fn": 0})
        if category == "correct":
            stats["tp"] += 1  # only reached for a genuine value-match; see caller
        elif category == "hallucinated":
            stats["fp"] += 1
        elif category == "wrongly_abstained":
            stats["fn"] += 1
        elif category == "wrong_value":
            stats["fp"] += 1
            stats["fn"] += 1
        # "correct" (both-abstained) and "missed_status" are true negatives.

    def merge(self, other: "Accumulator") -> None:
        for c, n in other.category_counts.items():
            self.category_counts[c] += n
        for field_name, s in other.field_stats.items():
            mine = self.field_stats.setdefault(field_name, {"tp": 0, "fp": 0, "fn": 0})
            for k in ("tp", "fp", "fn"):
                mine[k] += s[k]


def _record(acc: Accumulator, field_name: str, ex_field: dict, gt_field: dict) -> str:
    """Classify one field and record it, correctly distinguishing a genuine
    value-match "correct" (a true positive) from a both-abstained "correct"
    (a true negative, not counted in precision/recall)."""
    category = _classify_field(ex_field, gt_field, FIELD_MATCHERS[field_name])
    if category == "correct" and not _has_value(gt_field):
        acc.category_counts["correct"] += 1  # both sides correctly abstained -- true negative
    else:
        acc.record(field_name, category)
    return category


# --------------------------------------------------------------------------
# Per-document scoring
# --------------------------------------------------------------------------


def _ground_truth_is_ready(gt: dict) -> bool:
    return any(_has_value(gt[f]) for f in TOP_LEVEL_FIELDS)


def _score_document(gt: dict, ex: dict) -> dict:
    acc = Accumulator()
    fields = {}

    for field_name in TOP_LEVEL_FIELDS:
        gt_field, ex_field = gt[field_name], ex[field_name]
        category = _record(acc, field_name, ex_field, gt_field)
        fields[field_name] = {
            "category": category,
            "ground_truth_value": gt_field.get("value"),
            "extracted_value": ex_field.get("value"),
            "ground_truth_status": gt_field.get("status"),
            "extracted_status": ex_field.get("status"),
        }

    gt_items = gt.get("line_items", [])
    ex_items = ex.get("line_items", [])
    matches, unmatched_gt, unmatched_ex = _match_line_items(gt_items, ex_items)

    matched_rows = []
    for gi, ei, score in matches:
        gt_item, ex_item = gt_items[gi], ex_items[ei]
        row_fields = {}
        for field_name in LINE_ITEM_FIELDS:
            gt_field, ex_field = gt_item[field_name], ex_item[field_name]
            category = _record(acc, field_name, ex_field, gt_field)
            row_fields[field_name] = {
                "category": category,
                "ground_truth_value": gt_field.get("value"),
                "extracted_value": ex_field.get("value"),
            }
        matched_rows.append(
            {"ground_truth_index": gi, "extracted_index": ei, "match_score": round(score, 3), "fields": row_fields}
        )

    unmatched_gt_rows = []
    for gi in unmatched_gt:
        gt_item = gt_items[gi]
        missed_fields = [f for f in LINE_ITEM_FIELDS if _has_value(gt_item[f])]
        for field_name in missed_fields:
            acc.record(field_name, "wrongly_abstained")
        unmatched_gt_rows.append(
            {"index": gi, "description": gt_item["description"].get("value"), "fields_missed": missed_fields}
        )

    unmatched_ex_rows = []
    for ei in unmatched_ex:
        ex_item = ex_items[ei]
        hallucinated_fields = [f for f in LINE_ITEM_FIELDS if _has_value(ex_item[f])]
        for field_name in hallucinated_fields:
            acc.record(field_name, "hallucinated")
        unmatched_ex_rows.append(
            {"index": ei, "description": ex_item["description"].get("value"), "fields_hallucinated": hallucinated_fields}
        )

    return {
        "status": "scored",
        "fields": fields,
        "line_items": {
            "matched": matched_rows,
            "unmatched_ground_truth": unmatched_gt_rows,
            "unmatched_extracted": unmatched_ex_rows,
        },
        "category_counts": acc.category_counts,
        "_accumulator": acc,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Score saved extraction results against ground truth.")
    parser.add_argument("run_id", help="Eval run to score (outputs/eval_runs/<run_id>/).")
    args = parser.parse_args()

    run_dir = os.path.join(EVAL_RUNS_DIR, args.run_id)
    receipt_ids = sorted(
        os.path.splitext(n)[0] for n in os.listdir(GROUND_TRUTH_DIR) if n.endswith(".json")
    )

    global_acc = Accumulator()
    per_document_results = {}
    extraction_failed_breakdown: dict[str, int] = {}
    counts = {
        "scored": 0,
        "extraction_failed": 0,
        "pending_not_yet_extracted": 0,
        "ground_truth_not_ready": 0,
    }
    total_unmatched_gt_items = 0
    total_unmatched_ex_items = 0

    for receipt_id in receipt_ids:
        with open(os.path.join(GROUND_TRUTH_DIR, f"{receipt_id}.json"), encoding="utf-8") as f:
            gt = json.load(f)

        if not _ground_truth_is_ready(gt):
            counts["ground_truth_not_ready"] += 1
            per_document_results[receipt_id] = {"status": "ground_truth_not_ready"}
            continue

        result_path = os.path.join(run_dir, f"{receipt_id}.json")
        if not os.path.exists(result_path):
            counts["pending_not_yet_extracted"] += 1
            per_document_results[receipt_id] = {"status": "pending_not_yet_extracted"}
            continue

        with open(result_path, encoding="utf-8") as f:
            saved = json.load(f)

        if saved["result_type"] == "ExtractionError":
            counts["extraction_failed"] += 1
            failure_stage = saved["result"]["failure_stage"]
            extraction_failed_breakdown[failure_stage] = extraction_failed_breakdown.get(failure_stage, 0) + 1
            per_document_results[receipt_id] = {
                "status": "extraction_failed",
                "failure_stage": failure_stage,
                "message": saved["result"]["message"],
            }
            continue

        doc_result = _score_document(gt, saved["result"])
        global_acc.merge(doc_result.pop("_accumulator"))
        per_document_results[receipt_id] = doc_result
        counts["scored"] += 1
        total_unmatched_gt_items += len(doc_result["line_items"]["unmatched_ground_truth"])
        total_unmatched_ex_items += len(doc_result["line_items"]["unmatched_extracted"])

    precision_recall = {}
    for field_name, s in global_acc.field_stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        precision_recall[field_name] = {
            "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
            "recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    total_fields_judged = sum(global_acc.category_counts.values())
    accuracy = global_acc.category_counts["correct"] / total_fields_judged if total_fields_judged else None

    scorecard = {
        "run_id": args.run_id,
        "receipts_total": len(receipt_ids),
        "receipts_scored": counts["scored"],
        "receipts_extraction_failed": counts["extraction_failed"],
        "receipts_pending_not_yet_extracted": counts["pending_not_yet_extracted"],
        "receipts_ground_truth_not_ready": counts["ground_truth_not_ready"],
        "field_category_counts": global_acc.category_counts,
        "overall_field_accuracy": round(accuracy, 4) if accuracy is not None else None,
        "extraction_failed_breakdown": extraction_failed_breakdown,
        "line_items_unmatched": {
            "ground_truth_rows_missed": total_unmatched_gt_items,
            "extracted_rows_hallucinated": total_unmatched_ex_items,
        },
        "precision_recall_by_field": precision_recall,
    }

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "scorecard.json"), "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)
        f.write("\n")
    with open(os.path.join(run_dir, "per_document_results.json"), "w", encoding="utf-8") as f:
        json.dump(per_document_results, f, indent=2)
        f.write("\n")

    # --- console summary ---
    print(f"Run ID: {args.run_id}")
    print(
        f"Receipts: {len(receipt_ids)} total | {counts['scored']} scored | "
        f"{counts['extraction_failed']} extraction_failed | "
        f"{counts['pending_not_yet_extracted']} pending | "
        f"{counts['ground_truth_not_ready']} ground_truth_not_ready"
    )
    print()
    if accuracy is not None:
        print(f"Overall field accuracy: {accuracy:.1%} ({global_acc.category_counts['correct']}/{total_fields_judged})")
    else:
        print("Overall field accuracy: n/a (no receipts scored yet)")
    print("Field-level category breakdown:")
    for category in CATEGORIES:
        print(f"  {category:18s} {global_acc.category_counts[category]}")
    print(
        f"Line items: {total_unmatched_gt_items} ground-truth row(s) missed, "
        f"{total_unmatched_ex_items} extracted row(s) hallucinated"
    )
    if extraction_failed_breakdown:
        print("Extraction failures by stage:", extraction_failed_breakdown)
    print()
    print(f"Scorecard:          {os.path.join(run_dir, 'scorecard.json')}")
    print(f"Per-document detail: {os.path.join(run_dir, 'per_document_results.json')}")


if __name__ == "__main__":
    main()
