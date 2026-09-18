"""Step 13: compare two eval runs and catch regressions before they ship.

Run from the repo root:
    python scripts/diff_runs.py <run1_id> <run2_id>

Reads only outputs/eval_runs/<run_id>/scorecard.json and
per_document_results.json for both runs (both written by score_eval.py) --
no API calls, no dependency on extract.py or score_eval.py's internals.
This automates a comparison that was previously done by hand several times
during Step 9 (default vs prompt_v2 vs prompt_v3).

Regression / improvement detection
-------------------------------------
For every field in every receipt that both runs actually scored, this
compares its category between the two runs:
- regression: "correct" in run1, anything else in run2.
- improvement: anything else in run1, "correct" in run2.

Line-item fields are matched across runs by ground-truth row index, not
extracted-row index -- ground truth doesn't change between runs, but which
extracted row aligns to which ground-truth row can, so extracted-row index
isn't a stable identity to diff on. A ground-truth row that went unmatched
(wrongly_abstained) still gets a synthetic "wrongly_abstained" category per
missing field via that same gt-index key, so a line item that disappears
entirely between runs still shows up as a regression per field, not silently
dropped.

Line items an extraction *hallucinated* (rows with no ground-truth match at
all) are deliberately NOT diffed field-by-field: there's no principled way
to say "run1's invented row #3" is the "same" item as "run2's invented row
#1" -- there's no ground-truth anchor for either. Their counts are still
included in each receipt's line-item summary, just not as regression/
improvement entries.

A receipt whose overall status differs between runs (e.g. scored in run1,
extraction_failed in run2) is flagged separately as a status change. Its
fields still show up as regressions/improvements too where comparable,
since "this field is no longer correct" is true regardless of why.
"""

import argparse
import json
import os

EVAL_RUNS_DIR = os.path.join("outputs", "eval_runs")

CATEGORIES = ["correct", "wrong_value", "hallucinated", "wrongly_abstained", "missed_status"]


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise SystemExit(f"Missing file: {path} -- run score_eval.py for this run_id first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _flatten_document_fields(doc: dict) -> dict[str, dict]:
    """Map field_key -> {category, extracted_value, ground_truth_value} for
    one scored document. Returns {} for a receipt that wasn't scored in
    this run (extraction_failed / pending / ground_truth_not_ready)."""
    if doc.get("status") != "scored":
        return {}

    flat: dict[str, dict] = {}

    for field_name, info in doc["fields"].items():
        flat[field_name] = {
            "category": info["category"],
            "extracted_value": info["extracted_value"],
            "ground_truth_value": info["ground_truth_value"],
        }

    for row in doc["line_items"]["matched"]:
        gi = row["ground_truth_index"]
        for field_name, info in row["fields"].items():
            flat[f"line_item[gt={gi}].{field_name}"] = {
                "category": info["category"],
                "extracted_value": info["extracted_value"],
                "ground_truth_value": info["ground_truth_value"],
            }

    for row in doc["line_items"]["unmatched_ground_truth"]:
        gi = row["index"]
        for field_name in row["fields_missed"]:
            flat[f"line_item[gt={gi}].{field_name}"] = {
                "category": "wrongly_abstained",
                "extracted_value": None,
                # Not recorded in per_document_results.json for unmatched rows;
                # this script deliberately doesn't read data/ground_truth/ to
                # recover it (scoped to the two runs' saved output only).
                "ground_truth_value": None,
            }

    return flat


def _diff_receipt(receipt_id: str, doc1: dict, doc2: dict) -> tuple[list, list]:
    regressions, improvements = [], []
    status1, status2 = doc1.get("status"), doc2.get("status")

    flat1 = _flatten_document_fields(doc1)
    flat2 = _flatten_document_fields(doc2)

    for key in sorted(set(flat1) | set(flat2)):
        entry1 = flat1.get(key)
        entry2 = flat2.get(key)
        cat1 = entry1["category"] if entry1 else f"not_scored ({status1})"
        cat2 = entry2["category"] if entry2 else f"not_scored ({status2})"

        was_correct = cat1 == "correct"
        is_correct = cat2 == "correct"
        if was_correct == is_correct:
            continue

        record = {
            "receipt_id": receipt_id,
            "field": key,
            "ground_truth_value": (entry2 or entry1 or {}).get("ground_truth_value"),
            "run1_category": cat1,
            "run1_value": entry1["extracted_value"] if entry1 else None,
            "run2_category": cat2,
            "run2_value": entry2["extracted_value"] if entry2 else None,
        }
        (regressions if was_correct else improvements).append(record)

    return regressions, improvements


def main() -> None:
    parser = argparse.ArgumentParser(description="Diff two eval runs and flag regressions/improvements.")
    parser.add_argument("run1_id", help="Baseline run (e.g. the 'before').")
    parser.add_argument("run2_id", help="Comparison run (e.g. the 'after').")
    args = parser.parse_args()
    run1_id, run2_id = args.run1_id, args.run2_id

    scorecard1 = _load_json(os.path.join(EVAL_RUNS_DIR, run1_id, "scorecard.json"))
    scorecard2 = _load_json(os.path.join(EVAL_RUNS_DIR, run2_id, "scorecard.json"))
    per_doc1 = _load_json(os.path.join(EVAL_RUNS_DIR, run1_id, "per_document_results.json"))
    per_doc2 = _load_json(os.path.join(EVAL_RUNS_DIR, run2_id, "per_document_results.json"))

    # --- headline: accuracy + category breakdown, both runs + delta ---
    acc1 = scorecard1.get("overall_field_accuracy")
    acc2 = scorecard2.get("overall_field_accuracy")
    accuracy_delta = (acc2 - acc1) if (acc1 is not None and acc2 is not None) else None

    category_comparison = {}
    for category in CATEGORIES:
        c1 = scorecard1["field_category_counts"].get(category, 0)
        c2 = scorecard2["field_category_counts"].get(category, 0)
        category_comparison[category] = {"run1": c1, "run2": c2, "delta": c2 - c1}

    # --- receipt-level status changes (e.g. scored -> extraction_failed) ---
    all_receipt_ids = sorted(set(per_doc1) | set(per_doc2))
    status_changes = []
    for receipt_id in all_receipt_ids:
        status1 = per_doc1.get(receipt_id, {}).get("status", "missing")
        status2 = per_doc2.get(receipt_id, {}).get("status", "missing")
        if status1 != status2:
            status_changes.append({"receipt_id": receipt_id, "run1_status": status1, "run2_status": status2})

    # --- field-level regressions / improvements ---
    regressions, improvements = [], []
    for receipt_id in all_receipt_ids:
        doc1 = per_doc1.get(receipt_id, {"status": "missing"})
        doc2 = per_doc2.get(receipt_id, {"status": "missing"})
        r, i = _diff_receipt(receipt_id, doc1, doc2)
        regressions.extend(r)
        improvements.extend(i)

    net_change = len(improvements) - len(regressions)

    diff_output = {
        "run1_id": run1_id,
        "run2_id": run2_id,
        "headline": {
            "overall_field_accuracy": {"run1": acc1, "run2": acc2, "delta": accuracy_delta},
            "category_counts": category_comparison,
        },
        "summary": {
            "total_regressions": len(regressions),
            "total_improvements": len(improvements),
            "net_change": net_change,
        },
        "receipt_status_changes": status_changes,
        "regressions": regressions,
        "improvements": improvements,
    }

    output_path = os.path.join(EVAL_RUNS_DIR, f"{run1_id}_vs_{run2_id}_diff.json")
    os.makedirs(EVAL_RUNS_DIR, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(diff_output, f, indent=2)
        f.write("\n")

    # --- console summary ---
    print(f"Comparing {run1_id} -> {run2_id}\n")

    acc1_str = f"{acc1:.1%}" if acc1 is not None else "n/a"
    acc2_str = f"{acc2:.1%}" if acc2 is not None else "n/a"
    delta_str = f"{accuracy_delta:+.1%}" if accuracy_delta is not None else "n/a"
    print(f"Overall field accuracy: {acc1_str} -> {acc2_str}  ({delta_str})")
    print()
    print("Category breakdown:")
    print(f"  {'category':<18}{'run1':>8}{'run2':>8}{'delta':>8}")
    for category in CATEGORIES:
        c = category_comparison[category]
        print(f"  {category:<18}{c['run1']:>8}{c['run2']:>8}{c['delta']:>+8}")
    print()

    if status_changes:
        print(f"Receipt status changes ({len(status_changes)}):")
        for change in status_changes:
            print(f"  {change['receipt_id']}: {change['run1_status']} -> {change['run2_status']}")
        print()

    print(f"=== Summary: {len(regressions)} regression(s), {len(improvements)} improvement(s), net {net_change:+d} ===\n")

    if regressions:
        print(f"REGRESSIONS ({len(regressions)}) -- correct in {run1_id}, not correct in {run2_id}:")
        for r in regressions:
            print(
                f"  {r['receipt_id']} | {r['field']}: "
                f"{r['run1_category']} ({r['run1_value']!r}) -> {r['run2_category']} ({r['run2_value']!r})"
                f"  [ground truth: {r['ground_truth_value']!r}]"
            )
        print()

    if improvements:
        print(f"IMPROVEMENTS ({len(improvements)}) -- not correct in {run1_id}, correct in {run2_id}:")
        for i in improvements:
            print(
                f"  {i['receipt_id']} | {i['field']}: "
                f"{i['run1_category']} ({i['run1_value']!r}) -> {i['run2_category']} ({i['run2_value']!r})"
                f"  [ground truth: {i['ground_truth_value']!r}]"
            )
        print()

    print(f"Full diff saved to {output_path}")


if __name__ == "__main__":
    main()
