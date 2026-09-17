"""Step 9, part 1: checkpointed extraction runner.

Run from the repo root (package-style import of extract.py requires it):
    python scripts/run_eval_extractions.py [run_id]

For every receipt with a ground truth entry in data/ground_truth/, this
calls extract_receipt_with_usage() directly -- no HTTP layer, since this is
an offline batch job, not something serving live requests -- and writes the
result to outputs/eval_runs/<run_id>/<receipt_id>.json immediately after
each call. That means:

- A crash or interruption partway through never loses completed work.
- Re-running the same run_id skips every receipt that already has a saved
  file (checkpoint/resume), so it never re-spends an API call on a receipt
  that already succeeded -- or already failed. A saved ExtractionError is a
  valid, final outcome for this run, not something to retry.

run_id defaults to "default" (outputs/eval_runs/default/), so
re-running with no argument keeps resuming the same run rather than starting
a fresh, empty one each time. Pass an explicit run_id to keep a separate,
comparable eval run (e.g. for comparing prompt versions later) from
overwriting or resuming into this one.

This script only gets extraction results, resumably. Scoring the results
against ground truth is a separate script.
"""

import argparse
import glob
import json
import os
import sys
import time

# Running this file directly (`python scripts/run_eval_extractions.py`) puts
# scripts/ on sys.path, not the repo root -- `import src` fails otherwise
# regardless of the current working directory. Add the repo root explicitly
# so the package-style import below resolves the same way no matter how
# this script is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extraction.extract import ExtractionError, ReceiptExtraction, extract_receipt_with_usage

RAW_DATA_DIR = "Raw Data"
GROUND_TRUTH_DIR = os.path.join("data", "ground_truth")
EVAL_RUNS_DIR = os.path.join("outputs", "eval_runs")


def _find_image_paths(receipt_id: str) -> list[str]:
    """Map a ground-truth receipt id back to its source image(s) in Raw Data/.

    Multi-page receipts (e.g. receipt_026_p1.jpeg, receipt_026_p2.jpeg) are
    globbed and returned in page order. Otherwise falls back to the plain
    single-image path, even if it turns out not to exist -- a genuinely
    missing image then surfaces as an ordinary invalid_input ExtractionError
    from extract_receipt_with_usage() itself (checkpointed and reported like
    any other failure) rather than a separate special case here.
    """
    multi_paths = sorted(glob.glob(os.path.join(RAW_DATA_DIR, f"{receipt_id}_p*.jpeg")))
    if multi_paths:
        return multi_paths
    return [os.path.join(RAW_DATA_DIR, f"{receipt_id}.jpeg")]


def _list_receipt_ids() -> list[str]:
    return sorted(
        os.path.splitext(name)[0] for name in os.listdir(GROUND_TRUTH_DIR) if name.endswith(".json")
    )


def _build_output(result: ReceiptExtraction | ExtractionError, usage, latency_seconds: float) -> dict:
    return {
        "result_type": "ReceiptExtraction" if isinstance(result, ReceiptExtraction) else "ExtractionError",
        "result": json.loads(result.model_dump_json()),
        "latency_seconds": latency_seconds,
        "token_usage": usage.model_dump(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpointed extraction runner for eval.")
    parser.add_argument(
        "run_id",
        nargs="?",
        default="default",
        help=(
            "Eval run identifier (used as a subdirectory name). Defaults to 'default', so "
            "re-running with no argument always resumes the same run instead of starting a "
            "fresh, empty one -- pass an explicit run_id to keep a separate run (e.g. for "
            "comparing prompt versions) from overwriting/resuming into this one."
        ),
    )
    args = parser.parse_args()
    run_id = args.run_id

    run_dir = os.path.join(EVAL_RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    receipt_ids = _list_receipt_ids()
    total = len(receipt_ids)

    succeeded = 0
    failed = 0
    skipped = 0

    print(f"Run ID: {run_id}")
    print(f"Output dir: {run_dir}")
    print(f"Receipts to process: {total}\n")

    for i, receipt_id in enumerate(receipt_ids, start=1):
        output_path = os.path.join(run_dir, f"{receipt_id}.json")

        if os.path.exists(output_path):
            skipped += 1
            print(f"[{i}/{total}] {receipt_id}: skipped (already cached)")
            continue

        image_paths = _find_image_paths(receipt_id)

        start = time.perf_counter()
        result, usage = extract_receipt_with_usage(image_paths)
        latency_seconds = time.perf_counter() - start

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(_build_output(result, usage, latency_seconds), f, indent=2)
            f.write("\n")

        if isinstance(result, ReceiptExtraction):
            succeeded += 1
            print(f"[{i}/{total}] {receipt_id}: success ({latency_seconds:.1f}s)")
        else:
            failed += 1
            print(
                f"[{i}/{total}] {receipt_id}: FAILED ({result.failure_stage}) "
                "- see saved output for details"
            )

    print()
    print("=== Summary ===")
    print(f"Succeeded: {succeeded}")
    print(f"Failed:    {failed}")
    print(f"Skipped (cached): {skipped}")


if __name__ == "__main__":
    main()
