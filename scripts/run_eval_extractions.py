"""Step 9, part 1: checkpointed extraction runner.

Run from the repo root:
    python scripts/run_eval_extractions.py [run_id]

Calls the extraction API's POST /extract endpoint over HTTP for every
receipt with a ground truth entry in data/ground_truth/ -- not
extract_receipt_with_usage() directly. This is deliberate: the eval runner
is meant to exercise the same service a real caller would hit (and, once
Dockerized, runs in its own container talking to the api container over the
network), so it shouldn't have a separate, undocumented path that bypasses
the API layer. The saved file format and checkpointing behavior are
unchanged from when this called the function directly.

The API base URL is configurable via the API_BASE_URL environment
variable, defaulting to http://localhost:8000 for local runs outside
Docker. Inside Docker Compose, this is set to the api service's name
(e.g. http://api:8000) so Compose's internal DNS resolves it.

Results are written to outputs/eval_runs/<run_id>/<receipt_id>.json
immediately after each call. That means:

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

import requests

RAW_DATA_DIR = "Raw Data"
GROUND_TRUTH_DIR = os.path.join("data", "ground_truth")
EVAL_RUNS_DIR = os.path.join("outputs", "eval_runs")

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 180  # generous: a 2-image receipt has taken 70+s in practice

# Placeholder token usage for the (rare) case a request never reached the
# API at all -- keeps the saved file's shape identical to a normal response
# even when there's no real usage to report.
_EMPTY_TOKEN_USAGE = {
    "prompt_token_count": 0,
    "candidates_token_count": 0,
    "thoughts_token_count": 0,
    "cached_content_token_count": 0,
    "tool_use_prompt_token_count": 0,
    "total_token_count": 0,
}


def _find_image_paths(receipt_id: str) -> list[str]:
    """Map a ground-truth receipt id back to its source image(s) in Raw Data/.

    Multi-page receipts (e.g. receipt_026_p1.jpeg, receipt_026_p2.jpeg) are
    globbed and returned in page order. Otherwise falls back to the plain
    single-image path, even if it turns out not to exist -- a genuinely
    missing image then surfaces as an ordinary invalid_input ExtractionError
    from the API itself (checkpointed and reported like any other failure)
    rather than a separate special case here.
    """
    multi_paths = sorted(glob.glob(os.path.join(RAW_DATA_DIR, f"{receipt_id}_p*.jpeg")))
    if multi_paths:
        return multi_paths
    return [os.path.join(RAW_DATA_DIR, f"{receipt_id}.jpeg")]


def _list_receipt_ids() -> list[str]:
    return sorted(
        os.path.splitext(name)[0] for name in os.listdir(GROUND_TRUTH_DIR) if name.endswith(".json")
    )


def _call_extract_api(image_paths: list[str]) -> dict:
    """POST the receipt's image(s) to /extract and return the response envelope.

    A connection failure (API unreachable) or a non-200 response is not
    something to crash the whole run over -- it's recorded as an ordinary
    api_failure ExtractionError for this one receipt, same as any other
    failure category, so the run keeps going and this receipt can be
    retried later once the API is reachable again.
    """
    opened_files = [open(p, "rb") for p in image_paths]
    try:
        files = [("files", (os.path.basename(p), fh, "image/jpeg")) for p, fh in zip(image_paths, opened_files)]
        response = requests.post(f"{API_BASE_URL}/extract", files=files, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.RequestException as e:
        return {
            "result": {
                "image_paths": image_paths,
                "failure_stage": "api_failure",
                "message": f"Could not reach extraction API at {API_BASE_URL}: {e}",
            },
            "metadata": {"latency_seconds": 0.0, "token_usage": _EMPTY_TOKEN_USAGE},
        }
    finally:
        for fh in opened_files:
            fh.close()

    if response.status_code != 200:
        return {
            "result": {
                "image_paths": image_paths,
                "failure_stage": "api_failure",
                "message": f"API returned HTTP {response.status_code}: {response.text[:500]}",
            },
            "metadata": {"latency_seconds": 0.0, "token_usage": _EMPTY_TOKEN_USAGE},
        }

    return response.json()


def _build_output(response_envelope: dict) -> dict:
    result = response_envelope["result"]
    metadata = response_envelope["metadata"]
    result_type = "ExtractionError" if "failure_stage" in result else "ReceiptExtraction"
    return {
        "result_type": result_type,
        "result": result,
        "latency_seconds": metadata["latency_seconds"],
        "token_usage": metadata["token_usage"],
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
    print(f"API base URL: {API_BASE_URL}")
    print(f"Output dir: {run_dir}")
    print(f"Receipts to process: {total}\n")

    for i, receipt_id in enumerate(receipt_ids, start=1):
        output_path = os.path.join(run_dir, f"{receipt_id}.json")

        if os.path.exists(output_path):
            skipped += 1
            print(f"[{i}/{total}] {receipt_id}: skipped (already cached)")
            continue

        image_paths = _find_image_paths(receipt_id)
        response_envelope = _call_extract_api(image_paths)
        output = _build_output(response_envelope)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
            f.write("\n")

        latency_seconds = output["latency_seconds"]
        if output["result_type"] == "ReceiptExtraction":
            succeeded += 1
            print(f"[{i}/{total}] {receipt_id}: success ({latency_seconds:.1f}s)")
        else:
            failed += 1
            print(
                f"[{i}/{total}] {receipt_id}: FAILED ({output['result']['failure_stage']}) "
                "- see saved output for details"
            )

    print()
    print("=== Summary ===")
    print(f"Succeeded: {succeeded}")
    print(f"Failed:    {failed}")
    print(f"Skipped (cached): {skipped}")


if __name__ == "__main__":
    main()
