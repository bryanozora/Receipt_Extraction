"""Step 12: quota-conscious comparison of Gemini's thinking-config mechanisms.

Run from the repo root:
    python scripts/test_thinking_optimization.py

Calls extract_receipt_with_usage() directly (not through the API/eval
harness -- this is a small, deliberate 6-call comparison, not a batch job)
against two receipts, each under three thinking configurations:

- baseline (unconstrained): thinking_config=None, the current default.
- thinking_budget=0: attempts to disable thinking entirely via an explicit
  token budget.
- thinking_level=LOW: the newer Gemini-3-style coarse control.

Receipts: receipt_001 (simple, single image) and receipt_026 (complex,
multi-image, 20+ line items) -- chosen to see whether the effect (if any)
differs between a cheap call and an expensive one.

extract.py's THINKING_CONFIG is a module-level constant baked into its
_GENERATE_CONFIG at import time, meant to be toggled by hand between whole
runs (matching how SYSTEM_INSTRUCTION has been versioned so far) -- not a
per-call parameter. Rather than duplicating extract.py's config
construction here (system_instruction, response_schema, temperature) and
risking drift, this script swaps extract.py's already-built _GENERATE_CONFIG
in place via Pydantic's model_copy(), changing only thinking_config and
leaving everything else exactly as extract.py defines it. This works because
_generate_with_retry() reads the module-global _GENERATE_CONFIG fresh on
every call, not a value captured at import time.

Retries on transient errors still use extract.py's normal, already-limited
retry policy (no extra retrying added here). If a call still ends in an
ExtractionError, it's saved as-is and the script moves on to the next
config/receipt -- quota is limited, so this never retries a whole config
or aborts the comparison over one failure.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.genai import types

from src.extraction import extract as extract_module
from src.extraction.extract import ExtractionError, ReceiptExtraction, extract_receipt_with_usage

OUTPUT_PATH = os.path.join("outputs", "thinking_optimization_comparison.json")

RECEIPTS = {
    "receipt_001": [os.path.join("Raw Data", "receipt_001.jpeg")],
    "receipt_026": [
        os.path.join("Raw Data", "receipt_026_p1.jpeg"),
        os.path.join("Raw Data", "receipt_026_p2.jpeg"),
    ],
}

CONFIGS = {
    "baseline (unconstrained)": None,
    "thinking_budget=0": types.ThinkingConfig(thinking_budget=0),
    "thinking_level=LOW": types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
}


def _apply_thinking_config(thinking_config: types.ThinkingConfig | None) -> None:
    """Swap extract.py's _GENERATE_CONFIG to a different thinking_config,
    preserving every other setting (prompt, schema, temperature) exactly as
    extract.py currently defines them."""
    extract_module._GENERATE_CONFIG = extract_module._GENERATE_CONFIG.model_copy(
        update={"thinking_config": thinking_config}
    )


def _run_one(receipt_id: str, image_paths: list[str], config_name: str) -> dict:
    print(f"--- {receipt_id} | {config_name} ---")

    start = time.perf_counter()
    result, usage = extract_receipt_with_usage(image_paths)
    latency_seconds = time.perf_counter() - start

    result_type = "ReceiptExtraction" if isinstance(result, ReceiptExtraction) else "ExtractionError"
    result_json = json.loads(result.model_dump_json())

    entry = {
        "receipt_id": receipt_id,
        "config": config_name,
        "latency_seconds": latency_seconds,
        "token_usage": usage.model_dump(),
        "result_type": result_type,
        "result": result_json,
    }

    if result_type == "ExtractionError":
        print(f"  FAILED ({result.failure_stage}): {result.message}")
    else:
        print(
            f"  vendor={result.vendor_name.value!r}  total={result.total.value}  "
            f"line_items={len(result.line_items)}"
        )
    print(f"  latency={latency_seconds:.1f}s  tokens={usage.model_dump()}")
    print(f"  full result:\n{result.model_dump_json(indent=2)}\n")

    return entry


def _print_summary_table(runs: list[dict]) -> None:
    print("=== Summary: latency and tokens by receipt x config ===")
    header = f"{'receipt_id':<14}{'config':<26}{'outcome':<10}{'latency_s':>10}{'total_tokens':>14}"
    print(header)
    print("-" * len(header))
    for entry in runs:
        outcome = "OK" if entry["result_type"] == "ReceiptExtraction" else "FAILED"
        total_tokens = entry["token_usage"].get("total_token_count", 0)
        print(
            f"{entry['receipt_id']:<14}{entry['config']:<26}{outcome:<10}"
            f"{entry['latency_seconds']:>10.1f}{total_tokens:>14}"
        )


def main() -> None:
    original_generate_config = extract_module._GENERATE_CONFIG
    runs = []

    try:
        for receipt_id, image_paths in RECEIPTS.items():
            for config_name, thinking_config in CONFIGS.items():
                _apply_thinking_config(thinking_config)
                runs.append(_run_one(receipt_id, image_paths, config_name))
    finally:
        extract_module._GENERATE_CONFIG = original_generate_config

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)
        f.write("\n")
    print(f"Saved full comparison to {OUTPUT_PATH}\n")

    _print_summary_table(runs)


if __name__ == "__main__":
    main()
