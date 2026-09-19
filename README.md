# Receipt Data Extraction

A structured extraction system for Indonesian retail receipts, built for the Datasaur AI Engineer take-home test. A Gemini VLM call turns a photographed receipt into a typed, validated, confidence-aware JSON record, wrapped in a FastAPI service, and checked against hand-labeled ground truth with a scorecard-driven eval harness.

For the full, blow-by-blow reasoning behind every decision and bug fix in this document, see [Decision Log - Datasaur Technical Test.md](Descision%20Log%20-%20Datasaur%20Technical%20Test.md) (same content also available as a [Google Doc](https://docs.google.com/document/d/1NYBQ0tVH73NPe-zNJDkwjeDsbBbZxfpOVGI-mO-qG1Y/edit?usp=sharing) if you'd rather not open the markdown file directly). This README is the short version.

## Run commands

Requires Docker and a Google AI Studio API key ([aistudio.google.com](https://aistudio.google.com)).

```bash
cp .env.example .env        # then fill in GOOGLE_API_KEY

docker compose up            # starts the extraction API on :8000, leaves it serving
docker compose run --rm eval # runs the eval harness against that API, writes the scorecard
```

`eval` re-uses whatever is already saved under `outputs/eval_runs/default/` and only calls the API for receipts it hasn't extracted yet (see [Checkpointing](#checkpointing-and-resume) below), so re-running it is cheap. To score a fresh run instead of resuming the committed one:

```bash
RUN_ID=my_run docker compose run --rm eval
```

Without Docker (from the repo root, with the venv active and `.env` present):

```bash
uvicorn src.api.main:app --reload                     # the API
python scripts/run_eval_extractions.py [run_id]        # extract, checkpointed
python scripts/score_eval.py <run_id>                  # score
python scripts/diff_runs.py <run1_id> <run2_id>         # compare two runs
```

To try the running API on any receipt image yourself (a multi-page receipt takes multiple `files` entries in one request):

```python
import requests

path = "path/to/your_receipt.jpeg"
with open(path, "rb") as f:
    response = requests.post("http://localhost:8000/extract", files={"files": f})
print(response.json())
```

Note: Swagger UI (`/docs`) has a known cosmetic bug where the file upload field renders as a plain text/array input instead of a file picker (a FastAPI/Swagger UI version mismatch, not a bug in this code) — use the snippet above or any HTTP client to test the API instead.

`outputs/eval_runs/{default,prompt_v2,prompt_v3}/` are committed as real runs against the full 26-receipt dataset, so the eval results below can be inspected without spending API quota.

## Dataset, schema, and approach

**Dataset:** 26 real receipts, self-photographed (one is a 2-page/2-image long receipt), all IDR currency and DD/MM/YYYY dates, with a mix of Indonesian and English labels. Chosen deliberately for real-world messiness rather than clean scans: skew, low light, folds, wrinkles, a stamp, handwriting, receipts rotated 90°/diagonally. Currency and date format are intentionally kept uniform for this submission (see [Decision Log §2.2](Descision%20Log%20-%20Datasaur%20Technical%20Test.md)) — non-IDR/non-DD-MM-YYYY handling is flagged as future work below, not something the current prompt has been tested against.

**Extraction approach: a single VLM call (Gemini `gemini-3.6-flash`), not OCR+LLM or an agentic pipeline.** Receipts are flat, single-purpose documents — there's no multi-step reasoning or tool use that a single well-designed call with vision doesn't already cover, and a single call is also far easier to keep resilient, debuggable, and explainable in a live-code walkthrough. OCR+LLM was considered and rejected as the primary path (extra pipeline stage, accumulates OCR error before the LLM ever sees it) but is exactly the kind of comparison worth running as a follow-up experiment — see [What I'd do with more time](#what-id-do-with-more-time).

**Schema** ([src/extraction/schema.py](src/extraction/schema.py)): every field is wrapped in a generic `FieldValue[T]` — `{value, confidence, status, reason}` — instead of a bare scalar. `status` (`present` / `not_present` / `illegible`) is what lets the model abstain honestly instead of guessing; `confidence` and `status` are deliberately orthogonal (a field can be present with low confidence). `ReceiptExtraction` holds 7 header fields (vendor, date, receipt number, currency, subtotal, tax, total) plus `line_items: List[LineItem]`, and a `subtotal_tax_total_consistent` flag computed locally (not by the model) via `math.isclose`, as a secondary confidence signal rather than a hard validation rule. Pydantic v2's native generics made this work with zero per-type special-casing.

Gemini's `response_schema` is built directly from `ReceiptExtraction.model_json_schema()`, so the API itself enforces field names, nesting, and enum values — the model cannot return a shape that doesn't match the schema.

## Architecture

```
receipt image(s)
      │
      ▼
Gemini gemini-3.6-flash (structured output, response_schema = ReceiptExtraction)
      │
      ▼
src/extraction/extract.py  →  ReceiptExtraction | ExtractionError
      │
      ▼
FastAPI (src/api/main.py) — POST /extract, POST /extract/batch, GET /health
      │
      ▼
scripts/run_eval_extractions.py  →  outputs/eval_runs/<run_id>/*.json  (checkpointed)
      │
      ▼
scripts/score_eval.py  →  scorecard.json + per_document_results.json
      │
      ▼
scripts/diff_runs.py  →  regression/improvement diff between two runs
```

`docker compose up` starts only the `api` service; `eval` is profile-gated so it never starts automatically, and only runs via `docker compose run --rm eval`, waiting on `api`'s healthcheck.

**Resilience** ([src/extraction/extract.py](src/extraction/extract.py)): `extract_receipt()` returns `ReceiptExtraction | ExtractionError`, never raises, and never silently returns empty-looking-valid data. Three distinct failure categories are handled differently: invalid input files fail fast before any API call; transient API/network failures (5xx, 429, timeouts, and a blocked/empty response) retry with exponential backoff; persistent parse failures retry with the validation error fed back into the prompt. A single `isinstance` check tells a caller a hard failure from a normal (possibly low-confidence) success.

### Checkpointing and resume

`run_eval_extractions.py` writes each result to `outputs/eval_runs/<run_id>/<receipt_id>.json` immediately, and skips any receipt that already has a saved file — success or `ExtractionError` alike. This was not a nice-to-have: the free-tier API quota was hit mid-run multiple times during real development, and this is what let extraction resume cleanly (including across a change of API key) without re-spending quota on already-completed receipts.

## Key trade-offs

- **Single VLM call over OCR+LLM or an agentic pipeline** — simpler, faster to build and debug, and a better fit for flat single-page documents; the cost is no explicit intermediate OCR text to fall back on or audit separately.
- **Sequential batch processing, not concurrent** (`/extract/batch` loops, doesn't fan out) — a deliberate choice given a tight free-tier quota, at the cost of batch latency scaling linearly with receipt count. Flagged as the first thing to revisit for a production load, alongside async/background job handling (see below).
- **HTTP 200 for handled extraction failures, not just successes** — an `ExtractionError` is a controlled, meaningful outcome, not a server bug; only a malformed request (e.g. batch's file/group_sizes mismatch) gets 422, and 500 is reserved for genuine unexpected bugs. This keeps "the extraction failed" and "the request was broken" cleanly separable for a caller.
- **`subtotal_tax_total_consistent` as a soft flag, not a hard rejection rule** — receipts round differently and some are tax-inclusive; forcing exact consistency would reject valid receipts, so it's `math.isclose`-based and just becomes a secondary confidence signal.
- **Ground truth built by reading the physical receipts, never the model's own output** — avoids anchoring bias, at the cost of ground truth labeling being the single most time-consuming step in the whole project.

## Production: cost, latency, and the applied optimization

Every extraction response carries per-call latency and full token accounting (`prompt`, `candidates`, `thoughts`, `cached`, `tool_use`, `total`) in its metadata, so cost/latency are visible per request, not just in aggregate.

**Optimization applied:** Gemini's `thinking_config`. The SDK exposes two mechanisms — `thinking_budget` (explicit token cap, 0 disables) and `thinking_level` (coarser `MINIMAL`/`LOW`/`MEDIUM`/`HIGH` enum). A small, quota-conscious comparison (`scripts/test_thinking_optimization.py`) across 3 configs on 2 receipts (one simple, one complex/multi-image) found `thinking_level=LOW` was the clear winner:

| receipt | config | latency | total tokens |
|---|---|---|---|
| 001 (simple) | baseline | 19.9s | 5022 |
| 001 (simple) | thinking_level=LOW | 8.2s | 3977 |
| 026 (complex, 2 images) | baseline | 36.1s | 13088 |
| 026 (complex, 2 images) | thinking_level=LOW | 21.9s | 6896 |

**Latency down 39–58%, tokens down 21–47%, with no accuracy difference observed on this sample** (vendor name, total, and line-item count all matched baseline). `thinking_level=LOW` is now the default for every caller (`extract_receipt()`, the API, the eval harness). Caveat, stated plainly: this was validated on 2 receipts, not the full 26-receipt dataset — see [What I'd do with more time](#what-id-do-with-more-time).

## Eval results

Ground truth for all 26 receipts lives in `data/ground_truth/`, hand-labeled from the physical receipts. `scripts/score_eval.py` classifies every field into one of 5 categories (`correct`, `wrong_value`, `hallucinated`, `wrongly_abstained`, `missed_status`) and computes per-field precision/recall, with line items aligned across runs by description similarity + amount, not by row position.

Building ground truth this way — reading each of the 26 receipts directly, never the model's own output, to avoid anchoring bias — was the single most time-consuming part of the project. It also didn't stop at the initial labeling pass: several scorecard error patterns were cross-checked back against the physical receipts field by field (see the `receipt_002`, `receipt_011`, `receipt_013`, `receipt_021` findings in the [Decision Log](Descision%20Log%20-%20Datasaur%20Technical%20Test.md)), which is what caught a handful of ground truth mistakes rather than misattributing them to the model.

**Accuracy progression across three full 26-receipt runs**, each transition driven by specific, scorecard-evidenced prompt fixes (not guesses):

| run | overall field accuracy | correct / total | hallucinated | wrongly_abstained |
|---|---|---|---|---|
| `default` | 88.3% | 627/710 | 3 | 36 |
| `prompt_v2` | 94.2% | 669/710 | 0 | 3 |
| `prompt_v3` | 97.6% | 693/710 | 0 | 3 |

Hallucination is near-zero throughout (3/710 at worst, 0 after `prompt_v2`) — the system's core anti-hallucination goal held up under real evaluation, not just spot-checks.

Worth being explicit about: all 26 receipts were used both to iteratively tune the prompt across these three versions and to report the final 97.6% figure — there is no held-out set that stayed unseen during tuning. The number is real (it's what the current prompt scores on this dataset), but it isn't a clean measure of generalization to receipts the prompt was never shaped against.

**A regression was caught and fixed, not just improvements.** `prompt_v2` introduced a real bug: a new "derive `unit_price` from `amount ÷ quantity`" rule silently interacted with an earlier discount-consolidation rule, so on every discounted line item `unit_price` came back as the *discounted* per-unit price instead of the original. The scorecard's field-level breakdown caught this as 25 new `unit_price` mismatches, all on the same 5 discount-affected receipts — not spread randomly, which is what made the cause traceable. `prompt_v3` fixed the derivation to always use the pre-discount price. Re-verified with `scripts/diff_runs.py`: the discount fix itself held up cleanly — `receipt_016` and `receipt_018` are fully correct with no remaining errors — but the full `default` vs `prompt_v3` diff also shows two separate, unrelated regressions on two of the five affected receipts: `receipt_012` (`subtotal`, `tax`, `transaction_date` all wrong) and `receipt_025` (`receipt_number` now wrongly abstained). Neither is caused by the discount fix — `receipt_012` appears to simply be a hard-to-read receipt with several unrelated misreads, and `receipt_025`'s text is noted as fading — so these remain open, unresolved issues, not something to claim as zero regressions.

**Honest accounting of where the +9.3% came from:** re-running `diff_runs.py default prompt_v3` shows 73 field-level improvements, but at least 8 of those have an *identical extracted value* between `default` and `prompt_v3` — only the scoring classification changed, because ground truth or the scorer's own symbol-normalization was corrected in between (e.g. a ground truth typo, or `score_eval.py` learning to strip cosmetic punctuation like `"` vs `*` before comparing descriptions). The remaining ~65 improvements are genuine extraction changes from the `unit_price` and SKU-stripping prompt fixes. **The accuracy gain is a mix of real prompt improvements and ground-truth/scorer corrections — not attributable entirely to the model getting better**, and it would have been easy to overstate this without the diff tool making the distinction visible.

Full scorecards: `outputs/eval_runs/{default,prompt_v2,prompt_v3}/scorecard.json`, per-document detail in `per_document_results.json` alongside each, and the full diff in `outputs/eval_runs/default_vs_prompt_v3_diff.json`.

## Agentic development

Claude Code was used throughout, in scoped, iterative steps that mirrored the execution plan in the decision log (schema → extraction function → resilience → prompt fixes → ground truth → API → eval harness → Docker → optimization → this diff tool), rather than one large "build the whole thing" prompt. Each step was reviewed against real output before moving to the next — several claims Claude Code made along the way turned out to be inaccurate and were caught, not trusted blindly:

- An import-path bug (`ModuleNotFoundError: No module named 'schema'`, when `extract.py` was imported as a package member — e.g. by `src/api/main.py` during Step 8 — rather than run directly as a script) was initially "verified" with a method that didn't actually reproduce the failure mode — caught when the real run failed anyway, then re-verified correctly and fixed with a dual-mode import in `extract.py`.
- A token-accounting gap (`total_token_count` not matching the sum of tracked fields) was investigated live rather than assumed away, and traced to a missing `thoughts_token_count` field.
- Docker Compose's `$$RUN_ID` escaping was ambiguous from `docker compose config` output alone — rather than assuming it was fine, the actual container was built and run to confirm the variable resolved correctly at runtime.
- A claim that `receipt_001`–`receipt_006`'s ground truth templates were still blank turned out to be based on synthetic test data, not the real files — disproven by checking the actual JSON files in `data/ground_truth/`, which were fully labeled.
- A claimed description change on `receipt_018` between two extraction runs couldn't be located in either saved JSON file on manual search — an inaccurate detail in that report, not a real difference.
- An earlier draft of this README's Eval results section claimed the discount-attribution fix produced "0 regressions on the 5 affected receipts" — contradicted by the actual `default_vs_prompt_v3_diff.json` output, which shows two unrelated regressions on two of those five receipts (see above). Caught and corrected before this README was finalized.

This wasn't limited to these six examples — every non-trivial claim Claude Code made was checked against real saved data (JSON output, git history, or the decision log) before being trusted, rather than accepted at face value.

**Custom extension actually used: [scripts/diff_runs.py](scripts/diff_runs.py).** During Step 9's manual scorecard review, comparing two runs field-by-field (which regressed, which improved, keyed consistently across runs) was being done by hand, repeatedly, by eyeballing two `scorecard.json`/`per_document_results.json` files side by side. `diff_runs.py` automates exactly that comparison: it flags every field that was `correct` in run1 and isn't in run2 (regression) or vice versa (improvement), matches line items across runs by **ground-truth row index** rather than extracted-row index (since which extracted row aligns to which ground-truth row can shift between runs, but ground truth itself doesn't), and separately flags whole-receipt status changes (e.g. `scored` → `extraction_failed`). It deliberately does *not* try to diff hallucinated/unmatched-extracted rows field-by-field, since there's no principled way to say one run's invented row is "the same" as another's.

It was run for real against `default` vs `prompt_v3` (see Eval results above) and is what surfaced the "8 of 73 improvements are actually ground-truth/scorer corrections, not model improvements" finding — a distinction that would have been easy to miss and overstate by hand.

## What I'd do with more time

The binding constraint on this list wasn't ideas — it was the combination of the take-home's tight timeframe and Google AI Studio's free-tier daily rate limit, which was hit repeatedly throughout development and required switching across multiple API keys/accounts to keep extraction runs moving (see [Checkpointing and resume](#checkpointing-and-resume) above). Everything below was deferred for that reason, not attempted and abandoned.

- **Score against a genuinely held-out batch of receipts** never used during prompt tuning — the real test of generalization, which the current 97.6% (measured on the same 26 receipts the prompt was tuned against) doesn't provide.
- **Run an OCR+LLM comparison experiment** against the current single-VLM-call approach, on the same 26 receipts — cost, latency, and accuracy side by side, as originally scoped for the Production section but not completed given time.
- **Validate `thinking_level=LOW` against the full 26-receipt dataset**, not just the 2-receipt sample it was chosen from — the accuracy-neutrality claim is currently under-validated.
- **Test non-IDR currencies and non-DD-MM-YYYY date formats** — the current dataset and prompt are IDR/DD-MM-YYYY only by design (see Decision Log §2.2); the schema already carries a `currency` field for this, but the normalization logic hasn't been stress-tested against it.
- **Move to async/background job handling** for production timeout safety — the API currently blocks synchronously per request (in a threadpool), which is fine for this eval harness's scale but not for a production caller that can't tolerate a 20–50s HTTP request.
- **Add retry count to the response metadata** — noted during Step 8 as a fairness concern for latency comparisons (a call that silently retried 3 times looks the same in latency as one that didn't, from the metadata alone) and never circled back to.
- **Validate confidence calibration** — whether the model's self-reported `confidence` actually correlates with correctness (e.g. do low-confidence fields fail more often than high-confidence ones) hasn't been checked against the scorecard data, despite that data already existing to check it.
