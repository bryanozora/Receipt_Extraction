"""HTTP API service wrapping extract_receipt() (Step 8).

Run from the repo root (package-style imports require it):
    uvicorn src.api.main:app --reload

Endpoints
---------
- GET  /health        -- trivial liveness check.
- POST /extract        -- one receipt: one or more image files (multipart
  "files"), for a receipt photographed as multiple overlapping images.
- POST /extract/batch   -- multiple receipts in one request, processed
  sequentially (see below). Returns a list of results in submission order.

Status codes: both successful extractions and *handled* extraction failures
(ExtractionError -- invalid input, API failure, parse failure) are controlled
outcomes, not server errors, so both return HTTP 200. The response body's
`result` shape (a ReceiptExtraction vs. an ExtractionError) is what tells the
caller which happened, not the status code. HTTP 422 is used only for a
malformed *request* (e.g. /extract/batch's file/group_sizes mismatch -- a
client error, not an extraction outcome). HTTP 500 is reserved for genuine,
unexpected bugs outside extract_receipt()'s own error handling -- FastAPI's
default behavior already does this for us; nothing here needs to catch it.

Batch request shape
--------------------
Multipart form data doesn't have a native way to group repeated file fields
into receipts, so /extract/batch uses two fields:
- `files`: a flat, ordered list of every image across every receipt.
- `group_sizes`: a comma-separated list of ints, e.g. "1,2,1" -- how many
  consecutive files in `files` belong to each receipt, in order. A "1,2,1"
  example means: receipt 1 = files[0], receipt 2 = files[1:3] (a two-image
  receipt), receipt 3 = files[3].

Sequential, not concurrent
----------------------------
/extract/batch processes receipts one at a time in a plain loop, per the
project's tight Gemini API quota -- not because of any FastAPI limitation.
Endpoints are declared as plain `def` (not `async def`) so FastAPI/Starlette
runs them in a worker thread automatically, keeping the blocking Gemini
calls (network I/O plus retry backoff sleeps) off the event loop without
needing any async/await plumbing here.

Cost/latency logging
----------------------
Every request logs wall-clock latency and cumulative Gemini token usage
(prompt/candidates/total, summed across any retries extract_receipt_with_usage()
made internally) via the standard `logging` module, and returns the same
numbers in the response envelope's `metadata` -- kept separate from the
ReceiptExtraction/ExtractionError result itself, since token usage isn't a
receipt field. This is what Step 12's before/after optimization comparison
will read from.
"""

import logging
import os
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.extraction.extract import ExtractionError, ReceiptExtraction, TokenUsage, extract_receipt_with_usage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("receipt_extraction_api")

app = FastAPI(title="Receipt Extraction API")


class ExtractionMetadata(BaseModel):
    latency_seconds: float
    token_usage: TokenUsage


class ExtractionResponseEnvelope(BaseModel):
    result: ReceiptExtraction | ExtractionError
    metadata: ExtractionMetadata


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _save_uploads_to_temp(files: list[UploadFile], tmp_dir: str) -> list[str]:
    """Write uploaded files to a temp dir and return their paths.

    extract_receipt_with_usage() operates on file paths, not in-memory
    bytes, so uploads are written to disk rather than changing that
    function's signature just for this one caller.
    """
    paths = []
    for i, upload in enumerate(files):
        suffix = Path(upload.filename or "").suffix or ".jpg"
        path = os.path.join(tmp_dir, f"upload_{i}{suffix}")
        with open(path, "wb") as f:
            f.write(upload.file.read())
        paths.append(path)
    return paths


def _run_extraction(image_paths: list[str]) -> ExtractionResponseEnvelope:
    start = time.perf_counter()
    result, usage = extract_receipt_with_usage(image_paths)
    latency_seconds = time.perf_counter() - start

    outcome = "success" if isinstance(result, ReceiptExtraction) else result.failure_stage
    logger.info(
        "extraction outcome=%s latency_seconds=%.2f prompt_tokens=%d candidates_tokens=%d "
        "total_tokens=%d",
        outcome,
        latency_seconds,
        usage.prompt_token_count,
        usage.candidates_token_count,
        usage.total_token_count,
    )

    return ExtractionResponseEnvelope(
        result=result,
        metadata=ExtractionMetadata(latency_seconds=latency_seconds, token_usage=usage),
    )


@app.post("/extract", response_model=ExtractionResponseEnvelope)
def extract(files: list[UploadFile] = File(...)) -> ExtractionResponseEnvelope:
    with tempfile.TemporaryDirectory() as tmp_dir:
        image_paths = _save_uploads_to_temp(files, tmp_dir)
        return _run_extraction(image_paths)


@app.post("/extract/batch", response_model=list[ExtractionResponseEnvelope])
def extract_batch(
    files: list[UploadFile] = File(...),
    group_sizes: str = Form(...),
) -> list[ExtractionResponseEnvelope]:
    try:
        sizes = [int(s) for s in group_sizes.split(",") if s.strip()]
    except ValueError:
        raise HTTPException(status_code=422, detail=f"group_sizes must be comma-separated integers, got: {group_sizes!r}")

    if not sizes or any(size < 1 for size in sizes):
        raise HTTPException(status_code=422, detail="group_sizes must be one or more positive integers")

    if sum(sizes) != len(files):
        raise HTTPException(
            status_code=422,
            detail=f"group_sizes sums to {sum(sizes)} but {len(files)} file(s) were uploaded",
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        all_paths = _save_uploads_to_temp(files, tmp_dir)

        responses = []
        offset = 0
        for size in sizes:
            group_paths = all_paths[offset : offset + size]
            offset += size
            responses.append(_run_extraction(group_paths))
        return responses
