"""Core extraction: receipt image -> ReceiptExtraction.

Structured output strategy
---------------------------
Gemini's `response_schema` config option is built directly from
`ReceiptExtraction.model_json_schema()` (Pydantic's own JSON Schema
generator), so the API itself enforces the field names, nesting, and enum
values for every {value, confidence, status, reason} field -- the model
cannot return a shape that doesn't match the schema. This worked with zero
special-casing for the generic `FieldValue[T]`: each instantiation
(FieldValue[str], FieldValue[float], FieldValue[date]) resolves to its own
correctly-typed inline object with no manual schema-writing needed.

One field is stripped out of the schema sent to the model:
`subtotal_tax_total_consistent`. That field is computed locally by
ReceiptExtraction's own validator from whatever subtotal/tax/total come
back -- asking the model to also produce it would be redundant (and get
silently overwritten by the validator on parse regardless), so it's
removed from the request schema before use.

Two bits of friction worth flagging, neither caused by the generic
FieldValue[T] structure itself:
- This version of the google-genai SDK (2.23) has no `response.parsed`
  convenience -- there's no automatic reconstruction into the Pydantic
  class from a structured-output response, so parsing is done manually
  via `ReceiptExtraction.model_validate_json(response.text)`.
- `response_schema` only takes JSON-Schema-shaped input; Pydantic's own
  output uses `$defs`/`$ref` for reused submodels, which the SDK resolves
  internally, but it's a plain dict handed to the API, not a live schema
  object -- so the "strip one field" step above happens as dict surgery
  before the request goes out, not as a Pydantic model change.

The prompt is split into a system instruction (stable extraction rules:
schema shape, anti-hallucination, normalization) and a short per-call user
message, rather than folding everything into one block of text.

Resilience: low-confidence success vs. hard failure
-----------------------------------------------------
These are two different things and this module keeps them distinct:

- A **low-confidence success** is a valid `ReceiptExtraction` where some
  fields have low confidence or `status="illegible"`/`"not_present"`. This
  is normal, expected output, not an error -- it's the model doing its job
  honestly on a messy document.
- A **hard failure** means extraction genuinely did not complete: the input
  file couldn't be read, the API call kept failing, or the response never
  parsed into valid JSON even after retries. `extract_receipt` returns an
  `ExtractionError` for these instead of a `ReceiptExtraction`, and never
  raises an uncaught exception or silently returns empty-looking-but-valid
  data -- a caller can always tell the two cases apart with a single
  `isinstance` check.

Three failure categories are handled, each with its own retry policy:

1. Invalid input files -- checked before any API call is made (fail fast,
   don't spend a request on a corrupt file).
2. Transient API/network failures (5xx, rate limiting, connection
   timeouts) -- retried with exponential backoff *per call*. A response
   that comes back with no text at all (e.g. blocked by a safety filter)
   falls in this category too, since it's the API declining to produce
   output rather than a shape/parsing problem -- it's reported as an
   `api_failure` immediately rather than crashing on `None`.
3. Persistent parse/validation failures -- retried by feeding the parse
   error back into the prompt, *across calls* (up to two retries beyond
   the first attempt).
"""

import json
import mimetypes
import os
import time
from typing import Literal

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ValidationError

from schema import ReceiptExtraction

load_dotenv(override=True)

API_KEY = os.environ["GOOGLE_API_KEY"]
MODEL_NAME = "gemini-3.6-flash"

SYSTEM_INSTRUCTION = """You are a receipt data extraction system. You will be shown a photograph \
of a single physical retail/restaurant receipt. Extract its data into the exact JSON schema you \
have been given. Follow these rules exactly.

## Output shape
Every extracted field in the schema is an object with four keys: "value", "confidence", \
"status", and "reason" -- never a bare value. This applies to every top-level field and to every \
field inside every line item.

- "value": the extracted value in the correct type (string, number, or ISO date), or null.
- "status": one of "present", "not_present", "illegible".
  - "present": the field exists on the receipt and you could read it.
  - "not_present": the field simply does not exist on this receipt (e.g. no receipt/invoice \
number is printed anywhere on it).
  - "illegible": the field is on the receipt but you cannot read it confidently (blurred, \
cropped, covered, faded, cut off).
- "confidence": your honest 0.0-1.0 estimate that "value" is correct. Do not default to 1.0 -- \
reserve values near 1.0 for text that is printed clearly and unambiguously. Use lower values \
(e.g. 0.3-0.6) when you are reading a smudged digit, inferring from weak context, or unsure \
about a character.
- "reason": a short, one-sentence explanation for why confidence is low or status is not \
"present". Leave this null when status is "present" and confidence is high -- don't explain the \
obvious.

## Absolute rule: do not guess
If a value is not clearly visible, do not invent a plausible-looking value. Set status to \
"illegible" or "not_present" and value to null instead. A wrong value is worse than an honest \
"I don't know" -- this system is judged on never hallucinating data that isn't really on the \
receipt. This includes subtotal/tax/total: if one of these three is genuinely not printed on \
the receipt, mark it "not_present" rather than computing it yourself from the other two.

## Normalization
- Currency amounts: strip currency symbols ("Rp", "IDR", etc.) and separators, and return a \
plain number. Indonesian Rupiah amounts do not use fractional/decimal subunits in practice, so \
treat any "." or "," inside an amount as a thousands separator, never a decimal point -- e.g. \
"Rp 45.000", "45,000", and "45.000,00" must all normalize to 45000.0, not 45.0 or misread as \
having cents.
- Dates: receipts use DD/MM/YYYY or DD-MM-YYYY format. Convert to ISO YYYY-MM-DD. Do not swap \
day and month.
- "currency": the currency code, e.g. "IDR". Infer it from context (symbols, language) if it \
isn't printed explicitly, and lower your confidence accordingly.

## Line items
Extract one line_items entry per distinct row of the itemized purchase table, in the order they \
appear on the receipt. If a row states an amount but not an explicit quantity or unit price, \
mark quantity/unit_price as "not_present" rather than assuming a quantity of 1. If a receipt has \
no itemized table at all, return an empty line_items list rather than inventing a single generic \
row.

## Multi-image receipts
Sometimes you will be given more than one image in a single request. When that happens, the \
images are sequential, overlapping sections of one single physical receipt (it didn't fit in one \
photo), given in top-to-bottom order -- they are not separate receipts. Read them as one \
continuous document and extract a single, unified result. Because the images overlap, the same \
line item or the same header/footer text may appear in more than one image; do not double-count \
a line item that appears in the overlap between two consecutive images.

Receipts may be labeled in Indonesian, English, or a mix of both -- read the labels in whichever \
language the receipt actually uses.
"""

USER_MESSAGE = "Extract the structured data from this receipt image."


def _build_response_schema() -> dict:
    """Pydantic's JSON Schema for ReceiptExtraction, minus the locally-computed field."""
    schema = ReceiptExtraction.model_json_schema()
    schema["properties"].pop("subtotal_tax_total_consistent", None)
    schema["required"] = [
        name for name in schema.get("required", []) if name != "subtotal_tax_total_consistent"
    ]
    return schema


_RESPONSE_SCHEMA = _build_response_schema()

_GENERATE_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    response_mime_type="application/json",
    response_schema=_RESPONSE_SCHEMA,
    # Low temperature: this is an extraction task, not a creative one --
    # faithfulness to the document matters more than varied phrasing.
    temperature=0.1,
)

# Exponential backoff delays (seconds) applied before each retry of a single
# Gemini call when it fails with a transient error. 3 retries beyond the
# initial attempt, so up to 4 calls total for one generate_content invocation.
_API_RETRY_DELAYS_SECONDS = (1, 2, 4)

# How many times to re-prompt (feeding the parse error back) if the response
# doesn't validate against ReceiptExtraction. 2 retries beyond the initial
# attempt, so up to 3 calls total across the whole extract_receipt() run.
_PARSE_RETRY_COUNT = 2


class ExtractionError(BaseModel):
    """A hard failure: extraction did not complete for the given image(s).

    Distinct from a low-confidence ReceiptExtraction, which is a normal,
    valid result -- this is only returned when extraction genuinely could
    not produce one.
    """

    image_paths: list[str]
    failure_stage: Literal["invalid_input", "api_failure", "parse_failure"]
    message: str


def _is_transient(error: Exception) -> bool:
    """Whether an error is worth retrying: server errors, rate limiting, or a dropped connection."""
    if isinstance(error, errors.ServerError):
        return True
    if isinstance(error, errors.ClientError) and error.code == 429:
        return True
    if isinstance(error, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    return False


def _generate_with_retry(
    client: genai.Client, contents: list
) -> types.GenerateContentResponse:
    """Call Gemini once, retrying with exponential backoff on transient failures.

    Non-transient errors (e.g. a 400 bad request) are raised immediately --
    retrying those wouldn't help. If every attempt at a transient error is
    exhausted, the last error is raised for the caller to turn into a hard
    failure instead of letting it crash the whole process.
    """
    last_error: Exception = RuntimeError("unreachable")
    for delay in (0,) + _API_RETRY_DELAYS_SECONDS:
        if delay:
            time.sleep(delay)
        try:
            return client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=_GENERATE_CONFIG,
            )
        except (errors.APIError, httpx.TimeoutException, httpx.ConnectError) as e:
            last_error = e
            if not _is_transient(e):
                raise
    raise last_error


def _validate_input_files(image_paths: list[str]) -> str | None:
    """Return an error message if a file is missing or not a valid image, else None."""
    for path in image_paths:
        if not os.path.isfile(path):
            return f"File not found: {path}"
        try:
            with Image.open(path) as img:
                img.verify()
        except (UnidentifiedImageError, OSError) as e:
            return f"File is not a valid, readable image ({path}): {e}"
    return None


def _describe_empty_response(response: types.GenerateContentResponse) -> str:
    """Best-effort explanation for why a response has no text, e.g. a safety block."""
    if response.prompt_feedback and response.prompt_feedback.block_reason:
        return f"prompt blocked: {response.prompt_feedback.block_reason}"
    if response.candidates:
        finish_reason = response.candidates[0].finish_reason
        if finish_reason:
            return f"finish_reason={finish_reason}"
    return "empty response with no available block/finish reason"


def _load_image_parts(image_paths: list[str]) -> list[types.Part]:
    parts = []
    for path in image_paths:
        mime_type = mimetypes.guess_type(path)[0] or "image/jpeg"
        with open(path, "rb") as f:
            image_bytes = f.read()
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
    return parts


def extract_receipt(image_paths: str | list[str]) -> ReceiptExtraction | ExtractionError:
    """Extract structured fields from one receipt.

    `image_paths` is either a single image path, or a list of image paths
    for a receipt that had to be photographed as multiple overlapping
    images (too long for one frame). A list is sent as multiple Parts in
    one API call, in the given order, so the model reasons across them as
    one document rather than as separate receipts.

    Returns a `ReceiptExtraction` on success (which may still contain
    low-confidence or abstained fields -- that's not an error), or an
    `ExtractionError` if extraction genuinely didn't complete: the input
    file(s) were invalid, the API call kept failing, or the response never
    parsed into valid JSON even after retries.
    """
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    invalid_reason = _validate_input_files(image_paths)
    if invalid_reason is not None:
        return ExtractionError(
            image_paths=image_paths,
            failure_stage="invalid_input",
            message=invalid_reason,
        )

    client = genai.Client(api_key=API_KEY)
    image_parts = _load_image_parts(image_paths)

    message = USER_MESSAGE
    last_parse_error: Exception | None = None

    for _ in range(_PARSE_RETRY_COUNT + 1):
        try:
            response = _generate_with_retry(client, [*image_parts, message])
        except (errors.APIError, httpx.TimeoutException, httpx.ConnectError) as api_error:
            return ExtractionError(
                image_paths=image_paths,
                failure_stage="api_failure",
                message=f"Gemini API call failed: {api_error}",
            )

        if not response.text:
            # The API declined to produce output (e.g. a safety filter blocked the
            # prompt or response) rather than raising -- this is the API's call, not
            # a parsing problem, so it's an api_failure rather than a parse_failure.
            return ExtractionError(
                image_paths=image_paths,
                failure_stage="api_failure",
                message=f"Gemini returned no output ({_describe_empty_response(response)})",
            )

        try:
            return ReceiptExtraction.model_validate_json(response.text)
        except (ValidationError, json.JSONDecodeError) as parse_error:
            last_parse_error = parse_error
            message = (
                f"{USER_MESSAGE}\n\n"
                "Your previous response failed to parse against the required schema with this "
                f"error:\n{parse_error}\n\n"
                "Correct the output and return JSON that matches the schema exactly."
            )

    return ExtractionError(
        image_paths=image_paths,
        failure_stage="parse_failure",
        message=(
            f"Response did not parse into ReceiptExtraction after {_PARSE_RETRY_COUNT + 1} "
            f"attempts: {last_parse_error}"
        ),
    )


if __name__ == "__main__":
    receipt_numbers = ["014", "016", "018", "019", "025"]
    output_dir = os.path.join("outputs", "sample_runs")
    os.makedirs(output_dir, exist_ok=True)

    for number in receipt_numbers:
        image_path = os.path.join("Raw Data", f"receipt_{number}.jpeg")
        print(f"\n=== receipt_{number} ===")
        result = extract_receipt(image_path)

        if isinstance(result, ExtractionError):
            print(f"Extraction failed [{result.failure_stage}]: {result.message}")
            continue

        output_json = result.model_dump_json(indent=2)
        print(output_json)

        output_path = os.path.join(output_dir, f"receipt_{number}_output.json")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_json)
