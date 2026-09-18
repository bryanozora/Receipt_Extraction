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

Prompt refinements from Step 6 sample testing
------------------------------------------------
Five rules were added to the system instruction after real failures surfaced
during Step 6 sample testing (not hypothetical edge cases), kept here for
traceability: careful digit-by-digit year reading (a rotated receipt was
misread as 2024 instead of 2026), including occluded-but-visibly-present
line items as illegible rather than silently dropping the row, treating a
percentage discount line as modifying the preceding item's amount rather
than becoming its own line item, preferring printed vendor text over a
brand logo, and resolving subtotal ambiguity when a receipt prints both a
pre-tax "DPP" and a separately labeled "Subtotal" by checking which value
is arithmetically consistent with tax + total.

Prompt refinements from the first eval scorecard (Step 9)
-------------------------------------------------------------
Two more systemic patterns, found from the first real scorecard run rather
than spot-checking samples: unit_price was being marked not_present on
every row where it wasn't printed explicitly, even when it was trivially
derivable as amount / quantity -- now computed and reported as a present,
lower-confidence value instead of an unnecessary abstention. Item
descriptions were including a leading numeric SKU/product code straight
off the receipt (e.g. "36512328 TANGO WFR CHO 100G") instead of just the
human-readable name -- now stripped.

Three more fixes from a full scorecard review, on top of those two: queue/order/table numbers
were sometimes being extracted as receipt_number when no real receipt/invoice number was
printed -- now only a field actually labeled or functioning as a receipt/invoice/transaction
number qualifies, otherwise receipt_number is not_present. Receipt numbers were sometimes
truncated to a fragment of a longer printed sequence -- now the full printed form is required.
A discount line was sometimes forced onto the immediately preceding item even when the
arithmetic didn't plausibly fit (e.g. the discount exceeded that item's price) -- now that
forced attribution is disallowed, in favor of best judgment, lower confidence, and a noted
ambiguity in reason.

One more fix, from the prompt_v2 scorecard: the unit_price-derivation rule and the
discount-consolidation rule were interacting badly -- deriving unit_price as amount / quantity
silently produced a *discounted* unit price whenever amount had already been reduced by a
discount on that row. unit_price must always be the item's original, pre-discount per-unit
price; amount is the only field a discount should ever change. The derivation rule now divides
the original pre-discount price (the same one the discount was calculated from) by quantity,
and falls back to not_present if that original price can't be recovered, rather than deriving an
incorrect discounted value.

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

try:
    from schema import ReceiptExtraction  # run directly as a script (python src/extraction/extract.py)
except ImportError:
    from src.extraction.schema import ReceiptExtraction  # imported as a package (e.g. by src/api/main.py)

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

## Vendor name
When a receipt shows both an explicit printed store name/label (e.g. a "STORE:" line, a header, \
or an address block naming the store) and a separate brand/franchise logo that might differ from \
it, prefer the printed text for vendor_name. Only infer the vendor from a logo when there is no \
legible printed store name anywhere on the receipt, and note the lower confidence in "reason" \
when you do.

## Receipt number
Some receipts print a separate queue number, order number, or table number (e.g. for pickup or \
serving order) alongside or instead of a formal receipt/invoice/transaction number. Only extract \
receipt_number from something that is actually labeled as, or clearly functions as, a \
receipt/invoice/transaction number -- not a queue number, order number, or table number. If only \
a queue/order number is visible and no actual receipt number is printed anywhere on the receipt, \
mark receipt_number as "not_present" rather than using the queue/order number as a stand-in.

When a receipt prints several number sequences near each other and one of them is identified as \
the receipt number, extract its complete printed form -- not a partial substring of it.

## Subtotal ambiguity (e.g. "DPP" vs. a separately printed "Subtotal")
Some receipts print more than one value that could plausibly be "subtotal" -- for example both \
"DPP" (the pre-tax base amount) and a separately labeled "Subtotal" line, which are not always \
the same thing (a printed "Subtotal" sometimes already includes tax). This schema's subtotal \
field means the pre-tax amount. When more than one candidate value is present, prefer whichever \
one, added to tax, comes closest to matching total -- that is the mathematically consistent \
pre-tax figure. If "DPP" is present and satisfies this check, prefer it by name. If the check is \
ambiguous or no candidate fits well, use your best judgment, lower your confidence on subtotal, \
and note the ambiguity in reason.

## Normalization
- Currency amounts: strip currency symbols ("Rp", "IDR", etc.) and separators, and return a \
plain number. Indonesian Rupiah amounts do not use fractional/decimal subunits in practice, so \
treat any "." or "," inside an amount as a thousands separator, never a decimal point -- e.g. \
"Rp 45.000", "45,000", and "45.000,00" must all normalize to 45000.0, not 45.0 or misread as \
having cents.
- Dates: receipts use DD/MM/YYYY or DD-MM-YYYY format. Convert to ISO YYYY-MM-DD. Do not swap \
day and month. Read all four digits of the year carefully rather than assuming or guessing the \
decade/century -- a rotated, skewed, or low-quality image increases the risk of misreading a \
single digit, so check the year digit-by-digit rather than pattern-matching to a "typical" year.
- "currency": the currency code, e.g. "IDR". Infer it from context (symbols, language) if it \
isn't printed explicitly, and lower your confidence accordingly.

## Line items
Extract one line_items entry per distinct row of the itemized purchase table, in the order they \
appear on the receipt. If a row states an amount but not an explicit quantity, mark quantity as \
"not_present" rather than assuming a quantity of 1 -- do not guess it. If a receipt has no \
itemized table at all, return an empty line_items list rather than inventing a single generic \
row.

Item descriptions sometimes have a leading numeric SKU/product code before the human-readable \
name (e.g. "36512328 TANGO WFR CHO 100G"). Exclude the leading SKU/code from description -- keep \
only the human-readable item name (e.g. "TANGO WFR CHO 100G").

Unit price is often not printed explicitly, but it can be a legitimate derived fact rather than \
a guess: if quantity is known and the item's original, pre-discount per-unit price can be \
determined, compute unit_price = (original pre-discount price) / quantity, set status="present", \
note in reason that the value was computed rather than read directly, and use a somewhat lower \
confidence than you would for a value read directly off the receipt. unit_price must always \
represent the item's original, pre-discount per-unit price -- never a discounted figure. If a \
discount applies to this row (see below), do NOT compute unit_price from the discounted amount; \
divide the same original, pre-discount price the discount itself was calculated from by \
quantity instead. Only mark unit_price as "not_present" when the original per-unit price \
genuinely cannot be determined -- e.g. quantity itself is also unknown or not_present, or only a \
lump discounted total is printed with no way to recover the pre-discount figure.

If part of a line item is obscured (by handwriting, a stamp, a fold, or similar) but there is \
visible evidence the row exists (e.g. a partial line, a stray price, or a gap in the item \
sequence), still include it in line_items: fill in whichever fields you can read, and set \
status="illegible" on the ones you cannot. Never silently omit a row that is visibly present \
just because part of it is unreadable.

If a receipt shows a discount as a percentage or amount on its own line directly below an item \
(e.g. "Disc 5%") and no separate final discounted price is printed for that item, do not create \
a separate line_items entry for the discount line itself. Instead, treat it as modifying the \
item immediately above it: compute that item's amount as the post-discount total, and lower \
confidence and note the computation in reason if the arithmetic is uncertain. amount is the only \
field that reflects the discount -- unit_price for that row must still be the original, \
pre-discount per-unit price (see the unit_price rule above), never recomputed from the \
discounted amount.

If a discount amount doesn't plausibly apply to the item immediately above it (e.g. the discount \
exceeds that item's price, or the arithmetic is clearly inconsistent with it), do not force it \
onto that item regardless. Use your best judgment about which item it actually modifies, lower \
your confidence on the affected amount, and note the ambiguity in reason -- never silently \
produce an incorrect amount just to keep the arithmetic looking clean.

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

# Step 12 optimization knob: constrains gemini-3.6-flash's internal
# "thinking" token budget, which Step 8's latency investigation identified
# as the dominant cost of a slow call (a single-item receipt spent ~1,656
# thinking tokens vs. 855 candidate tokens; receipt_026's 23-item, 2-image
# call spent 3,300-4,100). types.ThinkingConfig exposes two distinct knobs
# and it isn't yet confirmed which one gemini-3.6-flash actually honors:
# - thinking_budget: an explicit token count (0 disables thinking, -1
#   requests "automatic"; allowed ranges are model-dependent per the SDK's
#   own docstring).
# - thinking_level: a coarser MINIMAL/LOW/MEDIUM/HIGH enum, the newer
#   Gemini-3-style interface that may supersede thinking_budget for this
#   model family.
# Left as the whole types.ThinkingConfig object (not pre-committed to one
# knob) so the live comparison can try either. None (default) means
# unconstrained -- current, unchanged behavior; extract_receipt() itself
# doesn't change unless this is explicitly set to something else.
THINKING_CONFIG: types.ThinkingConfig | None = None

_GENERATE_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    response_mime_type="application/json",
    response_schema=_RESPONSE_SCHEMA,
    # Low temperature: this is an extraction task, not a creative one --
    # faithfulness to the document matters more than varied phrasing.
    temperature=0.1,
    thinking_config=THINKING_CONFIG,
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


class TokenUsage(BaseModel):
    """Cumulative Gemini token usage across every API call made for one
    extract_receipt() run, including retries -- a request that needed
    retries genuinely cost more, so this must be summed across attempts
    rather than just reflecting the last one.

    Captures every *_token_count category the SDK's usage_metadata exposes
    (confirmed by inspecting a live response), not just prompt/candidates --
    gemini-3.6-flash does internal "thinking" before answering, and those
    tokens are billed and counted in total_token_count but excluded from
    candidates_token_count. Without thoughts_token_count here, the sum of
    the fields we report doesn't reconcile with total_token_count, which
    matters for accurate cost accounting in Step 12.
    """

    prompt_token_count: int = 0
    candidates_token_count: int = 0
    thoughts_token_count: int = 0
    cached_content_token_count: int = 0
    tool_use_prompt_token_count: int = 0
    total_token_count: int = 0

    def add(self, usage: types.GenerateContentResponseUsageMetadata | None) -> None:
        if usage is None:
            return
        self.prompt_token_count += usage.prompt_token_count or 0
        self.candidates_token_count += usage.candidates_token_count or 0
        self.thoughts_token_count += usage.thoughts_token_count or 0
        self.cached_content_token_count += usage.cached_content_token_count or 0
        self.tool_use_prompt_token_count += usage.tool_use_prompt_token_count or 0
        self.total_token_count += usage.total_token_count or 0


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


def extract_receipt_with_usage(
    image_paths: str | list[str],
) -> tuple[ReceiptExtraction | ExtractionError, TokenUsage]:
    """Same as extract_receipt(), but also returns cumulative token usage
    across every API call made for this run (including retries).

    Used by the API layer (src/api/main.py) to report accurate per-request
    cost/latency -- extract_receipt() itself stays as the simple, existing
    public entry point and just discards the usage info.
    """
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    usage = TokenUsage()

    invalid_reason = _validate_input_files(image_paths)
    if invalid_reason is not None:
        return (
            ExtractionError(
                image_paths=image_paths,
                failure_stage="invalid_input",
                message=invalid_reason,
            ),
            usage,
        )

    client = genai.Client(api_key=API_KEY)
    image_parts = _load_image_parts(image_paths)

    message = USER_MESSAGE
    last_parse_error: Exception | None = None

    for _ in range(_PARSE_RETRY_COUNT + 1):
        try:
            response = _generate_with_retry(client, [*image_parts, message])
        except (errors.APIError, httpx.TimeoutException, httpx.ConnectError) as api_error:
            return (
                ExtractionError(
                    image_paths=image_paths,
                    failure_stage="api_failure",
                    message=f"Gemini API call failed: {api_error}",
                ),
                usage,
            )

        usage.add(getattr(response, "usage_metadata", None))

        if not response.text:
            # The API declined to produce output (e.g. a safety filter blocked the
            # prompt or response) rather than raising -- this is the API's call, not
            # a parsing problem, so it's an api_failure rather than a parse_failure.
            return (
                ExtractionError(
                    image_paths=image_paths,
                    failure_stage="api_failure",
                    message=f"Gemini returned no output ({_describe_empty_response(response)})",
                ),
                usage,
            )

        try:
            return ReceiptExtraction.model_validate_json(response.text), usage
        except (ValidationError, json.JSONDecodeError) as parse_error:
            last_parse_error = parse_error
            message = (
                f"{USER_MESSAGE}\n\n"
                "Your previous response failed to parse against the required schema with this "
                f"error:\n{parse_error}\n\n"
                "Correct the output and return JSON that matches the schema exactly."
            )

    return (
        ExtractionError(
            image_paths=image_paths,
            failure_stage="parse_failure",
            message=(
                f"Response did not parse into ReceiptExtraction after {_PARSE_RETRY_COUNT + 1} "
                f"attempts: {last_parse_error}"
            ),
        ),
        usage,
    )


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
    result, _usage = extract_receipt_with_usage(image_paths)
    return result


if __name__ == "__main__":
    # "014", "016", "018", 
    receipt_numbers = ["019", "025"]
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
