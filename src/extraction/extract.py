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
"""

import json
import mimetypes
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError

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


def extract_receipt(image_path: str) -> ReceiptExtraction:
    """Extract structured fields from a single receipt image.

    Sends the image + prompt to Gemini once. If the response doesn't parse
    into ReceiptExtraction, retries once with the parse error fed back into
    the prompt so the model can correct its own output.
    """
    client = genai.Client(api_key=API_KEY)

    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[image_part, USER_MESSAGE],
        config=_GENERATE_CONFIG,
    )

    try:
        return ReceiptExtraction.model_validate_json(response.text)
    except (ValidationError, json.JSONDecodeError) as first_error:
        retry_message = (
            f"{USER_MESSAGE}\n\n"
            "Your previous response failed to parse against the required schema with this "
            f"error:\n{first_error}\n\n"
            "Correct the output and return JSON that matches the schema exactly."
        )
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[image_part, retry_message],
            config=_GENERATE_CONFIG,
        )
        return ReceiptExtraction.model_validate_json(response.text)


if __name__ == "__main__":
    image_path = os.path.join("Raw Data", "receipt_002.jpeg")
    result = extract_receipt(image_path)
    output_json = result.model_dump_json(indent=2)
    print(output_json)

    output_dir = os.path.join("outputs", "sample_runs")
    os.makedirs(output_dir, exist_ok=True)
    receipt_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{receipt_name}_output.json")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_json)
