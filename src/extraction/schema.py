"""Pydantic schema for structured receipt extraction.

Design: confidence, status, and abstention
-------------------------------------------
Every extracted field is wrapped in :class:`FieldValue`, not returned as a
bare scalar. This is deliberate: an extraction system that only ever returns
a value cannot distinguish "I read this confidently" from "I guessed" from
"this isn't on the receipt at all" — and those three cases need to be
handled very differently downstream (auto-accept, flag for review, skip).

Each field therefore carries three things instead of one:

- ``value``: the extracted value, or ``None`` if it could not be read.
- ``status``: whether the field is ``"present"`` on the document, confidently
  ``"not_present"`` (e.g. no receipt number printed), or ``"illegible"``
  (present but unreadable — smudged, cropped, cut off). This is what lets
  the model *abstain* instead of hallucinating a plausible-looking value:
  "not_present" and "illegible" are both valid, honest answers with
  ``value=None``, rather than the model inventing something to fill the
  slot.
- ``confidence``: a 0-1 self-reported estimate, independent of status. A
  field can be ``"present"`` with low confidence (the digits are blurry but
  a best guess was made) — status and confidence are deliberately
  orthogonal so downstream consumers can threshold on either.
- ``reason``: a short explanation, populated only when it earns its keep —
  i.e. when confidence is low or status isn't ``"present"``. On a clean,
  confident read it stays ``None`` so the common case isn't cluttered with
  boilerplate ("clearly printed on receipt" for every single field). This
  is a generation-time convention for whatever extraction call populates
  the schema; it is not enforced as a hard validation rule here, since
  "low confidence" is a threshold the caller gets to define.

Generic vs. per-type classes
-----------------------------
Pydantic v2 (this project pins >=2.x, running 2.13 locally) supports
``Generic[T]`` models cleanly — validation, JSON schema generation, and
nested model resolution all work through ``FieldValue[str]``,
``FieldValue[float]``, etc. without special-casing. So a single generic
``FieldValue[T]`` is used here rather than separate ``StringField`` /
``FloatField`` / ``DateField`` classes, which would just duplicate the same
four attributes four times for no behavioral difference.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Generic, List, Literal, Optional, TypeVar

from pydantic import BaseModel, Field, model_validator

T = TypeVar("T")

FieldStatus = Literal["present", "not_present", "illegible"]


class FieldValue(BaseModel, Generic[T]):
    """A single extracted field, wrapped with confidence and abstention metadata."""

    value: Optional[T] = None
    confidence: float = Field(ge=0.0, le=1.0)
    status: FieldStatus
    reason: Optional[str] = None


class LineItem(BaseModel):
    """A single row of a receipt's line-item table."""

    description: FieldValue[str]
    quantity: FieldValue[float]
    unit_price: FieldValue[float]
    amount: FieldValue[float]


class ReceiptExtraction(BaseModel):
    """Top-level structured extraction result for one receipt."""

    vendor_name: FieldValue[str]
    transaction_date: FieldValue[date]
    receipt_number: FieldValue[str]
    currency: FieldValue[str]
    subtotal: FieldValue[float]
    tax: FieldValue[float]
    total: FieldValue[float]
    line_items: List[LineItem] = Field(default_factory=list)

    subtotal_tax_total_consistent: Optional[bool] = Field(
        default=None,
        description=(
            "Secondary consistency signal, computed after extraction: whether "
            "subtotal + tax approximately equals total. None when any of the "
            "three fields is missing/not present, so this is never a hard "
            "rejection rule — just a flag for the eval harness / review UI."
        ),
    )

    @model_validator(mode="after")
    def _check_subtotal_tax_total(self) -> "ReceiptExtraction":
        subtotal, tax, total = self.subtotal, self.tax, self.total

        all_present = (
            subtotal.status == "present"
            and tax.status == "present"
            and total.status == "present"
            and subtotal.value is not None
            and tax.value is not None
            and total.value is not None
        )
        if not all_present:
            self.subtotal_tax_total_consistent = None
            return self

        # Rounding/tax-calculation quirks across receipts mean an exact
        # match isn't realistic; allow a small absolute and relative slack.
        self.subtotal_tax_total_consistent = math.isclose(
            subtotal.value + tax.value, total.value, rel_tol=0.01, abs_tol=0.01
        )
        return self
