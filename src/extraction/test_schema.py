"""Lightweight regression check for the ReceiptExtraction schema.

Exercises the three branches of the subtotal/tax/total consistency
validator in schema.py (consistent / inconsistent / a field missing) so a
future change to that validator can't silently break one of them.

Run directly: python src/extraction/test_schema.py
"""

from datetime import date

from schema import FieldValue, LineItem, ReceiptExtraction


def _sample_line_item() -> LineItem:
    return LineItem(
        description=FieldValue[str](value="Widget", confidence=0.9, status="present"),
        quantity=FieldValue[float](value=1, confidence=0.9, status="present"),
        unit_price=FieldValue[float](value=10.0, confidence=0.9, status="present"),
        amount=FieldValue[float](value=10.0, confidence=0.9, status="present"),
    )


def _base_kwargs() -> dict:
    return dict(
        vendor_name=FieldValue[str](value="Acme Store", confidence=0.95, status="present"),
        transaction_date=FieldValue[date](value=date(2024, 1, 1), confidence=0.9, status="present"),
        receipt_number=FieldValue[str](value=None, confidence=0.0, status="not_present"),
        currency=FieldValue[str](value="IDR", confidence=0.9, status="present"),
        line_items=[_sample_line_item()],
    )


def build_consistent() -> ReceiptExtraction:
    """subtotal + tax == total exactly -> should flag True."""
    return ReceiptExtraction(
        **_base_kwargs(),
        subtotal=FieldValue[float](value=10.0, confidence=0.9, status="present"),
        tax=FieldValue[float](value=1.0, confidence=0.9, status="present"),
        total=FieldValue[float](value=11.0, confidence=0.9, status="present"),
    )


def build_inconsistent() -> ReceiptExtraction:
    """All three present, but the math doesn't add up -> should flag False."""
    return ReceiptExtraction(
        **_base_kwargs(),
        subtotal=FieldValue[float](value=10.0, confidence=0.9, status="present"),
        tax=FieldValue[float](value=1.0, confidence=0.9, status="present"),
        total=FieldValue[float](value=99.0, confidence=0.9, status="present"),
    )


def build_missing_field() -> ReceiptExtraction:
    """total is not_present -> should flag None, not True/False."""
    return ReceiptExtraction(
        **_base_kwargs(),
        subtotal=FieldValue[float](value=10.0, confidence=0.9, status="present"),
        tax=FieldValue[float](value=1.0, confidence=0.9, status="present"),
        total=FieldValue[float](value=None, confidence=0.0, status="not_present"),
    )


def main() -> None:
    scenarios = [
        ("consistent", build_consistent(), True),
        ("inconsistent", build_inconsistent(), False),
        ("missing field", build_missing_field(), None),
    ]

    for name, receipt, expected in scenarios:
        actual = receipt.subtotal_tax_total_consistent
        print(f"{name}: subtotal_tax_total_consistent = {actual!r} (expected {expected!r})")
        assert actual == expected, f"{name} scenario: expected {expected!r}, got {actual!r}"

    print("All scenarios passed.")


if __name__ == "__main__":
    main()
