"""Manual verification script for Step 5 resilience handling.

Run directly: python src/extraction/test_resilience.py

Scenarios 1, 2, and 4 run automatically and need no internet interruption.
Scenario 3 requires you to manually disconnect your internet connection
when prompted, since it tests a real transient network failure.
"""

import os

import extract
from extract import ExtractionError, extract_receipt


def scenario_1_missing_file():
    print("\n=== Scenario 1: missing file ===")
    result = extract_receipt("Raw Data/does_not_exist.jpeg")
    print(result)
    assert isinstance(result, ExtractionError)
    assert result.failure_stage == "invalid_input"
    print("PASS: correctly returned invalid_input")


def scenario_2_corrupt_file():
    print("\n=== Scenario 2: corrupt file ===")
    corrupt_path = "Raw Data/corrupt_test.jpeg"
    with open(corrupt_path, "w") as f:
        f.write("this is not a real image")

    try:
        result = extract_receipt(corrupt_path)
        print(result)
        assert isinstance(result, ExtractionError)
        assert result.failure_stage == "invalid_input"
        print("PASS: correctly returned invalid_input")
    finally:
        os.remove(corrupt_path)


def scenario_3_network_failure():
    print("\n=== Scenario 3: transient network failure ===")
    input(
        "Turn off your WiFi/internet connection now, then press Enter to continue "
        "(this will take ~7+ seconds while it retries)..."
    )
    result = extract_receipt(os.path.join("Raw Data", "receipt_001.jpeg"))
    print(result)
    assert isinstance(result, ExtractionError)
    assert result.failure_stage == "api_failure"
    print("PASS: correctly returned api_failure after retries")
    input("Turn your internet back on, then press Enter to finish...")


def scenario_4_blocked_response():
    print("\n=== Scenario 4: blocked/empty response (e.g. safety filter) ===")
    # A real safety block can't be reliably triggered on demand, so this mocks
    # the client to return a response whose .text is None, the way the SDK
    # represents a blocked prompt/response instead of raising an APIError.

    class FakeFinishReason:
        name = "SAFETY"

        def __repr__(self):
            return "FinishReason.SAFETY"

    class FakeCandidate:
        finish_reason = FakeFinishReason()

    class FakeResponse:
        text = None
        candidates = [FakeCandidate()]
        prompt_feedback = None

    class FakeModels:
        def generate_content(self, **kwargs):
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key=None):
            self.models = FakeModels()

    original_client_cls = extract.genai.Client
    extract.genai.Client = FakeClient
    try:
        result = extract_receipt(os.path.join("Raw Data", "receipt_001.jpeg"))
    finally:
        extract.genai.Client = original_client_cls

    print(result)
    assert isinstance(result, ExtractionError)
    assert result.failure_stage == "api_failure"
    print("PASS: correctly returned api_failure instead of raising on response.text is None")


if __name__ == "__main__":
    scenario_1_missing_file()
    scenario_2_corrupt_file()
    scenario_3_network_failure()
    scenario_4_blocked_response()
    print("\nAll resilience scenarios passed.")