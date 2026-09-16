"""Trivial connectivity test: send one receipt image to Gemini and print a free-text description.

Just confirms auth, image handling, and API access work before building
structured extraction.
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(override=True)

API_KEY = os.environ["GOOGLE_API_KEY"]
IMAGE_PATH = os.path.join("Raw Data", "receipt_001.jpeg")
MODEL_NAME = "gemini-3.6-flash"


def main() -> None:
    client = genai.Client(api_key=API_KEY)

    with open(IMAGE_PATH, "rb") as f:
        image_bytes = f.read()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            "Describe what's on this receipt in 2-3 sentences.",
        ],
    )

    print(response.text)


if __name__ == "__main__":
    main()
