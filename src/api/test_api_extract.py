import requests

with open("Raw Data/receipt_001.jpeg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/extract",
        files={"files": f}
    )

print(response.status_code)
print(response.json())