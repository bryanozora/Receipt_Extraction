import requests

# Three receipts in one batch: two single-image receipts and one two-image
# (multi-page) receipt in the middle, to exercise grouping.
receipts = [
    ["Raw Data/receipt_001.jpeg"],
    ["Raw Data/receipt_026_p1.jpeg", "Raw Data/receipt_026_p2.jpeg"],
    ["Raw Data/receipt_002.jpeg"],
]

group_sizes = ",".join(str(len(paths)) for paths in receipts)

files = []
opened = []
for paths in receipts:
    for path in paths:
        f = open(path, "rb")
        opened.append(f)
        files.append(("files", (path, f, "image/jpeg")))

try:
    response = requests.post(
        "http://localhost:8000/extract/batch",
        files=files,
        data={"group_sizes": group_sizes},
    )
finally:
    for f in opened:
        f.close()

print(response.status_code)
print(response.json())
