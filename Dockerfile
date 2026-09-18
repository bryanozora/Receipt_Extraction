# Single image, two uses: the API service and the eval scripts. Which one
# actually runs is decided per-service in docker-compose.yml's `command:`,
# not baked in here -- this image just has the code and dependencies ready
# for either.
FROM python:3.13-slim

WORKDIR /app

# Install dependencies first so this layer is cached across code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: serve the API. The eval service overrides this command entirely
# (see docker-compose.yml) to run the eval scripts instead.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
