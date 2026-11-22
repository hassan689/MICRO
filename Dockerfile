FROM python:3.11-slim
WORKDIR /app

# Install system dependencies required by OpenCV and DVC/Git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Default single command: validate data, verify split, run baseline, train model, register in MLflow, export logs
CMD ["python", "run_pipeline.py"]
