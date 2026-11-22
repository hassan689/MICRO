FROM python:3.11-slim
WORKDIR /app
# Install system dependencies required by OpenCV and DVC/Git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# Install CPU-only PyTorch wheels first to avoid downloading multi-GB CUDA libraries
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Default single command: validate data, verify split, run baseline, train model, register in MLflow, export logs
CMD ["python", "run_pipeline.py", "--smoke-test", "--no-pretrained"]
