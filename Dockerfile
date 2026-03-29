FROM python:3.11-slim

WORKDIR /app

# System deps for rendering (pygame/SDL needs these even in headless mode)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libsdl2-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.6+ with CUDA 12.6 — required for Blackwell (RTX 50xx, sm_120)
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# Install remaining Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create runs dir
RUN mkdir -p runs

# Default command
CMD ["python", "train.py"]
