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

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create runs dir
RUN mkdir -p runs

# Default command
CMD ["python", "train.py"]
