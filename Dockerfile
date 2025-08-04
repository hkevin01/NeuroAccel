# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies and set up Python
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    python3.9 \
    python3.9-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --set python3 /usr/bin/python3.9 \
    && pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install ROCm dependencies if AMD GPUs are available
RUN if [ "$(lspci | grep -i amd)" ]; then \
    curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/trusted.gpg.d/rocm.gpg && \
    echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian jammy main' | tee /etc/apt/sources.list.d/rocm.list && \
    apt-get update && apt-get install -y rocm-dev && \
    rm -rf /var/lib/apt/lists/* ; \
    fi

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Copy source code, install package, and set up environment
COPY . .

# Install package, run tests, and create user
RUN pip3 install -e . && \
    pytest tests/ -v --tb=short && \
    useradd -m -u 1000 neuroaccel
USER neuroaccel

# Set environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV AMD_VISIBLE_DEVICES=all

EXPOSE 8000

# Start the API server
CMD ["python3", "-m", "neuroaccel.api.server"]
