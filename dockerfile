# NVIDIA CUDA
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Set timezone to Kenya
ENV TZ=Africa/Nairobi
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tzdata \
        python3 \
        python3-pip \
        python3-dev \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        git \
        libsm6 \
        libxext6 && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Make python -> python3
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app
COPY . .

# Upgrade pip safely
RUN python -m pip install --upgrade pip

# Install dependencies (CRITICAL)
RUN python -m pip install --no-cache-dir -r requirements.txt

# Expose production port
EXPOSE 5005

# Run app on port 5005
CMD ["uvicorn", "index2:app", "--host", "0.0.0.0", "--port", "5005"]

