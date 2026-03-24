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

# Ensure the recovery folder exists
RUN mkdir -p /apps/packmat_pwani_updated/Pwani_packmat_

# Set working directory
WORKDIR /app
COPY . .

# Upgrade pip safely
RUN python -m pip install --upgrade pip

# Install dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt

# Expose production port
EXPOSE 5005

# Run app
CMD ["uvicorn", "index4:app", "--host", "0.0.0.0", "--port", "5005"]