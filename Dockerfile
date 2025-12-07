# Use an NVIDIA CUDA runtime base. This image includes CUDA libs but will
# still run fine on a machine with no GPU, it just won't see any GPUs.
# Use CUDA 12.4 to match the PyTorch cu124 wheels
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# System packages including optimized BLAS libraries for CPU
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git build-essential curl ca-certificates \
    libopenblas-dev liblapack-dev libomp-dev && \
    rm -rf /var/lib/apt/lists/*


# Set OpenBLAS to use all available CPU threads for optimal performance
ENV OPENBLAS_NUM_THREADS=0
ENV OPENBLAS_MAIN_FREE=1
ENV OMP_NUM_THREADS=0


# Copy requirements and install

COPY requirements*.txt /tmp/requirements.txt




RUN pip3 install --no-cache-dir -r /tmp/requirements.txt


#RUN pip3 install torch==2.9.1+cu130 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130

# Create app dir (use /workspace so it matches the compose mount)
WORKDIR /workspace



# Copy the rest of your code
COPY . .

# Expose common dev ports
EXPOSE 5000 8888

# Install a tiny entrypoint script
RUN mkdir -p /usr/local/bin && \
    cp ./docker_entrypoint.sh /usr/local/bin/docker_entrypoint.sh && \
    chmod +x /usr/local/bin/docker_entrypoint.sh && \
    ln -s /usr/bin/python3 /usr/bin/python

ENV OMP_NUM_THREADS=1
# Default command: keep container alive; docker-compose overrides this per-service
CMD ["/usr/local/bin/docker_entrypoint.sh"]
