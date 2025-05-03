# Use official PyTorch image with CUDA / I used
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set non-interactive mode for apt% default
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies opencv's
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application code get all 
COPY . .

# Optional: set architecture flags and logging environment, default
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9"
ENV VERBOSE=1
ENV DEBUG=1

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose default FastAPI port - default but ??? we can delete 
EXPOSE 9000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
