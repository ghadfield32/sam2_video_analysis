# Dockerfile changes
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Define ARGs
ARG ENV_NAME=data_science_sam2
ARG PYTHON_VER=3.10

# Prevent tzdata from asking for user input
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV ENV_NAME=$ENV_NAME

# Install system dependencies
RUN apt-get update --fix-missing && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    vim \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -O ~/miniconda.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p $CONDA_DIR \
    && rm ~/miniconda.sh \
    && echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate $ENV_NAME" >> ~/.bashrc \
    && find $CONDA_DIR/ -follow -type f -name '*.a' -delete \
    && find $CONDA_DIR/ -follow -type f -name '*.js.map' -delete \
    && $CONDA_DIR/bin/conda clean -afy

# Copy environment.yml and perform variable substitution
COPY .devcontainer/environment.yml /tmp/environment.yml
RUN sed -i 's/${ENV_NAME}/'"$ENV_NAME"'/g' /tmp/environment.yml && \
    sed -i 's/${PYTHON_VER}/'"$PYTHON_VER"'/g' /tmp/environment.yml

# Create Conda environment
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Install PyTorch and other packages
RUN conda run -n $ENV_NAME pip install --no-cache-dir --upgrade pip \
    && conda run -n $ENV_NAME pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 \
    && conda run -n $ENV_NAME pip install --no-cache-dir opencv-python-headless ultralytics supervision

# Verify OpenCV installation
RUN conda run -n $ENV_NAME python -c 'import cv2; print(cv2.__version__)'

# Set working directory
WORKDIR /workspace

# Set the default command to bash
CMD ["/bin/bash", "-c", "source ~/.bashrc && exec bash"]