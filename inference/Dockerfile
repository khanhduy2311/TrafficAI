FROM ultralytics/ultralytics:latest-jetson-jetpack6

# Set the environment variable to prevent interactive prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive

# Copy all your project files into the container's workspace
WORKDIR /app
COPY ./FE8K .

# --- BUILD AND INSTALL ALL DEPENDENCIES IN A SINGLE LAYER ---
# This is the robust and optimized way to install packages.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # System dependencies for building C++
        build-essential \
        ninja-build \
        pkg-config \
        # System dependencies for your code
        libopencv-dev \
    && \
    # Install Python packages
    pip install --no-cache-dir \
        pycocotools \
        ensemble-boxes \
    && \
    # --- CLEANUP ---
    # Remove apt cache to keep the image size down
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN bash ./build_extensions.sh

RUN bash ./build_engine.sh

# Set the default command to run when the container starts
# This is where you would put your final inference command.
CMD ["bash"]
