FROM registry.cn-hangzhou.aliyuncs.com/peterande/dfine:v1

# Set working directory
WORKDIR /workspace

# Cài đặt kaggle trong môi trường base
RUN pip install kaggle

# Clone D-FINE repository
RUN git clone https://github.com/Peterande/D-FINE.git /workspace/D-FINE

RUN . /root/miniconda3/etc/profile.d/conda.sh && \
    conda create -n dfine python=3.8 -y && \
    conda activate dfine && \
    pip install numpy==1.24.4 && \
    pip install torch>=2.0.1 torchvision>=0.15.2 --index-url https://download.pytorch.org/whl/cu117 && \
    cd /workspace/D-FINE && \
    pip install --default-timeout=600 --retries 10 \
    faster-coco-eval>=1.6.5 \
    PyYAML \
    tensorboard \
    scipy \
    calflops \
    transformers \
    loguru \
    matplotlib pillow tqdm opencv-python pycocotools && \
    conda clean -afy
    
RUN conda create -n yolo python=3.8 -y && \
    conda clean -afy && \
    . /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate yolo && \
    pip install numpy==1.24.4 && \
    pip install ultralytics

RUN echo ". /root/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc 

RUN echo '#!/bin/bash\n. /root/miniconda3/etc/profile.d/conda.sh\nconda activate dfine\nexec "$@"' > /usr/local/bin/with-dfine && \
    echo '#!/bin/bash\n. /root/miniconda3/etc/profile.d/conda.sh\nconda activate yolo\nexec "$@"' > /usr/local/bin/with-yolo && \
    chmod +x /usr/local/bin/with-dfine /usr/local/bin/with-yolo

# Thiết lập entrypoint
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["bash"]
