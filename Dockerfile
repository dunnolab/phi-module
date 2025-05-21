FROM cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py310:0.0.36

USER root

RUN apt-get update && apt-get install -y \
    build-essential \        
    libxc-dev \              
    libblas-dev \            
    liblapack-dev \         
    gfortran \              
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir scipy
RUN pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124 
RUN pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.4.0+cu124.html 
RUN pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.4.0+cu124.htmldo
RUN pip3 install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.4.0+cu124.html
RUN pip3 install torch-geometric torch-spline-conv
RUN pip3 install --no-cache-dir --default-timeout=300 fairchem-core

USER user