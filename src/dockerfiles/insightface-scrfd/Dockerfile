FROM continuumio/miniconda3:22.11.1

# see https://pythonspeed.com/articles/activate-conda-dockerfile/ for conda in docker
# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

RUN conda create --name openmmlab python=3.8 -y && \
    conda init bash && \
    conda activate openmmlab && \
    conda install pytorch==1.6.0 torchvision==0.7.0 cpuonly -c pytorch && \
    conda install -c conda-forge pycocotools==2.0.6 && \
    conda install onnx==1.13.0 scipy==1.9.3 onnxruntime==1.12.1 && \
    pip install -U openmim onnxsim==0.4.0 && \
    mim install mmcv-full==1.3.3

# RUN conda install pytorch torchvision cpuonly -c pytorch
#RUN conda install pytorch==1.6.0 torchvision==0.7.0 cpuonly -c pytorch && \
#    conda install -c conda-forge pycocotools && \
#    conda install onnx

RUN apt-get update -q && \
    apt-get install -q -y --no-install-recommends gcc libgl1-mesa-dev libc-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

#RUN pip install -U openmim && \
#    mim install mmcv-full==1.3.3

# RUN pip install cython
# RUN pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

#RUN conda activate openmmlab && \
#    pip install mmdet==2.7.0

#RUN apt-get update -q && \
#    apt-get install -q -y --no-install-recommends libgl1-mesa-dev && \
#    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN cd ~ && git clone https://github.com/deepinsight/insightface && \
    cd insightface && \
    git reset --hard f091989568cad5a0244e05be1b8d58723de210b0
RUN cd ~/insightface/detection/scrfd && \
    conda activate openmmlab && \
    sed -i '/mmpycocotools/d' ./requirements/runtime.txt && \
    pip install -r requirements/build.txt && \
    pip install -v -e .
