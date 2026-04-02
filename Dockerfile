# 1. 基础镜像：支持 RTX 30 系列的 CUDA 11.3 编译环境
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# 设置环境变量，避免交互式安装提示
ENV DEBIAN_FRONTEND=noninteractive
ENV COPPELIASIM_ROOT=/opt/CoppeliaSim
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
ENV QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
ENV PATH="/usr/bin/python3.8:$PATH"

# 2. 系统依赖项 (添加了 ca-certificates 修复 SSL 报错)
RUN apt update && apt install -y \
    ca-certificates \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libosmesa6-dev \
    libgles2-mesa-dev \
    libegl1-mesa-dev \
    ffmpeg \
    libsdl2-dev \
    wget \
    unzip \
    tar \
    xz-utils \
    python3.8 \
    python3.8-dev \
    python3-pip \
    curl \
    liblua5.3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. 设置 Python 3.8 为默认并修复 Pip (使用 3.8 专用脚本)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    curl https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3.8 && \
    python3 -m pip install "pip<24.1" "setuptools==59.5.0"

# 4. 下载并安装 CoppeliaSim 4.1.0
WORKDIR /opt
RUN wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu20_04.tar.xz \
    && tar -xvf CoppeliaSim_Player_V4_1_0_Ubuntu20_04.tar.xz \
    && mv CoppeliaSim_Player_V4_1_0_Ubuntu20_04 CoppeliaSim \
    && rm CoppeliaSim_Player_V4_1_0_Ubuntu20_04.tar.xz

# 5. 安装依赖仓库 (PyRep, RLBench, YARR)
WORKDIR /app
RUN git clone https://github.com/stepjam/PyRep.git && cd PyRep && pip install -r requirements.txt && pip install .
RUN git clone -b peract https://github.com/MohitShridhar/RLBench.git && cd RLBench && pip install -r requirements.txt && python3 setup.py develop

# 核心修正：安装 YARR 并强制锁定冲突包版本
RUN git clone -b peract https://github.com/MohitShridhar/YARR.git && \
    cd YARR && \
    pip install "moviepy<2.0.0" "numpy<1.25.0" "omegaconf==2.0.6" "antlr4-python3-runtime==4.8" && \
    pip install -r requirements.txt && python3 setup.py develop

# 6. 安装 PerAct 主仓库与 PyTorch 1.11.0 (匹配 PyTorch3D 链接)
RUN git clone https://github.com/peract/peract.git && \
    cd peract && \
    pip install git+https://github.com/openai/CLIP.git && \
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# 7. 精准安装 PyTorch3D (预编译包)
RUN pip install fvcore iopath && \
    pip install --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html

# 8. 安装 PerAct 剩余依赖 (手动预装 transformers 避开 setup.py 的 SSL 限制)
RUN cd /app/peract && \
    pip install "transformers==4.3.2" \
                "absl-py==0.15.0" \
                "einops==0.3.2" \
                "hydra-core==1.0.5" \
                "matplotlib" \
                "pandas==1.4.1" \
                "pyrender==0.1.45" \
                "scipy==1.4.1" \
                "trimesh==3.9.34" && \
    # 使用 pip install -e 代替 setup.py develop，减少联网依赖错误
    pip install -e .

# 9. YOLH: 安装额外依赖
RUN pip install tensorboard

WORKDIR /app/peract
