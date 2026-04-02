#!/bin/bash

# 允许 Docker 访问宿主机显示器 (用于仿真界面)
xhost +local:docker > /dev/null

# 启动容器
docker run -it \
    --gpus all \
    --net=host \
    --privileged \
    --shm-size=16gb \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/EECS467/peract:/app/peract \
    -v ~/EECS467/train.py:/app/train.py \
    -v ~/EECS467/data:/app/data \
    --name peract_dev \
    peract /bin/bash
