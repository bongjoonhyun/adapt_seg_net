#!/bin/bash

xhost +

nvidia-docker run \
  -it \
  --runtime=nvidia \
  --ipc=host \
  -v ~/adapt_seg_net/adaptation_assignment3-master:/adapt_seg_net \
  bongjoonhyun/adapt_seg_net:latest \
  /bin/bash