#!/bin/bash

DATABASE_ADDRESS=52.91.209.88
DATABASE_NAME=bvlc_alexnet_v1_0
# DATABASE_NAME=resnet50_v1_0
# NUM_FILE_PARTS=10
NUM_FILE_PARTS=-1
# MODEL_NAME=ResNet50
MODEL_NAME=BVLC-AlexNet
MODEL_VERSION=1.0
TRACE_LEVEL=NO_TRACE
BATCH_SIZE=1

nvidia-docker run -t -v $HOME:/root carml/caffe-agent:amd64-gpu-latest predict dataset \
      --fail_on_error=true \
      --verbose \
      --publish=true \
      --publish_predictions=false \
      --gpu=1 \
      --num_file_parts=$NUM_FILE_PARTS \
      --batch_size=$BATCH_SIZE \
      --model_name=$MODEL_NAME \
      --model_version=$MODEL_VERSION \
      --database_address=$DATABASE_ADDRESS \
      --database_name=$DATABASE_ADDRESS_docker \
      --trace_level=$TRACE_LEVEL
exit
