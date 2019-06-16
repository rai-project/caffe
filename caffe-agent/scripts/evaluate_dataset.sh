#!/bin/bash

DATABASE_ADDRESS=52.91.209.88
DATABASE_NAME=resnet50_v1_0
MODEL_NAME=ResNet50
NUM_FILE_PARTS=-1
MODEL_VERSION=1.0
TRACE_LEVEL=NO_TRACE
BATCH_SIZE=48

go build

./caffe-agent predict dataset \
      --fail_on_error=true \
      --verbose \
      --publish=false \
      --publish_predictions=false \
      --gpu=1 \
      --num_file_parts=$NUM_FILE_PARTS \
      --batch_size=$BATCH_SIZE \
      --model_name=$MODEL_NAME \
      --model_version=$MODEL_VERSION \
      --database_address=$DATABASE_ADDRESS \
      --database_name=$DATABASE_NAME \
      --trace_level=$TRACE_LEVEL

exit
