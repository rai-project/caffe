#!/bin/bash

DATABASE_ADDRESS=52.91.209.88
DATABASE_NAME=bvlc_alexnet_v1_0
# DATABASE_NAME=resnet50_v1_0
DUPLICATE_INPUT=160
# MODEL_NAME=ResNet50
MODEL_NAME=BVLC-AlexNet
MODEL_VERSION=1.0
TRACE_LEVEL=NO_TRACE
BATCH_SIZE=16

go build

./caffe-agent predict url \
      --fail_on_error=true \
      --verbose \
      --publish=false \
      --publish_predictions=false \
      --gpu=1 \
      --duplicate_input=$DUPLICATE_INPUT \
      --batch_size=$BATCH_SIZE \
      --model_name=$MODEL_NAME \
      --model_version=$MODEL_VERSION \
      --database_address=$DATABASE_ADDRESS \
      --database_name=$DATABASE_NAME \
      --trace_level=$TRACE_LEVEL
exit