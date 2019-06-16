#!/bin/bash

DATABASE_ADDRESS=52.91.209.88
DATABASE_NAME=resnet50_v1_0
DUPLICATE_INPUT=160
MODEL_NAME=ResNet50
MODEL_VERSION=1.0
TRACE_LEVEL=MODEL_TRACE
BATCH_SIZE=(1 2 4 8 16 32 64 128 256 512 768)

go build

for b in ${BATCH_SIZE[@]}; do
  DUPLICATE_INPUT=$((10 * $b));
  ./caffe-agent predict url \
      --fail_on_error=true \
      --verbose \
      --publish=true\
      --publish_predictions=false \
      --gpu=1 \
      --duplicate_input=$DUPLICATE_INPUT \
      --batch_size=$b \
      --model_name=$MODEL_NAME \
      --model_version=$MODEL_VERSION \
      --database_address=$DATABASE_ADDRESS \
      --database_name=$DATABASE_NAME \
      --trace_level=$TRACE_LEVEL
done

exit
