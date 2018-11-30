#!/bin/bash

DATABASE_ADDRESS=52.91.29.125
DATABASE_NAME=resnet_50_v1_0
NUM_FILE_PARTS=10
MODEL_NAME=ResNet50
MODEL_VERSION=1.0
TRACE_LEVEL=FULL_TRACE
BATCH_SIZE=32

go build

 ./caffe-agent predict dataset --verbose \
   --fail_on_error=true --verbose \
  --publish=true --publish_predictions=false --gpu=1 \
  --num_file_parts=$NUM_FILE_PARTS --batch_size=$BATCH_SIZE \
  --model_name=$MODEL_NAME --model_version=$MODEL_VERSION \
  --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME \
  --trace_level=FULL_TRACE$ \

exit
 ./caffe-agent predict dataset --verbose \
   --fail_on_error=true --verbose \
  --publish=true --publish_predictions=false --gpu=0 \
  --num_file_parts=$NUM_FILE_PARTS --batch_size=$BATCH_SIZE \
  --model_name=$MODEL_NAME --model_version=$MODEL_VERSION \
  --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME \
  --trace_level=FULL_TRACE
