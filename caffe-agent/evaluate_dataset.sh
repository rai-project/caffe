#!/bin/bash

go build

 ./caffe-agent predict dataset --verbose \
  --publish=true --publish_predictions=false \
  --num_file_parts=100 --batch_size=32 --gpu=1 \
  --model_name=ResNet50 --model_version=1.0 \
  --database_address=192.17.102.10 --database_name=resnet_50_1.0 \
  --trace_level=MODEL_TRACE \
  --fail_on_error=true --verbose

 ./caffe-agent predict dataset --verbose \
  --publish=true --publish_predictions=false \
  --num_file_parts=100 --batch_size=32 --gpu=0 \
  --model_name=ResNet50 --model_version=1.0 \
  --database_address=192.17.102.10 --database_name=resnet_50_1.0 \
  --trace_level=MODEL_TRACE \
  --fail_on_error=true --verbose
