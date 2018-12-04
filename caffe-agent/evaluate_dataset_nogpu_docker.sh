#!/bin/bash

DATABASE_ADDRESS=52.91.209.88
DATABASE_NAME=resnet50_v1_0_docker_framework_trace_mlperf
MODEL_NAME=ResNet50
NUM_FILE_PARTS=10
MODEL_VERSION=1.0
TRACE_LEVEL=FRAMEWORK_TRACE
BATCH_SIZE=(1)
# BATCH_SIZE=(1 2 4 8)

for b in ${BATCH_SIZE[@]}; do
  docker run --network host -t -v $HOME:/root carml/caffe-agent:amd64-cpu-mlperf-latest predict dataset \
        --fail_on_error=true \
        --verbose \
        --publish=true \
        --publish_predictions=false \
        --gpu=0 \
        --num_file_parts=$NUM_FILE_PARTS \
        --batch_size=$b \
        --model_name=$MODEL_NAME \
        --model_version=$MODEL_VERSION \
        --database_address=$DATABASE_ADDRESS \
        --database_name=$DATABASE_NAME \
        --trace_level=$TRACE_LEVEL
done

exit
