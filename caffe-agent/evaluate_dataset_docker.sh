#!/bin/bash

DATABASE_ADDRESS=52.91.209.88
# DATABASE_NAME=sphereface_v1_0
# MODEL_NAME=SphereFace
DATABASE_NAME=resnet50_v1_0
MODEL_NAME=ResNet50
NUM_FILE_PARTS=10
# NUM_FILE_PARTS=-1
MODEL_VERSION=1.0
TRACE_LEVEL=MODEL_TRACE
BATCH_SIZE=(1 2 4 8 16 32 48)

for b in ${BATCH_SIZE[@]}; do
  nvidia-docker run --network host -t -v $HOME:/root carml/caffe-agent:amd64-gpu-latest predict dataset \
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
        --database_name=$DATABASE_ADDRESS_docker \
        --trace_level=$TRACE_LEVEL

  nvidia-docker run --network host -t -v $HOME:/root carml/caffe-agent:amd64-gpu-latest predict dataset \
        --fail_on_error=true \
        --verbose \
        --publish=true \
        --publish_predictions=false \
        --gpu=1 \
        --num_file_parts=$NUM_FILE_PARTS \
        --batch_size=$b \
        --model_name=$MODEL_NAME \
        --model_version=$MODEL_VERSION \
        --database_address=$DATABASE_ADDRESS \
        --database_name=$DATABASE_ADDRESS_docker \
        --trace_level=$TRACE_LEVEL
done

exit

