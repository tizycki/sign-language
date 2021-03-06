# Video-based sign-language translator

Project in early development stage. First sprint results: 

https://www.youtube.com/watch?v=7gSr-jBhM1c&ab_channel=TomaszI%C5%BCycki

## 1. Installation
### 1.1 Environment
```
pip install -r requirements.txt
```
### 1.2 OpenPose docker setup
1.2.1 Prepare openpose docker image prerequisites:
https://github.com/gormonn/openpose-docker

1.2.2 Pull openpose docker image:
```
docker build -t openpose-custom .
```
1.2.3 Run container:

```
docker run \
    -it \
    --rm \
    --mount type=bind,source="$(pwd)"/data,target=/sign-language/data \
    --mount type=bind,source="$(pwd)"/utils,target=/sign-language/utils \
    --net=host \
    -e DISPLAY --runtime=nvidia openpose-custom
```

1.2.4 Example - processing images:
```
bash ./build/examples/openpose/openpose.bin \
    --image_dir ./data/video_frames \
    --display 0 \
    --write_json ./data/openpose_output \
    --write_images ./data/openpose_output \
    --hand
```
```
python3 ./utils/openpose_wrapper.py \
    --image_dir ./data/video_frames/internal_resource_2337_fps10 \
    --write_json ./data/keypoints/internal_resource_2337_fps10 \
    --write_images ./data/rendered_images/internal_resource_2337_fps10
```
