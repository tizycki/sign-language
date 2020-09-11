# Video-based sign-language translator

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
    --mount type=bind,source="$(pwd)"/data,target=/sign-language/data \
    --net=host \
    -e DISPLAY --runtime=nvidia openpose-custom
```

1.2.4 Example - processing images:
```
./build/examples/openpose/openpose.bin \
    --image_dir ./data/video_frames \
    --display 0 \
    --write_json ./data/openpose_output \
    --write_images ./data/openpose_output \
    --hand
```
