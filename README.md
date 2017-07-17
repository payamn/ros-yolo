# ROS YOLOv2 integration
Integrates [YOLOv2](http://pjreddie.com/darknet/yolo/) with ROS, generating messages with position and confidence of detection.

## Requirements
* ROS (tested on Jade and Kinetic)

* GPU supporting CUDA

## Installation
This repository should be cloned to the `src` directory of your workspace. Alternatively, the following `.rosinstall` entry adds this repository for usage with `wstool`:

```
- git:
    local-name: yolo2
    uri: https://github.com/ThundeRatz/ros_yolo2.git
```

## Setup
Copy your network config to `data/yolo.cfg` and network weights as `data/yolo.weights`.

## Message format

For every image received, an [ImageDetections](msg/ImageDetections.msg) message is generated at the `yolo2_detections` topic, containing a Header and a vector of [Detection](msg/Detection.msg). The header time stamp is copied from the received image message; this way, a set of detections can be matched to a specific image. Each [Detection](msg/Detection.msg) contains the following attributes:

* `uint32 class_id` - Class number configured during training;
* `float32 confidence` - Confidence;
* `float32 x` - X position of the object's center relative to the image (between 0 = left and 1 = right);
* `float32 y` - Y position of the object's center relative to the image (between 0 = top and 1 = bottom);
* `float32 width` - Width of the object relative to the image (between 0 and 1);
* `float32 height` - Height of the object relative to the image (between 0 and 1).

## Contributors
Written by the ThundeRatz robotics team.
