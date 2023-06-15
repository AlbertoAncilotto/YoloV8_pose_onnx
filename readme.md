# ONNX YOLOv8 Pose Detection for Embedded Devices
This repository contains code that is developed to run ONNX models of YOLOv8 pose detection on embedded devices, such as Raspberry Pi and Nvidia Jetson, with minimal dependencies and without requiring the installation of PyTorch.

## Overview
The code in this repository provides a lightweight solution for running YOLOv8 pose detection models in ONNX format on embedded devices. The code is designed to be efficient and easy to use, with minimal dependencies and no need to install PyTorch.

The repository contains a Python script that can be used to load an ONNX model file and run inference on input images or video streams. The script uses the OpenCV library to handle image input/output and drawing of detected poses.


## Dependencies
Python 3.x

numpy
python-opencv 
onnxruntime
matplotlib

## Usage
To use the code in this repository, you can follow these steps:

Clone the repository to your device: git clone https://github.com/AlbertoAncilotto/YoloV8_pose_onnx

Install the required dependencies: pip install numpy opencv-python onnxruntime matplotlib

Download an ONNX model file for YOLOv8 pose detection and place it in the root directory.

## License
The code in this repository is licensed under the MIT License. Feel free to use and modify the code as needed.