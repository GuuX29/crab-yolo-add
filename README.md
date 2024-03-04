# Crab Image Analysis using YOLOv5/v8 and EfficientNet

## Introduction

This project's code is designed for analyzing  crab (in coastal wetland) images based on YOLOv5/v8 and EfficientNet. It includes functionalities such as crab object detection, counting, classification, feature extraction, crab burrow detection and counting, sampling frame positioning, and biomass estimation. The project will be continuously updated to achieve full functionality.

## Usage

To use the code, simply copy and paste it into YOLOv5 v6.0 (Download link: [YOLOv5 v6.0](https://github.com/ultralytics/yolov5/releases/tag/v6.0)) and choose to overwrite. It is recommended to use the project's requirements.txt for installation (`pip install -r requirements.txt`). A user-friendly UI version will be updated in the future for easier operation.

Analysis should follow the steps below. A cohesive code will be updated in the future:
1. Use `detect-frame.py` to detect sampling frames in the images.
2. Perform perspective cropping based on the results of step 1 to obtain images containing only the sampling frame area.
3. Use `detect-xml-2c.py` to detect crab burrows in the images.
4. Use `detect-2c-cpm.py` to detect crabs in the images and extract the width of the cephalothorax.
5. Utilize the results for ecological analysis.

The code is partially inspired by [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) and [Efficientnet_pytorch_cbam_gui](https://github.com/whisperLiang/Efficientnet_pytorch_cbam_gui).

Feel free to adapt the steps and customize the usage according to your specific requirements. Thank you for using our project!
