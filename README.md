# Crab Image Analysis using YOLOv5/v8 and EfficientNet

## Introduction

This project's code is designed for analyzing  crab (in coastal wetland) images based on YOLOv5/v8 [Ultralytics](https://github.com/ultralytics/ultralytics) and EfficientNet. It includes functionalities such as crab object detection, counting, classification, feature extraction, crab burrow detection and counting, sampling frame positioning, and biomass estimation. The project will be continuously updated to achieve full functionality.
The model weights can be found at https://doi.org/10.5281/zenodo.10776556 , please refer to the instructions for details.
## Usage

To use the code, simply copy and paste it into YOLOv5 v6.0 (Download link: [YOLOv5 v6.0](https://github.com/ultralytics/yolov5/releases/tag/v6.0)) and choose to overwrite. It is recommended to use the project's requirements.txt for installation (`pip install -r requirements.txt`). A user-friendly UI version will be updated in the future for easier operation.
The files starting with `detect-...py` and the code in the `utils.py` folder are all modified from [YOLOv5 v6.0](https://github.com/ultralytics/yolov5/releases/tag/v6.0), with explanations provided only for the modified parts. 

Analysis should follow the steps below. A cohesive code will be updated in the future:
1. Use `detect-frame.py` to detect sampling frames in the images.
2. Perform perspective cropping based on the results of step 1 to obtain images containing only the sampling frame area.
3. Use `detect-xml-2c.py` to detect crab burrows in the images.
4. Use `detect-2c-cpm.py` to detect crabs in the images and extract the width of the carapace width.
5. Utilize the results for ecological analysis.
In addition, `efn_eval_new.py` is used to evaluate the performance of the classification model in this approach, and `size-conf-test.sh` is used to test the impact of input image size and confidence threshold on prediction accuracy. The prediction results can also be found on Zenodo(https://doi.org/10.5281/zenodo.10776556).

Some parameter explanations: 
- In `detect-2c-cpm.py`, use the following commands:
    - `parser.add_argument('--classify', action='store_true', help='apply second classifier class 12,14')`
    - `parser.add_argument('--carapace', action='store_true', help='measure carapace width')`

These additions have been made to the original `detect.py` functionality. The `--classify` command is used to activate the second-stage classifier with category selection, while the `--carapace` command is used to start the segmentation model for segmenting the carapace of crustaceans and estimating the width of the carapace. This includes a fault-tolerant mechanism where, if the carapace cannot be effectively detected, the width is estimated based on the diagonal of the bounding box. For more information, refer to the article (under submission, link to be updated).

- `detect-xml-2c.py` allows only the selection of whether to call the second-stage classifier, similar to `--classify`. It can also save the results as an XML file for annotation inspection, use `--save-xml`. The classes inside are predefined; if used, please modify them to match your own trained model's classes.

- In `detect-frame.py`, a photo distortion correction is performed during data loading (based on calibration data, article link to be uploaded).

The code is partially inspired by [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) and [Efficientnet_pytorch_cbam_gui](https://github.com/whisperLiang/Efficientnet_pytorch_cbam_gui).
Attention module added from https://github.com/ZjjConan/SimAM and https://github.com/hujie-frank/SENet


Feel free to adapt the steps and customize the usage according to your specific requirements. Thank you for using our project!
