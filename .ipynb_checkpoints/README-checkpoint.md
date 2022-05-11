# Remote control with hand gestures
The training scripts are modfied from the [CVND---Gesture-Recognition repositoy](https://github.com/udacity/CVND---Gesture-Recognition). The original training scripts are intended  to serve as a starting point for 3D CNN gesture recognition training using the Jester dataset. The training codes are modified to train 2D CNN with temporal shift module. The 2D CNN backbone in this project is mobilenetV2_tsm, which is adapted from the paper [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/abs/1811.08383). ([Their gitub link](https://github.com/MIT-HAN-LAB/temporal-shift-module))

### Training instruction: 

1. Download the Jester gesture recoginition dataset. (Available in Kaggle)

2. Put the video files in the chosen path.

3. change the training config inside the configs directory.

4. start training by running the following command: `python train.py --config configs/config.json -g 0`

5. To start from a checkpoint, run: 'python train.py --config configs/config.json -g 0 -r True'


### Edge deployment instruction:
For Jetson Nano 2GB:

![1652235364(1)](https://user-images.githubusercontent.com/35386051/167755419-2b2faeec-e786-4790-ba45-e22f2499ba92.png)

1. Prerequisite: to run the original version, only `torch` is needed. To run the TVM version, you need to install the `TVM`, `onnx`, `onnx-simplifier`
 library. To run the TensorRT version, you need to install the `torch2trt` library.
 
2. change the host IP address inside the inference function.

3. run the inference notebook on the edge.

4. run the remote_control notebook on your PC.

### Supported Gesture:

Drumming Finger

Stop Sign

Thumb Up

Thumb Down

Swiping Left

Swiping Right

Zooming In With Full Hand

Zooming Out With Full Hand

Zooming In With Two Fingers

Zooming Out With Two Fingers

Shaking Hand