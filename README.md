# Estimate-6D-Pose-from-Single-RGB-Image-with-Backpropagatable-PnP

## Introduction
Object pose estimation has a wide range of applications in the field of robots and self-driving since it is essential to get precise orientations and translations of objects in these tasks. Methods that combined with depth map can get good results. However, depth map is not easy to get. As such, our project is dedicated to estimating poses of known objects using RGB images only.

Instead of directly predicting representations of poses, many methods solve the pose using Perspective-n-Points (PnP) through the correspondence be- tween 2D and 3D. But this has a disadvantage: the training of the network is not end-to-end. The pose estimated by PnP does not guide the update of front-end network parameters. The key problem is that when we use back-propagation, the gradient has no way to pass through PnP. In this project, we originally combined a backpropagatable PnP method in the training framework and introduced a projection loss, so as to achieve an end-to-end training.

Another significant challenge in pose estimation is to estimate correct poses when objects are occluded or symmet- ric. In this project, we simply ignore the rotation relative to the axis on which infinite and continuous symmetric poses exist. For the symmetric objects that have a finite number of ambiguous views, we adopt the transformer loss to handle it. We also use synthesized images and data augmen- tations to generate more train data and relieve the problem of occlusions.

To summarize, the main work of this project are: (1) Im- plement and train the Mask-RCNN for object detection and instance segmentation. (2) Implement the main framework of Pix2Pose in Pytorch and achieve the reported per- formance. (3) One major novelty of this work is through combining the backpropagatable PnP method, we achieve an end-to-end training process and get a better performance.

## Our Methods
This section provides a detailed description of our net- work architecture and loss functions for training. The tar- get of our method is to predict R and T for each object, which describes the rotation and translation of cameras in world coordinate system.

![figure1](https://github.com/WallaceSUI/Estimate-6D-Pose-from-Single-RGB-Image-with-Backpropagatable-PnP/blob/main/figures-equations/figure1.png)

![figure2](https://github.com/WallaceSUI/Estimate-6D-Pose-from-Single-RGB-Image-with-Backpropagatable-PnP/blob/main/figures-equations/figure2.png)

![figure3-4](https://github.com/WallaceSUI/Estimate-6D-Pose-from-Single-RGB-Image-with-Backpropagatable-PnP/blob/main/figures-equations/figure3-4.png)

### Object Detection
In the first stage of the pipeline, an independent Mask- RCNN is employed for 2-D object detection and in- stance segmentation. This stage takes the whole RGB image as input and will provide classes, 2-D bounding boxes and instance masks for detected objects. RGB images as well as masks of these detected objects will then be cropped according to enlarged 2-D bounding boxes in order to deal with possible occlusions. These cropped images will then be processed by lateral Unet.

### 3-D Coordinate Prediction
The architecture of the our network is described in Fig.3. We use a Unet with skip connection network architecture to make a pixel-to-pixel prediction of 3-D coor- dinates in object coordinate system.

![equation1](https://github.com/WallaceSUI/Estimate-6D-Pose-from-Single-RGB-Image-with-Backpropagatable-PnP/blob/main/figures-equations/equation1.png)

The input of the network is cropped images from the re- sults of Mask-RCNN, Iinput. The outputs of the network are normalized 3-D coordinates of each pixel in the object coordinate system, I3D, and the error level map of pixels, Ie. The ground truth Igt is a map with same size as input image Iinput and three channels. These three channels de- note the x, y, z coordinate of each pixel of Iinput in object coordinate system. A sample pair of Iinput and Igt is shown in Fig.4. Since Igt can also be seen as a image, we can gen- erate Igt by rendering 3-D model of a object with its 3-D coordinate encoded as vertex colors with same pose as its ground truth.

The target of training is to predict the target coordinate image from an input image. To evaluate the performance of the prediction, a comprehension loss function is intro- duced. The first part of the loss is called transformer loss, Ltrans, which will be illustrated in this section. The next section will elaborate the another part of loss function, the projection loss, Lproj.

To make the prediction more close to Igt, the average L1 distance of each pixel loss is used. The construction of Lr is defined as:
